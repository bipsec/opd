"""PLAN 6.1: OPD loss on toy distributions (CPU torch)."""

import pytest

torch = pytest.importorskip("torch")

from qopd.config import CONFIG_ROOT, OPDCfg, load  # noqa: E402
from qopd.train.opd_loss import (  # noqa: E402
    OPDLossCfg,
    gather_logprobs,
    loss_cfg_from,
    opd_loss,
    per_token_reward,
    token_entropy,
    topk_reverse_kl,
)

V = 6
BASE = OPDLossCfg()


def toy(seed=0):
    g = torch.Generator().manual_seed(seed)
    theta = torch.randn(V, generator=g, requires_grad=True)  # student logits
    q = torch.log_softmax(torch.randn(V, generator=g), -1)  # teacher log-probs
    return theta, q


def analytic_kl_grad(theta, q):
    th = theta.detach().clone().requires_grad_(True)
    lp = torch.log_softmax(th, -1)
    (lp.exp() * (lp - q)).sum().backward()
    return th.grad


def test_mc_gradient_matches_reverse_kl():
    """E_{y~p}[grad of sampled loss] == grad KL(p || q)."""
    theta, q = toy()
    p = torch.softmax(theta.detach(), -1)
    n = 200_000
    y = torch.multinomial(p, n, replacement=True, generator=torch.Generator().manual_seed(1))
    lp = torch.log_softmax(theta, -1)[y][None]  # [1, N]
    out = opd_loss(lp, q[y][None], torch.ones(1, n), BASE)
    out.loss.backward()
    assert torch.allclose(theta.grad, analytic_kl_grad(theta, q), atol=2e-2)


def test_sign_convention():
    """Teacher prefers the sampled token -> positive reward -> its probability goes up."""
    theta = torch.zeros(V, requires_grad=True)
    lp = torch.log_softmax(theta, -1)
    teacher = torch.full((V,), -10.0)
    teacher[2] = -0.01
    out = opd_loss(lp[[2]][None], teacher[[2]][None], torch.ones(1, 1), BASE)
    assert out.reward_mean > 0
    out.loss.backward()
    assert theta.grad[2] < 0  # gradient descent raises logit 2


def test_masking_prompt_and_padding():
    theta, q = toy()
    lp = torch.log_softmax(theta, -1)
    ids = torch.tensor([[1, 2, 3, 4, 0]])  # pretend 2 prompt tokens, 2 response, 1 pad
    mask = torch.tensor([[0, 0, 1, 1, 0]])
    full = opd_loss(lp[ids], q[ids], mask, BASE)
    junk_teacher = q[ids].clone()
    junk_teacher[0, [0, 1, 4]] = 99.0  # changes outside the mask must not matter
    other = opd_loss(lp[ids], junk_teacher, mask, BASE)
    assert full.n_tokens == 2
    assert torch.allclose(full.loss, other.loss)
    r, _ = per_token_reward(q[ids], lp[ids], mask)
    assert torch.all(r[mask == 0] == 0)


def test_reward_is_stop_gradient():
    theta, q = toy()
    lp = torch.log_softmax(theta, -1)[None]
    r, _ = per_token_reward(q[None], lp, torch.ones(1, V))
    assert not r.requires_grad


def test_disabled_options_equal_base():
    theta, q = toy()
    lp = torch.log_softmax(theta, -1)[None]
    m = torch.ones(1, V)
    base = opd_loss(lp, q[None], m, BASE).loss
    for cfg in (
        OPDLossCfg(clip_tau=None),
        OPDLossCfg(hybrid_weight=0.0),
        OPDLossCfg(top_k_kl=False),
    ):
        assert torch.allclose(
            opd_loss(lp, q[None], m, cfg, verifier_reward=torch.ones(1)).loss, base
        )
    big_tau = opd_loss(lp, q[None], m, OPDLossCfg(clip_tau=1e6)).loss  # clip never binds
    assert torch.allclose(big_tau, base)


def test_clipping_caps_reward():
    lp = torch.log(torch.tensor([[0.5, 0.5]]))
    teacher = torch.tensor([[-20.0, 0.0]])
    out = opd_loss(lp, teacher, torch.ones(1, 2), OPDLossCfg(clip_tau=2.0))
    r, clipped = per_token_reward(teacher, lp, torch.ones(1, 2), clip_tau=2.0)
    assert r.abs().max() <= 2.0 and clipped.tolist() == [[True, False]]
    assert out.clip_frac == 0.5


def test_topk_full_vocab_equals_exact_kl_gradient():
    theta, q = toy()
    lp_all = torch.log_softmax(theta, -1)
    out = opd_loss(
        lp_all[[0]][None],
        q[[0]][None],
        torch.ones(1, 1),
        OPDLossCfg(top_k_kl=True),
        student_topk_logprobs=lp_all[None, None],
        teacher_topk_logprobs=q[None, None],
    )
    out.loss.backward()
    assert torch.allclose(theta.grad, analytic_kl_grad(theta, q), atol=1e-6)


def test_topk_renormalizes_on_support():
    s = torch.log(torch.tensor([[[0.2, 0.2]]]))  # same shape, different mass outside K
    t = torch.log(torch.tensor([[[0.4, 0.4]]]))
    assert torch.allclose(topk_reverse_kl(s, t), torch.zeros(1, 1), atol=1e-7)


def test_hybrid_adds_verifier_reward_per_sequence():
    lp = torch.log(torch.full((2, 3), 0.5))
    teacher = lp.clone()  # r_t = 0 without the verifier
    r, _ = per_token_reward(
        teacher, lp, torch.ones(2, 3), verifier_reward=torch.tensor([1.0, 0.0]), hybrid_weight=0.5
    )
    assert r[0].tolist() == [0.5] * 3 and r[1].tolist() == [0.0] * 3
    with pytest.raises(ValueError):
        opd_loss(lp, teacher, torch.ones(2, 3), OPDLossCfg(hybrid_weight=0.5))


def test_microbatch_accumulation_equals_full_batch():
    g = torch.Generator().manual_seed(3)
    theta = torch.randn(4, V, generator=g, requires_grad=True)
    teacher = torch.log_softmax(torch.randn(4, 5, V, generator=g), -1)
    ids = torch.randint(0, V, (4, 5), generator=g)
    mask = (torch.rand(4, 5, generator=g) > 0.3).long()

    def lps(rows):
        logits = theta[rows][:, None, :].expand(-1, 5, -1)
        return gather_logprobs(logits, ids[rows]), teacher[rows].gather(-1, ids[rows][..., None])[
            ..., 0
        ]

    s, t = lps(slice(0, 4))
    opd_loss(s, t, mask, BASE).loss.backward()
    full = theta.grad.clone()
    theta.grad = None
    total = int(mask.sum())
    for rows in (slice(0, 2), slice(2, 4)):
        s, t = lps(rows)
        opd_loss(s, t, mask[rows], BASE, n_tokens_total=total).loss.backward()
    assert torch.allclose(theta.grad, full, atol=1e-6)


def test_helpers():
    logits = torch.tensor([[[0.0, 0.0]]])
    assert torch.allclose(token_entropy(logits), torch.log(torch.tensor([[2.0]])))
    assert torch.allclose(
        gather_logprobs(logits, torch.tensor([[1]])), torch.log(torch.tensor([[0.5]]))
    )


def test_cfg_from_yaml():
    cfg = loss_cfg_from(load(CONFIG_ROOT / "train" / "opd.yaml", cls=OPDCfg).loss)
    assert cfg == OPDLossCfg()
