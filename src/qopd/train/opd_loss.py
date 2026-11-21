"""Sampled-token reverse-KL policy-gradient loss for on-policy distillation (PLAN 6.1).

Per response token t (all log-probs of the realized token y_t):

    r_t = log p_T(y_t) - log p_S(y_t)                     (stop-gradient)
    L   = - sum_{b,t} m_bt * r_bt * log p_S(y_bt) / N     (N = response tokens in the batch)

Its expectation over y ~ p_S has gradient ∇ KL(p_S || p_T), the reverse KL. N is the number
of response tokens in the *whole* optimizer batch (doc: "mean over all response tokens in the
batch"); pass ``n_tokens_total`` when the batch is split into micro-batches for gradient
accumulation, so the summed micro-batch gradients equal the full-batch gradient.

Options (each is exactly the base loss when disabled):

- ``clip_tau``: r_t <- clip(r_t, -tau, tau)                        (E3: tau = 2, 5)
- ``top_k_kl``: replace the sampled-token term by the analytic reverse KL over the teacher's
  top-k tokens at each position, with student and teacher renormalized on that set:
  KL_t = sum_{k in K_t} p~_S(k) (log p~_S(k) - log p~_T(k))          (E3 "top-k smoothing")
- ``hybrid_weight`` lambda: r_t <- r_t + lambda * R_b, the sequence's verifier reward added to
  every response token of that sequence                               (E3 "hybrid")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class OPDLossCfg:
    clip_tau: float | None = None  # None = no clipping
    top_k_kl: bool = False  # True = analytic KL over the teacher top-k (needs top-k inputs)
    hybrid_weight: float = 0.0  # lambda for the verifier reward


@dataclass
class OPDLossOutput:
    loss: torch.Tensor
    reward_mean: float  # over response tokens, after clip/hybrid
    reward_std: float
    clip_frac: float  # share of tokens whose reward was clipped
    n_tokens: int


def gather_logprobs(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """log softmax(logits)[..., target] in float32. logits [B,T,V], targets [B,T] -> [B,T]."""
    import torch  # noqa: PLC0415

    lp = torch.log_softmax(logits.float(), dim=-1)
    return lp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)


def token_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax(logits) per position (float32), [B,T,V] -> [B,T]."""
    import torch  # noqa: PLC0415

    lp = torch.log_softmax(logits.float(), dim=-1)
    return -(lp.exp() * lp).sum(-1)


def per_token_reward(
    teacher_logprobs: torch.Tensor,  # [B, T]
    student_logprobs: torch.Tensor,  # [B, T]
    mask: torch.Tensor,  # [B, T] 1 = response token
    clip_tau: float | None = None,
    verifier_reward: torch.Tensor | None = None,  # [B]
    hybrid_weight: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(r_t detached and zero where masked, bool tensor of clipped positions)."""
    import torch  # noqa: PLC0415

    m = mask.to(torch.float32)
    r = (teacher_logprobs.float() - student_logprobs.detach().float()) * m
    if hybrid_weight:
        if verifier_reward is None:
            raise ValueError("hybrid_weight > 0 needs verifier_reward")
        r = r + hybrid_weight * verifier_reward.float()[:, None] * m
    clipped = torch.zeros_like(m, dtype=torch.bool)
    if clip_tau is not None:
        clipped = (r.abs() > clip_tau) & mask.bool()
        r = r.clamp(-clip_tau, clip_tau)
    return r.detach(), clipped


def topk_reverse_kl(
    student_topk_logprobs: torch.Tensor,  # [B, T, k] full-vocab student log-probs at the ids
    teacher_topk_logprobs: torch.Tensor,  # [B, T, k] teacher log-probs at the same ids
) -> torch.Tensor:
    """KL(p~_S || p~_T) per position over the k ids, both renormalized: [B, T]."""
    import torch  # noqa: PLC0415

    ls = torch.log_softmax(student_topk_logprobs.float(), dim=-1)  # renormalize on K
    lt = torch.log_softmax(teacher_topk_logprobs.float(), dim=-1)
    return (ls.exp() * (ls - lt)).sum(-1)


def opd_loss(
    student_logprobs: torch.Tensor,  # [B, T] log p_S(y_t), requires grad
    teacher_logprobs: torch.Tensor,  # [B, T] log p_T(y_t)
    mask: torch.Tensor,  # [B, T] 1 = response token
    cfg: OPDLossCfg,
    *,
    verifier_reward: torch.Tensor | None = None,  # [B], hybrid only
    student_topk_logprobs: torch.Tensor | None = None,  # [B, T, k], top-k KL only
    teacher_topk_logprobs: torch.Tensor | None = None,  # [B, T, k], top-k KL only
    n_tokens_total: int | None = None,  # response tokens in the whole optimizer batch
) -> OPDLossOutput:
    import torch  # noqa: PLC0415

    m = mask.to(torch.float32)
    n_local = int(m.sum().item())
    denom = float(n_tokens_total if n_tokens_total is not None else max(n_local, 1))
    r, clipped = per_token_reward(
        teacher_logprobs,
        student_logprobs,
        mask,
        cfg.clip_tau,
        verifier_reward,
        cfg.hybrid_weight,
    )
    if cfg.top_k_kl:
        if student_topk_logprobs is None or teacher_topk_logprobs is None:
            raise ValueError("top_k_kl needs student_topk_logprobs and teacher_topk_logprobs")
        per_tok = topk_reverse_kl(student_topk_logprobs, teacher_topk_logprobs) * m
        if cfg.hybrid_weight:  # verifier term stays a policy-gradient term
            assert verifier_reward is not None
            per_tok = per_tok - cfg.hybrid_weight * (
                verifier_reward.float()[:, None] * m * student_logprobs.float()
            )
    else:
        per_tok = -(r * student_logprobs.float()) * m
    loss = per_tok.sum() / denom

    sel = r[mask.bool()]
    return OPDLossOutput(
        loss=loss,
        reward_mean=float(sel.mean()) if sel.numel() else float("nan"),
        reward_std=float(sel.std()) if sel.numel() > 1 else float("nan"),
        clip_frac=float(clipped.sum()) / max(n_local, 1),
        n_tokens=n_local,
    )


def loss_cfg_from(model: object) -> OPDLossCfg:
    """Build from ``qopd.config.OPDLossCfgModel`` (configs/train/opd.yaml ``loss:``)."""
    return OPDLossCfg(
        clip_tau=getattr(model, "clip_tau", None),
        top_k_kl=bool(getattr(model, "top_k_kl", False)),
        hybrid_weight=float(getattr(model, "hybrid_weight", 0.0)),
    )
