"""PLAN 6.2: OPD round loop on CPU with a tiny on-policy student and a fixed tiny teacher."""

import copy

import pytest

torch = pytest.importorskip("torch")

from qopd.config import CONFIG_ROOT, OPDCfg, load  # noqa: E402
from qopd.train.opd_loop import (  # noqa: E402
    LoopComponents,
    percentiles,
    repetition_rate,
    round_prompts,
    run_opd,
    warmup_constant,
)
from qopd.types import FinishReason, Problem, Rollout, TeacherScores  # noqa: E402
from qopd.utils import io  # noqa: E402
from qopd.utils.run import create_run_dir  # noqa: E402

V, RESP_LEN = 16, 5


def tiny(seed):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(seed)
    c = Qwen3Config(
        vocab_size=V,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        max_position_embeddings=64,
        tie_word_embeddings=True,
    )
    return Qwen3ForCausalLM(c).float()


class FakeEngine:
    """Samples from its own copy of the student; weights arrive only through restart(path)."""

    def __init__(self, init_model):
        self.net = copy.deepcopy(init_model).eval()
        self.running, self.model_path, self.starts = False, "init", 0
        self.log = []

    def start(self):
        self.running = True
        self.starts += 1
        self.log.append("engine+")

    def stop(self):
        if self.running:
            self.log.append("engine-")
        self.running = False

    def restart(self, path, revision=None):
        was = self.running
        self.net.load_state_dict(torch.load(path, weights_only=True))
        self.model_path = path
        self.running = was

    @torch.no_grad()
    def generate(self, prompt_ids, sampling, n=1, problem_ids=None, seed=None, seeds=None):
        assert self.running
        out = []
        for p, pid, s in zip(prompt_ids, problem_ids, seeds, strict=True):
            g = torch.Generator().manual_seed(s)
            for k in range(n):
                seq, lps = list(p), []
                for _ in range(RESP_LEN):
                    lp = torch.log_softmax(self.net(torch.tensor([seq])).logits[0, -1], -1)
                    t = int(torch.multinomial(lp.exp(), 1, generator=g))
                    seq.append(t)
                    lps.append(float(lp[t]))
                out.append(Rollout(list(p), seq[len(p) :], lps, FinishReason.LENGTH, pid, k))
        return out, None


class FakeTeacher:
    SHARP = 3.0  # peaked target distribution, far from the near-uniform random student

    def __init__(self, seed=99):
        self.net, self.running, self.log = tiny(seed).eval(), False, None

    def logprobs(self, x):
        return torch.log_softmax(self.SHARP * self.net(x).logits, -1)

    def start(self):
        self.running = True
        if self.log is not None:
            self.log.append("teacher+")

    def stop(self):
        if self.running and self.log is not None:
            self.log.append("teacher-")
        self.running = False

    @torch.no_grad()
    def score(self, rollouts, top_k):
        assert self.running
        res = []
        for r in rollouts:
            lp = self.logprobs(torch.tensor([r.input_ids]))[0]
            rows = lp[len(r.prompt_ids) - 1 : len(r.input_ids) - 1]
            tok = [float(rows[i, t]) for i, t in enumerate(r.response_ids)]
            v, ix = rows.topk(top_k, -1)
            res.append(TeacherScores(tok, ix.tolist(), v.tolist()))
        return res


class FileSync:
    def __init__(self, d):
        self.d = d

    def sync(self, engine, model, step):
        self.d.mkdir(parents=True, exist_ok=True)
        path = str(self.d / f"sync_{step}.pt")
        torch.save(model.state_dict(), path)
        engine.restart(path)
        return 0.0


PROBLEMS = [Problem(f"p{i}", f"q{i}", "1", "t") for i in range(40)]


def prompt_of(p):
    return [1 + int(p.id[1:]) % 7, 9]


def cfg(**kw):
    base = load(CONFIG_ROOT / "train" / "opd.yaml", cls=OPDCfg)
    upd = {
        "lr": 5e-3,
        "prompts_per_round": 4,
        "rollouts_per_prompt": 2,
        "rounds": 6,
        "checkpoint_every": 3,
        "warmup_rounds": 1,
        "teacher_top_k": 4,
        "micro_batch_size": 3,
        "rollout": base.rollout.model_copy(update={"max_tokens": RESP_LEN}),
        **kw,
    }
    return base.model_copy(update=upd)


def comps(tmp, student, teacher=None):
    return LoopComponents(
        model=student,
        engine=FakeEngine(student),
        teacher=teacher or FakeTeacher(),
        weight_sync=FileSync(tmp / "sync"),
        problems=PROBLEMS,
        prompt_ids=prompt_of,
        pad_id=0,
        device="cpu",
    )


def exact_kl(student, teacher):
    """Mean KL(student || teacher) of the first response token, over every training prompt."""
    with torch.no_grad():
        tot = 0.0
        for a in range(1, 8):
            x = torch.tensor([[a, 9]])
            ls = torch.log_softmax(student(x).logits[0, -1], -1)
            lt = teacher.logprobs(x)[0, -1]
            tot += float((ls.exp() * (ls - lt)).sum())
        return tot / 7


# --- helpers --------------------------------------------------------------------------------


def test_repetition_rate_definition():
    loop = [1, 2, 3, 4, 5, 6, 7, 8] * 3
    assert repetition_rate([loop], n=8) == 1.0
    assert repetition_rate([list(range(30))], n=8) == 0.0
    assert repetition_rate([[1] * 10], n=8) == 0.0  # shorter than 2n: not counted
    assert repetition_rate([list(range(20)) + list(range(12, 20))], n=8) == 1.0  # tail seen earlier


def test_round_prompts_deterministic_no_repeats_first_pass():
    seen = [p.id for r in range(10) for p in round_prompts(PROBLEMS, r, 4, seed=0)]
    assert len(set(seen)) == 40
    assert round_prompts(PROBLEMS, 3, 4, 0) == round_prompts(list(reversed(PROBLEMS)), 3, 4, 0)


def test_schedule_and_percentiles():
    f = warmup_constant(5)
    assert [f(i) for i in range(7)] == [0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0]
    assert percentiles(list(range(1, 101)))["p50"] == 50


# --- loop -----------------------------------------------------------------------------------


def test_co_resident_loop_logs_and_checkpoints(tmp_path):
    run = create_run_dir("opd", "r", {"x": 1}, root=tmp_path)
    c = comps(tmp_path, tiny(0))
    st = run_opd(cfg(), run, c, "co_resident")
    rows = io.read_jsonl(run.metrics)
    assert st["status"] == "trained" and len(rows) == 6
    for k in (
        "reward_mean",
        "reward_std",
        "reward_p5",
        "reward_p95",
        "entropy_mean",
        "response_len_mean",
        "truncation_rate",
        "repetition_rate",
        "round_seconds",
        "peak_mem_used_gb",
        "student_logprob_mismatch",
        "lr",
    ):
        assert k in rows[0], k
    assert set(rows[0]["seconds"]) == {"sample", "score", "train", "sync"}
    assert rows[0]["lr"] == pytest.approx(5e-3)  # warmup 1 round
    assert rows[0]["student_logprob_mismatch"] < 1e-4  # engine weights == trainer weights
    assert (run.checkpoints / "trainer_state.pt").exists()
    assert not c.engine.running and c.engine.starts == 1


def test_on_policy_distillation_reduces_reverse_kl(tmp_path):
    run = create_run_dir("opd", "kl", {"x": 1}, root=tmp_path)
    student, teacher = tiny(0), FakeTeacher()
    before = exact_kl(student, teacher)
    run_opd(
        cfg(rounds=40, checkpoint_every=40, lr=1e-2),
        run,
        comps(tmp_path, student, teacher),
        "co_resident",
    )
    after = exact_kl(student, teacher)
    assert after < 0.6 * before, (before, after)


def test_resume_matches_uninterrupted(tmp_path):
    a = create_run_dir("opd", "a", {"x": 1}, root=tmp_path)
    s1 = tiny(0)
    run_opd(cfg(), a, comps(tmp_path / "a", s1), "co_resident")

    b = create_run_dir("opd", "b", {"x": 1}, root=tmp_path)
    s2 = tiny(0)
    st = run_opd(cfg(), b, comps(tmp_path / "b", s2), "co_resident", stop_after_rounds=4)
    assert st["round"] == 4
    s3 = tiny(7)  # fresh process: different init + fresh engine; resume must restore both
    run_opd(cfg(), b, comps(tmp_path / "b", s3), "co_resident")
    p1 = torch.cat([p.detach().flatten() for p in s1.parameters()])
    p3 = torch.cat([p.detach().flatten() for p in s3.parameters()])
    assert torch.allclose(p1, p3, atol=1e-5)
    assert [r["round"] for r in io.read_jsonl(b.metrics)] == list(range(1, 7))


def test_phase_alternating_never_coresident(tmp_path):
    run = create_run_dir("opd", "pa", {"x": 1}, root=tmp_path)
    c = comps(tmp_path, tiny(0))
    events = []
    c.engine.log = events
    c.teacher.log = events
    run_opd(cfg(rounds=3), run, c, "phase_alternating")
    assert events == ["engine+", "engine-", "teacher+", "teacher-"] * 3


def test_topk_and_clip_variants_run(tmp_path):
    for i, loss in enumerate(({"top_k_kl": True}, {"clip_tau": 2.0})):
        run = create_run_dir("opd", f"v{i}", {"x": 1}, root=tmp_path)
        base = cfg(rounds=2)
        c2 = base.model_copy(update={"loss": base.loss.model_copy(update=loss)})
        st = run_opd(c2, run, comps(tmp_path / f"v{i}", tiny(0)), "co_resident")
        assert st["status"] == "trained"


def test_guards(tmp_path):
    run = create_run_dir("opd", "g", {"x": 1}, root=tmp_path)
    with pytest.raises(ValueError, match="lr"):
        run_opd(cfg(lr=None), run, comps(tmp_path, tiny(0)), "co_resident")
    base = cfg()
    hyb = base.model_copy(update={"loss": base.loss.model_copy(update={"hybrid_weight": 0.5})})
    with pytest.raises(ValueError, match="verifier"):
        run_opd(hyb, run, comps(tmp_path, tiny(0)), "co_resident")
