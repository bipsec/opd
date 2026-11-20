"""PLAN 3.2: runner bookkeeping with a fake engine (CPU): pass@1, summary, resume."""

from dataclasses import dataclass

import pytest

from qopd.config import EvalCfg, SamplingCfg, SuiteCfg
from qopd.eval.runner import EvalRunner, pass_at_1, sample_seed
from qopd.types import FinishReason, Problem, Rollout
from qopd.utils import io
from qopd.utils.run import create_run_dir


class FakeTok:
    """Chars <-> ids; the chat template is just the problem text."""

    def apply_chat_template(self, messages, **kw):
        return [ord(c) for c in messages[0]["content"]]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in ids)


@dataclass
class Stats:
    peak_mem_gb: float = 1.0


class FakeEngine:
    """Answers correctly iff the request seed is even; optionally crashes after N calls."""

    def __init__(self, answers, crash_after=None):
        self.stop_ids = [0]
        self.answers, self.crash_after, self.calls, self.started = answers, crash_after, 0, False

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def generate(self, prompt_ids, sampling, n=1, problem_ids=None, seed=None, seeds=None):
        self.calls += 1
        if self.crash_after is not None and self.calls > self.crash_after:
            raise RuntimeError("simulated crash")
        out = []
        for pid, s in zip(problem_ids, seeds, strict=True):
            ans = self.answers[pid] if s % 2 == 0 else "wrong"
            text = f"<think>hm</think> \\boxed{{{ans}}}"
            reason = FinishReason.STOP if s % 3 else FinishReason.LENGTH
            out.append(Rollout([1], [ord(c) for c in text], [-0.1] * len(text), reason, pid))
        return out, Stats()


SUITE = [Problem(f"s/{i}", f"What is {i}+{i}?", str(2 * i), "s") for i in range(10)]
CFG = EvalCfg(
    sampling=SamplingCfg(temperature=0.6, top_p=0.95, max_tokens=64),
    suites={"s": SuiteCfg(n_samples=3)},
)
ANSWERS = {p.id: p.answer for p in SUITE}


def expected_pass_at_1(n):
    return sum(sum(sample_seed(0, p.id, k) % 2 == 0 for k in range(n)) / n for p in SUITE) / len(
        SUITE
    )


def test_pass_at_1():
    assert pass_at_1({"a": [True, False], "b": [True, True]}) == 0.75
    with pytest.raises(ValueError):
        pass_at_1({})


def test_seed_deterministic():
    assert sample_seed(0, "x", 1) == sample_seed(0, "x", 1) != sample_seed(0, "x", 2)


def test_full_run(tmp_path):
    run = create_run_dir("eval", "r", {"x": 1}, root=tmp_path)
    eng = FakeEngine(ANSWERS)
    s = EvalRunner(FakeTok(), CFG, run, lambda max_len: eng, chunk_size=7).run(
        {"s": SUITE}, {"s": 3}
    )
    r = s["suites"]["s"]
    assert r["complete"] and r["n_samples_done"] == 30
    assert r["pass_at_1"] == pytest.approx(expected_pass_at_1(3))
    assert 0 < r["truncation_rate"] < 1
    assert run.is_done and (run.path / "summary.json").exists()
    rows = io.read_jsonl(run.path / "samples.jsonl")
    assert {(x["problem_id"], x["sample_idx"]) for x in rows} == {
        (p.id, k) for p in SUITE for k in range(3)
    }
    assert all(x["reason"] in ("correct", "wrong") for x in rows)


def test_resume_after_crash_matches_uninterrupted(tmp_path):
    run = create_run_dir("eval", "r", {"x": 1}, root=tmp_path)
    crashing = FakeEngine(ANSWERS, crash_after=2)
    runner = EvalRunner(FakeTok(), CFG, run, lambda max_len: crashing, chunk_size=7)
    with pytest.raises(RuntimeError, match="simulated crash"):
        runner.run({"s": SUITE}, {"s": 3})
    assert not crashing.started  # engine stopped even on failure
    partial = io.read_jsonl(run.path / "samples.jsonl")
    assert len(partial) == 14 and not run.is_done
    resumed = FakeEngine(ANSWERS)
    s = EvalRunner(FakeTok(), CFG, run, lambda max_len: resumed, chunk_size=7).run(
        {"s": SUITE}, {"s": 3}
    )
    rows = io.read_jsonl(run.path / "samples.jsonl")
    assert len(rows) == 30 and len({(x["problem_id"], x["sample_idx"]) for x in rows}) == 30
    assert s["suites"]["s"]["pass_at_1"] == pytest.approx(expected_pass_at_1(3))
    assert run.is_done


def test_nothing_to_do_does_not_start_engine(tmp_path):
    run = create_run_dir("eval", "r", {"x": 1}, root=tmp_path)
    EvalRunner(FakeTok(), CFG, run, lambda m: FakeEngine(ANSWERS)).run({"s": SUITE}, {"s": 1})

    def boom(max_len):
        raise AssertionError("engine should not be built")

    s = EvalRunner(FakeTok(), CFG, run, boom).run({"s": SUITE}, {"s": 1})
    assert s["suites"]["s"]["complete"]


@pytest.mark.gpu
def test_s2_gsm8k_smoke(tmp_path):
    from qopd.config import CONFIG_ROOT, load
    from qopd.eval.runner import evaluate

    cfg = load(CONFIG_ROOT / "eval" / "default.yaml", ["sampling.max_tokens=512"], cls=EvalCfg)
    s = evaluate("S2", ["gsm8k"], cfg, n_samples=1, limit=4, kv_cache_gb=2.0, root=tmp_path)
    r = s["suites"]["gsm8k"]
    assert r["complete"] and r["n_samples_done"] == 4
    assert 0.0 <= r["pass_at_1"] <= 1.0
