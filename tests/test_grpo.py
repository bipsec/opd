"""PLAN 8.2: GRPO baseline on CPU with a tiny on-policy student and a toy verifier."""

import itertools
import math

import pytest

torch = pytest.importorskip("torch")

from qopd.config import GRPOCfg  # noqa: E402
from qopd.train.grpo import group_advantages, run_grpo  # noqa: E402
from qopd.train.opd_loop import LoopComponents  # noqa: E402
from qopd.utils import io  # noqa: E402
from qopd.utils.run import create_run_dir  # noqa: E402
from tests.test_opd_loop import PROBLEMS, FakeEngine, FileSync, cfg, prompt_of, tiny  # noqa: E402

TARGET = 7


def comps(tmp, student, verifier=lambda p, ids: TARGET in ids):
    return LoopComponents(
        model=student,
        engine=FakeEngine(student),
        teacher=None,
        weight_sync=FileSync(tmp / "sync"),
        problems=PROBLEMS,
        prompt_ids=prompt_of,
        pad_id=0,
        verifier=verifier,
        device="cpu",
    )


def p_target_first(model):
    with torch.no_grad():
        return (
            sum(
                float(torch.softmax(model(torch.tensor([[a, 9]])).logits[0, -1], -1)[TARGET])
                for a in range(1, 8)
            )
            / 7
        )


def test_group_advantages_known_values():
    adv = group_advantages([1, 0, 0, 0, 1, 1, 1, 1], ["a"] * 4 + ["b"] * 4)
    sd = math.sqrt(0.1875)
    assert adv[:4] == pytest.approx([0.75 / sd, -0.25 / sd, -0.25 / sd, -0.25 / sd], rel=1e-5)
    assert adv[4:] == [0.0] * 4  # no signal when the whole group agrees
    raw = group_advantages([1, 0], ["g", "g"], normalize_std=False)
    assert raw == [0.5, -0.5]


def test_grpo_raises_rewarded_behaviour(tmp_path):
    run = create_run_dir("grpo", "learn", {"x": 1}, root=tmp_path)
    student = tiny(0)
    before = p_target_first(student)
    c = cfg(rounds=30, checkpoint_every=30, lr=2e-2, rollouts_per_prompt=4)
    st = run_grpo(c, GRPOCfg(group_size=4), run, comps(tmp_path, student))
    after = p_target_first(student)
    assert st["status"] == "trained" and after > 2 * before, (before, after)
    rows = io.read_jsonl(run.metrics)
    assert {"reward_mean", "zero_signal_groups", "entropy_mean", "repetition_rate"} <= set(rows[0])


def test_wall_clock_budget_stops_and_is_resumable(tmp_path):
    run = create_run_dir("grpo", "wc", {"x": 1}, root=tmp_path)
    ticks = itertools.count(step=3600.0)  # every clock() call advances one hour
    clock = lambda: next(ticks)  # noqa: E731
    g = GRPOCfg(wall_clock_hours=2.5)
    st = run_grpo(cfg(rounds=50), g, run, comps(tmp_path, tiny(0)), clock=clock)
    assert st["status"] == "trained" and st["stopped_by"] == "wall_clock"
    assert st["round"] == 3 and len(io.read_jsonl(run.metrics)) == 3
    st2 = run_grpo(cfg(rounds=50), g, run, comps(tmp_path, tiny(1)), clock=clock)
    assert st2["round"] == 3  # budget already used: resume does nothing more


def test_resume_matches_uninterrupted(tmp_path):
    g = GRPOCfg(group_size=2)
    a = create_run_dir("grpo", "a", {"x": 1}, root=tmp_path)
    s1 = tiny(0)
    run_grpo(cfg(rounds=5), g, a, comps(tmp_path / "a", s1))
    b = create_run_dir("grpo", "b", {"x": 1}, root=tmp_path)
    run_grpo(cfg(rounds=5), g, b, comps(tmp_path / "b", tiny(0)), stop_after_rounds=2)
    s3 = tiny(5)
    run_grpo(cfg(rounds=5), g, b, comps(tmp_path / "b", s3))
    p1 = torch.cat([p.detach().flatten() for p in s1.parameters()])
    p3 = torch.cat([p.detach().flatten() for p in s3.parameters()])
    assert torch.allclose(p1, p3, atol=1e-5)


def test_guards(tmp_path):
    run = create_run_dir("grpo", "g", {"x": 1}, root=tmp_path)
    with pytest.raises(ValueError, match="verifier"):
        run_grpo(cfg(), GRPOCfg(), run, comps(tmp_path, tiny(0), verifier=None))
    with pytest.raises(NotImplementedError):
        run_grpo(cfg(), GRPOCfg(kl_coef=0.1), run, comps(tmp_path, tiny(0)))
