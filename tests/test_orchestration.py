"""PLAN 8.1: grid expansion and the sequential resumable queue (fake tmux, CPU)."""

import json

import pytest

from qopd.config import CONFIG_ROOT, ExperimentCfg, GridCfg, load
from qopd.orchestration.grid import expand, plan
from qopd.orchestration.queue import (
    JobState,
    RunSpec,
    docker_prefix,
    job_state,
    load_queue,
    next_queued,
    run_id,
    status,
    supervise,
)


class FakeTmux:
    """Sessions 'run' instantly: a launched session finishes on the next has_session poll."""

    def __init__(self, succeed=lambda rid: True, runs_root=None):
        self.live, self.launched, self.succeed, self.root = {}, [], succeed, runs_root

    def available(self):
        return True

    def has_session(self, name):
        if name in self.live:
            self.live[name] -= 1
            if self.live[name] < 0:
                del self.live[name]
                return False
            return True
        return False

    def new_session(self, name, command):
        self.launched.append(name)
        self.live[name] = 1  # alive for one poll
        if self.succeed(name):
            exp = "e"
            (self.root / exp / name).mkdir(parents=True, exist_ok=True)
            (self.root / exp / name / "DONE").write_text("x")


def specs(n=3, optional_last=False):
    out = []
    for i in range(n):
        out.append(
            RunSpec(
                "e",
                run_id("opd", "S2", "T16", i),
                "opd",
                "S2",
                "T16",
                i,
                f"qopd train opd --spec runs/e/specs/{i}.yaml",
                f"runs/e/specs/{i}.yaml",
                optional=optional_last and i == n - 1,
                estimate_key="opd:S2",
                estimate_hours=6.0,
            )
        )
    return out


def test_e2_grid_expands_to_doc_counts(tmp_path):
    grid = load(CONFIG_ROOT / "grids" / "e2.yaml", cls=GridCfg)
    sp = expand(grid, tmp_path)
    req = [s for s in sp if not s.optional]
    opd_main = [s for s in req if s.kind == "opd"]
    assert len(opd_main) == 20  # 5 rungs x 2 students x 2 seeds
    assert len([s for s in sp if s.optional and s.teacher == "T2"]) == 2
    assert len([s for s in sp if s.optional and s.seed == 2]) == 3
    assert {s.kind for s in req} == {"opd", "eval", "sft_offpolicy", "grpo"}
    assert len({s.run_id for s in sp}) == len(sp)
    one = load(tmp_path / "opd__S1__T4-int__s1.yaml", cls=ExperimentCfg)
    assert (one.student, one.teacher, one.seeds, one.name) == ("S1", "T4-int", [1], "e2")
    assert load(tmp_path / "grpo__S1__none__s0.yaml", cls=ExperimentCfg).teacher is None
    s1 = next(s for s in sp if s.run_id == "opd__S1__T16__s0")
    assert s1.estimate_hours == 17 and "--spec" in s1.command


def test_plan_writes_queue(tmp_path):
    path, sp = plan("e2", runs_root=tmp_path)
    assert path == tmp_path / "e2" / "queue.json"
    assert [s.run_id for s in load_queue(tmp_path, "e2")] == [s.run_id for s in sp]
    assert json.loads(path.read_text())["grid_file"] == "configs/grids/e2.yaml"


def test_unknown_model_fails_early(tmp_path):
    from qopd.config import GridBlockCfg, RunKind

    bad = GridCfg(
        name="x",
        base="experiments/e2_base.yaml",
        commands={RunKind.OPD: "c {spec}"},
        blocks=[GridBlockCfg(students=["S9"], teachers=["T16"], seeds=[0])],
    )
    with pytest.raises(KeyError):
        expand(bad, tmp_path)


def test_states_and_eta(tmp_path):
    sp = specs()
    tm = FakeTmux(runs_root=tmp_path)
    (tmp_path / "e" / sp[0].run_id).mkdir(parents=True)
    (tmp_path / "e" / sp[0].run_id / "DONE").write_text("x")
    (tmp_path / "e" / sp[1].run_id).mkdir(parents=True)
    (tmp_path / "e" / sp[1].run_id / "queue_started.json").write_text('{"t": 0}')
    assert job_state(sp[0], tmp_path, tm) == JobState.DONE
    assert job_state(sp[1], tmp_path, tm) == JobState.FAILED  # started, no DONE, no session
    assert job_state(sp[2], tmp_path, tm) == JobState.QUEUED
    st = status(sp, tmp_path, tmux=tm)
    assert st.count(JobState.DONE) == 1 and st.eta_hours == pytest.approx(12.0)


def test_supervise_runs_sequentially_and_skips_done(tmp_path):
    sp = specs(3, optional_last=True)
    tm = FakeTmux(runs_root=tmp_path)
    n = supervise(sp, tmp_path, "docker run x", tmux=tm, sleep=lambda s: None, log=lambda m: None)
    assert n == 2 and tm.launched == [sp[0].run_id, sp[1].run_id]  # optional skipped
    tm2 = FakeTmux(runs_root=tmp_path)
    supervise(sp, tmp_path, "p", tmux=tm2, sleep=lambda s: None, log=lambda m: None)
    assert tm2.launched == []  # everything required is DONE: nothing relaunched
    supervise(
        sp, tmp_path, "p", include_optional=True, tmux=tm2, sleep=lambda s: None, log=lambda m: None
    )
    assert tm2.launched == [sp[2].run_id]


def test_supervise_stops_on_failure_then_retries(tmp_path):
    sp = specs(3)
    tm = FakeTmux(succeed=lambda rid: rid != sp[1].run_id, runs_root=tmp_path)
    msgs = []
    n = supervise(sp, tmp_path, "p", tmux=tm, sleep=lambda s: None, log=msgs.append)
    assert n == 1 and tm.launched == [sp[0].run_id, sp[1].run_id]
    assert any("FAILED" in m for m in msgs)
    assert next_queued(sp, tmp_path, False, FakeTmux(runs_root=tmp_path)).run_id == sp[1].run_id
    tm_ok = FakeTmux(runs_root=tmp_path)
    supervise(sp, tmp_path, "p", tmux=tm_ok, sleep=lambda s: None, log=msgs.append)
    assert tm_ok.launched == [sp[1].run_id, sp[2].run_id]  # retried, then continued


def test_docker_prefix_mounts(tmp_path):
    p = docker_prefix(tmp_path, tmp_path / "hf")
    assert "--gpus all" in p and f"{tmp_path}:/workspace/opd" in p and "qopd-dev:26.10" in p


def test_queue_module_is_stdlib_only():
    import ast
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "src" / "qopd" / "orchestration" / "queue.py"
    tree = ast.parse(src.read_text())
    mods = {
        a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names
    }
    mods |= {
        n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module
    }
    assert mods <= {
        "__future__",
        "json",
        "os",
        "shlex",
        "shutil",
        "subprocess",
        "time",
        "collections",
        "dataclasses",
        "enum",
        "pathlib",
        "typing",
    }, mods
