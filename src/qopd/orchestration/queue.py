"""Sequential, resumable job queue (PLAN 8.1). **Standard library only.**

It runs on the host, where tmux lives; qopd's own dependencies live in the container.
``qopd queue plan <grid>`` (container) writes ``runs/<experiment>/queue.json``. This module
(host: ``python3 scripts/queue_host.py``) reads it and, one run at a time:

1. skips runs that have a ``DONE`` marker;
2. starts the run's in-container command via ``docker run`` inside a tmux session named after
   the run id, with the session output appended to ``<run dir>/tmux.log``;
3. waits for the session to end, then checks for ``DONE``. On a failure the queue stops so a
   human can look (CLAUDE.md: never plough on silently).

Re-running ``run`` resumes: done runs are skipped, an interrupted run is started again (each
run resumes itself from its own ``state.json``). ``status`` prints done / running / queued /
failed with an ETA from measured wall-clock where available, else the grid's estimates.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

QUEUE_FILE = "queue.json"


@dataclass(frozen=True)
class RunSpec:
    experiment: str
    run_id: str  # e.g. "opd__S1__T4-int__s0"; also the tmux session name
    kind: str
    student: str
    teacher: str | None
    seed: int
    command: str  # in-container command line
    spec_file: str  # project-relative path of the resolved run config
    optional: bool = False
    estimate_key: str = ""
    estimate_hours: float | None = None
    note: str = ""
    overrides: list[str] = field(default_factory=list)


class JobState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"  # started, ended without DONE


def run_id(kind: str, student: str, teacher: str | None, seed: int) -> str:
    return f"{kind}__{student}__{teacher or 'none'}__s{seed}"


# --- queue file -------------------------------------------------------------------------------


def write_queue(
    specs: list[RunSpec], runs_root: Path, experiment: str, meta: dict[str, Any]
) -> Path:
    path = runs_root / experiment / QUEUE_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"experiment": experiment, **meta, "runs": [asdict(s) for s in specs]}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)
    return path


def load_queue(runs_root: Path, experiment: str) -> list[RunSpec]:
    data = json.loads((runs_root / experiment / QUEUE_FILE).read_text())
    return [RunSpec(**r) for r in data["runs"]]


def list_queues(runs_root: Path) -> list[str]:
    return sorted(p.parent.name for p in runs_root.glob(f"*/{QUEUE_FILE}"))


# --- tmux --------------------------------------------------------------------------------------


class Tmux:
    """Thin tmux wrapper (injectable for tests)."""

    def available(self) -> bool:
        return shutil.which("tmux") is not None

    def has_session(self, name: str) -> bool:
        if not self.available():
            return False
        return (
            subprocess.run(
                ["tmux", "has-session", "-t", f"={name}"], capture_output=True, check=False
            ).returncode
            == 0
        )

    def new_session(self, name: str, command: str) -> None:
        subprocess.run(["tmux", "new-session", "-d", "-s", name, command], check=True)


# --- state ---------------------------------------------------------------------------------------


def run_dir(runs_root: Path, spec: RunSpec) -> Path:
    return runs_root / spec.experiment / spec.run_id


def job_state(spec: RunSpec, runs_root: Path, tmux: Tmux | None = None) -> JobState:
    d = run_dir(runs_root, spec)
    if (d / "DONE").exists():
        return JobState.DONE
    if (tmux or Tmux()).has_session(spec.run_id):
        return JobState.RUNNING
    if (d / "queue_started.json").exists():
        return JobState.FAILED
    return JobState.QUEUED


def _measured_hours(specs: list[RunSpec], runs_root: Path) -> dict[str, float]:
    """Mean wall-clock (hours) of finished runs, per estimate key."""
    acc: dict[str, list[float]] = {}
    for s in specs:
        d = run_dir(runs_root, s)
        start, done = d / "queue_started.json", d / "DONE"
        if start.exists() and done.exists():
            t0 = json.loads(start.read_text())["t"]
            acc.setdefault(s.estimate_key, []).append((done.stat().st_mtime - t0) / 3600)
    return {k: sum(v) / len(v) for k, v in acc.items()}


@dataclass
class QueueStatus:
    rows: list[tuple[RunSpec, JobState]]
    eta_hours: float | None  # None if any remaining run has no estimate
    eta_source: str

    def count(self, state: JobState) -> int:
        return sum(1 for _, st in self.rows if st == state)


def status(
    specs: list[RunSpec], runs_root: Path, include_optional: bool = False, tmux: Tmux | None = None
) -> QueueStatus:
    tm = tmux or Tmux()
    rows = [(s, job_state(s, runs_root, tm)) for s in specs if include_optional or not s.optional]
    measured = _measured_hours(specs, runs_root)
    eta, unknown = 0.0, False
    for s, st in rows:
        if st in (JobState.QUEUED, JobState.RUNNING, JobState.FAILED):
            h = measured.get(s.estimate_key, s.estimate_hours)
            if h is None:
                unknown = True
            else:
                eta += h
    source = "measured where available, else grid estimates" if measured else "grid estimates"
    return QueueStatus(rows, None if unknown else eta, source)


# --- running --------------------------------------------------------------------------------


def docker_prefix(project: Path, hf_cache: Path, image: str = "qopd-dev:26.10") -> str:
    """The same container invocation as the Makefile, built on the host."""
    uid, gid = os.getuid(), os.getgid()
    parts = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "--ipc=host",
        "-u",
        f"{uid}:{gid}",
        "-e",
        "HOME=/tmp/home",
        "-e",
        "HF_HUB_OFFLINE=1",
        "-v",
        "/etc/passwd:/etc/passwd:ro",
        "-v",
        "/etc/group:/etc/group:ro",
        "-v",
        f"{project}:/workspace/opd",
        "-v",
        f"{hf_cache}:/hf",
        "-w",
        "/workspace/opd",
        image,
    ]
    return " ".join(shlex.quote(p) for p in parts)


def launch(spec: RunSpec, runs_root: Path, prefix: str, tmux: Tmux | None = None) -> None:
    """Start one run in its own tmux session; output appended to <run dir>/tmux.log."""
    d = run_dir(runs_root, spec)
    d.mkdir(parents=True, exist_ok=True)
    (d / "queue_started.json").write_text(json.dumps({"t": time.time(), "command": spec.command}))
    log = shlex.quote(str(d / "tmux.log"))
    inner = f"{prefix} {spec.command} 2>&1 | tee -a {log}"
    (tmux or Tmux()).new_session(spec.run_id, f"bash -o pipefail -c {shlex.quote(inner)}")


def next_queued(
    specs: list[RunSpec], runs_root: Path, include_optional: bool, tmux: Tmux | None = None
) -> RunSpec | None:
    for s in specs:
        if s.optional and not include_optional:
            continue
        st = job_state(s, runs_root, tmux)
        if st in (JobState.QUEUED, JobState.FAILED):
            return s
        if st == JobState.RUNNING:
            return None  # sequential: wait for it
    return None


def _wait(
    spec: RunSpec, runs_root: Path, tm: Tmux, poll_s: float, sleep: Callable[[float], None]
) -> bool:
    """Block until the run's tmux session ends; True if it left a DONE marker."""
    while tm.has_session(spec.run_id):
        sleep(poll_s)
    return (run_dir(runs_root, spec) / "DONE").exists()


def supervise(
    specs: list[RunSpec],
    runs_root: Path,
    prefix: str,
    include_optional: bool = False,
    poll_s: float = 30.0,
    tmux: Tmux | None = None,
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] = print,
) -> int:
    """Run the queue to the end or the first failure; returns the number of runs finished."""
    tm = tmux or Tmux()
    finished = 0
    for spec in specs:  # a run already running (e.g. supervisor restarted) is waited for first
        if job_state(spec, runs_root, tm) == JobState.RUNNING:
            log(f"waiting for {spec.run_id}")
            if not _wait(spec, runs_root, tm, poll_s, sleep):
                log(f"FAILED: {spec.run_id} ended without DONE; stopping the queue")
                return finished
            finished += 1
    while (spec := next_queued(specs, runs_root, include_optional, tm)) is not None:
        if job_state(spec, runs_root, tm) == JobState.FAILED:
            log(f"retrying interrupted run {spec.run_id} (it resumes from its state.json)")
        log(f"launching {spec.run_id}: {spec.command}")
        launch(spec, runs_root, prefix, tm)
        if not _wait(spec, runs_root, tm, poll_s, sleep):
            log(f"FAILED: {spec.run_id} ended without DONE; stopping the queue")
            return finished
        finished += 1
    log("queue finished")
    return finished
