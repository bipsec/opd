"""Sequential, resumable job queue (PLAN 8.1).

Skips runs with a DONE marker; each run gets its own tmux session (name = run id) with
logs in the run directory; ``qopd status`` shows done / running / queued and ETA.
Jobs may target the peer Spark only when ``qopd peer`` reports it free.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from qopd.orchestration.grid import RunSpec


class JobState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass
class QueueStatus:
    done: list[str]
    running: list[str]
    queued: list[str]
    failed: list[str]
    eta_hours: float | None


def job_state(spec: RunSpec, runs_root: Path) -> JobState:
    raise NotImplementedError("step 8.1")


def launch_next(specs: list[RunSpec], runs_root: Path) -> RunSpec | None:
    """Start the first queued run in a tmux session; None if nothing is left."""
    raise NotImplementedError("step 8.1")


def status(specs: list[RunSpec], runs_root: Path) -> QueueStatus:
    raise NotImplementedError("step 8.1")
