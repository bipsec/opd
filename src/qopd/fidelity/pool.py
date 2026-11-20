"""Fixed rollout pools for E1 (PLAN 7.1, 7.2).

2,000 rollouts per cold-start student from the probe prompts, saved with exact token
ids; each approved rung scores the same pool (top-20), one file per rung per student.
"""

from __future__ import annotations

from pathlib import Path

from qopd.types import Rollout, TeacherScores


def build_pool(student: str, probe_split: Path, n_rollouts: int, out: Path) -> Path:
    raise NotImplementedError("step 7.1")


def load_pool(path: Path) -> list[Rollout]:
    raise NotImplementedError("step 7.1")


def score_pool(pool: Path, rung: str, top_k: int, out: Path) -> Path:
    raise NotImplementedError("step 7.2")


def load_scores(path: Path) -> list[TeacherScores]:
    raise NotImplementedError("step 7.2")
