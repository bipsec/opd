"""Evaluation runner (PLAN 3.2): generate, verify, pass@1 averaged over samples.

Writes per-sample ``samples.jsonl`` and ``summary.json`` into the run directory and
resumes from partial results (already-scored (problem, sample) pairs are skipped).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qopd.config import EvalCfg
    from qopd.utils.run import RunDir


@dataclass
class SuiteResult:
    suite: str
    n_problems: int
    n_samples: int
    pass_at_1: float
    seconds: float


def pass_at_1(correct_by_problem: Mapping[str, Sequence[bool]]) -> float:
    """Mean over problems of the per-problem fraction of correct samples."""
    if not correct_by_problem:
        raise ValueError("no problems")
    return sum(sum(v) / len(v) for v in correct_by_problem.values()) / len(correct_by_problem)


def evaluate(model_id: str, suites: Sequence[str], cfg: EvalCfg, run: RunDir) -> list[SuiteResult]:
    raise NotImplementedError("step 3.2")
