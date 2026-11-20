"""Confidence intervals (PLAN 10.1)."""

from __future__ import annotations

from collections.abc import Sequence


def seed_ci(values: Sequence[float], level: float = 0.95) -> tuple[float, float, float]:
    """(mean, low, high) t-interval across seeds."""
    raise NotImplementedError("step 10.1")


def bootstrap_problems(
    correct: Sequence[Sequence[float]],  # [problem][sample] in {0, 1}
    n_boot: int = 10_000,
    level: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """(pass@1, low, high) with problems resampled with replacement."""
    raise NotImplementedError("step 10.1")
