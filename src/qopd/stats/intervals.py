"""Confidence intervals (PLAN 10.1).

- :func:`seed_ci`: t-interval of a mean over seeds (few values, so wide by design).
- :func:`bootstrap_problems`: percentile bootstrap of pass@1, resampling problems.
- :func:`bootstrap_diff`: paired bootstrap of the pass@1 difference between two conditions
  evaluated on the same problems (resampling problems jointly).

``correct`` arrays are ``[problem][sample]`` with entries in {0, 1}; pass@1 is the mean over
problems of the per-problem mean over samples (rows may have different sample counts).
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Interval:
    estimate: float
    low: float
    high: float
    level: float

    def __iter__(self) -> Iterator[float]:  # allows ``est, lo, hi = interval``
        return iter((self.estimate, self.low, self.high))


def seed_ci(values: Sequence[float], level: float = 0.95) -> Interval:
    """Student-t interval for the mean of ``values`` (one value per seed)."""
    from scipy import stats  # noqa: PLC0415

    x = np.asarray(values, dtype=float)
    if x.size == 0:
        raise ValueError("no values")
    mean = float(x.mean())
    if x.size == 1:
        return Interval(mean, float("nan"), float("nan"), level)
    half = float(stats.t.ppf(0.5 + level / 2, x.size - 1) * x.std(ddof=1) / math.sqrt(x.size))
    return Interval(mean, mean - half, mean + half, level)


def per_problem_rates(correct: Sequence[Sequence[float]]) -> np.ndarray:
    if not correct:
        raise ValueError("no problems")
    return np.array([float(np.mean(row)) for row in correct], dtype=float)


def _percentile(boot: np.ndarray, level: float) -> tuple[float, float]:
    lo, hi = np.quantile(boot, [0.5 - level / 2, 0.5 + level / 2])
    return float(lo), float(hi)


def bootstrap_problems(
    correct: Sequence[Sequence[float]],
    n_boot: int = 10_000,
    level: float = 0.95,
    seed: int = 0,
) -> Interval:
    """pass@1 with a percentile bootstrap interval over problems."""
    rates = per_problem_rates(correct)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, rates.size, size=(n_boot, rates.size))
    boot = rates[idx].mean(axis=1)
    lo, hi = _percentile(boot, level)
    return Interval(float(rates.mean()), lo, hi, level)


def bootstrap_diff(
    correct_a: Sequence[Sequence[float]],
    correct_b: Sequence[Sequence[float]],
    n_boot: int = 10_000,
    level: float = 0.95,
    seed: int = 0,
) -> Interval:
    """pass@1(a) - pass@1(b) on the same problems (row i = same problem in both)."""
    a, b = per_problem_rates(correct_a), per_problem_rates(correct_b)
    if a.size != b.size:
        raise ValueError("conditions must cover the same problems")
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    lo, hi = _percentile(d[idx].mean(axis=1), level)
    return Interval(float(d.mean()), lo, hi, level)
