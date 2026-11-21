"""Hypothesis tests (PLAN 10.1): TOST, Spearman comparison, Holm correction.

- H1 (equivalence within ±1.0 MATH-500 point): two one-sided t-tests. ``tost_paired`` works on
  per-problem pass-rate differences between a rung and T16 (same problems, so paired);
  ``tost_two_sample`` (Welch) works on independent values such as per-seed scores.
  Margins and values are in the same units (pass@1 as a fraction: 1 point = 0.01).
- H2: :func:`compare_spearman` rank-correlates two predictors with the outcome across
  conditions and bootstraps the difference of the two correlations over conditions.
- Pre-registered comparisons are corrected with :func:`holm`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TOSTResult:
    diff: float  # mean(a) - mean(b), or mean of paired differences
    p_lower: float  # H0: diff <= -margin
    p_upper: float  # H0: diff >= +margin
    p: float  # max of the two one-sided p-values
    equivalent: bool
    margin: float
    ci90: tuple[float, float]  # 1 - 2*alpha interval; inside ±margin <=> equivalent


def _tost(diff: float, se: float, df: float, margin: float, alpha: float) -> TOSTResult:
    from scipy import stats  # noqa: PLC0415

    if se == 0:
        inside = -margin < diff < margin
        p = 0.0 if inside else 1.0
        return TOSTResult(diff, p, p, p, inside, margin, (diff, diff))
    t_low = (diff + margin) / se
    t_up = (diff - margin) / se
    p_low = float(stats.t.sf(t_low, df))
    p_up = float(stats.t.cdf(t_up, df))
    half = float(stats.t.ppf(1 - alpha, df) * se)
    p = max(p_low, p_up)
    return TOSTResult(diff, p_low, p_up, p, p < alpha, margin, (diff - half, diff + half))


def tost_paired(d: Sequence[float], margin: float = 0.01, alpha: float = 0.05) -> TOSTResult:
    """Equivalence of the mean paired difference within ±margin."""
    x = np.asarray(d, dtype=float)
    if x.size < 2:
        raise ValueError("need at least two paired differences")
    return _tost(
        float(x.mean()), float(x.std(ddof=1) / math.sqrt(x.size)), x.size - 1, margin, alpha
    )


def tost_two_sample(
    a: Sequence[float], b: Sequence[float], margin: float = 0.01, alpha: float = 0.05
) -> TOSTResult:
    """Welch two-sample TOST for mean(a) - mean(b) within ±margin."""
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if x.size < 2 or y.size < 2:
        raise ValueError("need at least two values per group")
    vx, vy = x.var(ddof=1) / x.size, y.var(ddof=1) / y.size
    se = math.sqrt(vx + vy)
    df = (vx + vy) ** 2 / (vx**2 / (x.size - 1) + vy**2 / (y.size - 1)) if se > 0 else 1.0
    return _tost(float(x.mean() - y.mean()), se, df, margin, alpha)


def tost(
    a: Sequence[float], b: Sequence[float], margin: float = 0.01, alpha: float = 0.05
) -> TOSTResult:
    """Backwards-compatible alias for :func:`tost_two_sample`."""
    return tost_two_sample(a, b, margin, alpha)


@dataclass(frozen=True)
class SpearmanComparison:
    rho_1: float  # predictor 1 (e.g. teacher-signal fidelity) vs outcome
    rho_2: float  # predictor 2 (e.g. teacher's own accuracy) vs outcome
    diff: float  # |rho_1| - |rho_2|
    ci: tuple[float, float]  # bootstrap interval of diff over conditions
    p_boot: float  # two-sided bootstrap p-value for diff = 0
    n: int


def compare_spearman(
    y: Sequence[float],
    x1: Sequence[float],
    x2: Sequence[float],
    n_boot: int = 10_000,
    level: float = 0.95,
    seed: int = 0,
) -> SpearmanComparison:
    """Is x1 a better (rank) predictor of y than x2? Units are conditions (rung x student)."""
    from scipy import stats  # noqa: PLC0415

    yy, a, b = (np.asarray(v, dtype=float) for v in (y, x1, x2))
    if not (yy.size == a.size == b.size) or yy.size < 3:
        raise ValueError("need >= 3 conditions with all three values")

    def rho(u: np.ndarray, v: np.ndarray) -> float:
        r = stats.spearmanr(u, v).statistic
        return 0.0 if np.isnan(r) else float(r)

    r1, r2 = rho(a, yy), rho(b, yy)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, yy.size, size=yy.size)
        diffs[i] = abs(rho(a[idx], yy[idx])) - abs(rho(b[idx], yy[idx]))
    lo, hi = np.quantile(diffs, [0.5 - level / 2, 0.5 + level / 2])
    p = float(min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())))
    return SpearmanComparison(r1, r2, abs(r1) - abs(r2), (float(lo), float(hi)), p, int(yy.size))


def holm(pvalues: Sequence[float], alpha: float = 0.05) -> tuple[list[bool], list[float]]:
    """Holm step-down. Returns (reject flags, adjusted p-values), both in input order."""
    p = np.asarray(pvalues, dtype=float)
    m = p.size
    order = np.argsort(p, kind="stable")
    adj_sorted = np.maximum.accumulate(np.minimum(1.0, (m - np.arange(m)) * p[order]))
    adj = np.empty(m)
    adj[order] = adj_sorted
    return [bool(x <= alpha) for x in adj], [float(x) for x in adj]
