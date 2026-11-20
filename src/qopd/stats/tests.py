"""Hypothesis tests (PLAN 10.1): TOST, Spearman comparison, Holm correction."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class TOSTResult:
    diff: float
    p_lower: float
    p_upper: float
    p: float  # max of the two one-sided p-values
    equivalent: bool
    margin: float


def tost(
    a: Sequence[float], b: Sequence[float], margin: float = 1.0, alpha: float = 0.05
) -> TOSTResult:
    """Equivalence of means within ±margin points (MATH-500 default ±1.0)."""
    raise NotImplementedError("step 10.1")


def compare_spearman(
    y: Sequence[float], x1: Sequence[float], x2: Sequence[float]
) -> dict[str, float]:
    """Is fidelity (x1) a better predictor of y than teacher accuracy (x2)?"""
    raise NotImplementedError("step 10.1")


def holm(pvalues: Sequence[float], alpha: float = 0.05) -> list[bool]:
    """Holm step-down: reject flags in the input order."""
    raise NotImplementedError("step 10.1")
