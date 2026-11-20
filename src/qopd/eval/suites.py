"""Benchmark suite definitions (PLAN 2.2, 3.2).

Samples per problem from PLAN 3.2. Source datasets are pinned in step 2.2; a suite
that cannot be found under its expected name stops the step (no substitution).
"""

from __future__ import annotations

from dataclasses import dataclass

from qopd.types import ProblemKind


@dataclass(frozen=True)
class Suite:
    name: str
    n_samples: int
    kind: ProblemKind = ProblemKind.MATH
    gated: bool = False  # needs the user's HF access


SUITES: dict[str, Suite] = {
    s.name: s
    for s in (
        Suite("gsm8k", 1),
        Suite("math500", 4),
        Suite("amc23", 8),
        Suite("aime24", 16),
        Suite("aime25", 16),
        Suite("aime26", 16),
        Suite("hmmt_feb26", 16),
        Suite("gpqa_diamond", 4, ProblemKind.MULTIPLE_CHOICE, gated=True),
    )
}


def get_suite(name: str) -> Suite:
    try:
        return SUITES[name]
    except KeyError as e:
        raise KeyError(f"unknown suite {name!r}; known: {sorted(SUITES)}") from e
