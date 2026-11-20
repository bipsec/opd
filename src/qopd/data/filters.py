"""Answer parsability, normalization and deduplication (PLAN 2.1).

Each stage returns the kept problems plus a count of what it dropped, so
``reports/data.md`` can show counts per source and per filter stage.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from qopd.types import Problem


@dataclass
class FilterResult:
    stage: str
    kept: list[Problem]
    dropped: dict[str, int] = field(default_factory=dict)  # reason -> count


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace and strip ends."""
    return " ".join(text.split())


def keep_parsable_answers(problems: list[Problem]) -> FilterResult:
    """Keep problems whose answer math-verify can parse."""
    raise NotImplementedError("step 2.1")


def dedup_exact(problems: list[Problem]) -> FilterResult:
    """Drop exact duplicates of the normalized problem text."""
    raise NotImplementedError("step 2.1")


def dedup_near(problems: list[Problem], threshold: float = 0.9, num_perm: int = 128) -> FilterResult:
    """Drop near-duplicates with MinHash LSH (datasketch)."""
    raise NotImplementedError("step 2.1")
