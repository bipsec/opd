"""13-gram overlap filter between training prompts and evaluation items (PLAN 2.3).

Any training prompt sharing an n-gram (default n=13, on normalized word tokens) with any
evaluation item is dropped, and every drop is logged with the matching eval item.
"""

from __future__ import annotations

from dataclasses import dataclass

from qopd.types import Problem


@dataclass(frozen=True)
class ContaminationHit:
    train_id: str
    eval_id: str
    ngram: str


def ngrams(text: str, n: int = 13) -> set[tuple[str, ...]]:
    """Word n-grams of lower-cased, whitespace-normalized text."""
    words = text.lower().split()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def find_contamination(
    train: list[Problem], evals: list[Problem], n: int = 13
) -> list[ContaminationHit]:
    raise NotImplementedError("step 2.3")


def drop_contaminated(
    train: list[Problem], hits: list[ContaminationHit]
) -> tuple[list[Problem], list[ContaminationHit]]:
    raise NotImplementedError("step 2.3")
