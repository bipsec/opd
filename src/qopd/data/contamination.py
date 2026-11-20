"""13-gram overlap filter between training prompts and evaluation items (PLAN 2.3).

Tokens: lower-cased word tokens (``[a-z0-9]+`` runs), so LaTeX markup, punctuation and
spacing cannot hide an overlap (``$x^2$`` and ``x^2`` give the same tokens). Any training
prompt that shares at least one 13-gram with any evaluation item is dropped; every drop is
logged with the matching evaluation ids, the number of shared 13-grams and an example.

Evaluation items shorter than 13 tokens cannot be matched; their count is reported.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

from qopd.types import Problem

_TOKEN = re.compile(r"[a-z0-9]+")


def tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def ngrams(text: str, n: int = 13) -> set[tuple[str, ...]]:
    """Token n-grams of lower-cased text (empty if the text has fewer than n tokens)."""
    t = tokens(text)
    return {tuple(t[i : i + n]) for i in range(len(t) - n + 1)}


@dataclass(frozen=True)
class ContaminationHit:
    train_id: str
    eval_id: str
    ngram: str  # one shared n-gram (example)
    n_shared: int  # number of distinct shared n-grams for this pair


@dataclass
class EvalIndex:
    n: int
    index: dict[tuple[str, ...], list[str]] = field(default_factory=dict)  # ngram -> eval ids
    too_short: dict[str, int] = field(default_factory=dict)  # suite -> items < n tokens
    n_items: dict[str, int] = field(default_factory=dict)  # suite -> items indexed


def build_index(evals: list[Problem], n: int = 13) -> EvalIndex:
    idx = EvalIndex(n)
    tmp: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for p in evals:
        idx.n_items[p.source] = idx.n_items.get(p.source, 0) + 1
        grams = ngrams(p.problem, n)
        if not grams:
            idx.too_short[p.source] = idx.too_short.get(p.source, 0) + 1
        for g in grams:
            tmp[g].append(p.id)
    idx.index = dict(tmp)
    return idx


def find_contamination(
    train: list[Problem], evals: list[Problem] | EvalIndex, n: int = 13
) -> list[ContaminationHit]:
    """One hit per (training prompt, evaluation item) pair sharing >= 1 n-gram."""
    idx = evals if isinstance(evals, EvalIndex) else build_index(evals, n)
    hits: list[ContaminationHit] = []
    for p in train:
        shared: dict[str, list[tuple[str, ...]]] = defaultdict(list)
        for g in ngrams(p.problem, idx.n):
            for eid in idx.index.get(g, ()):
                shared[eid].append(g)
        for eid, grams in sorted(shared.items()):
            hits.append(ContaminationHit(p.id, eid, " ".join(min(grams)), len(grams)))
    return hits


def drop_contaminated(
    train: list[Problem], hits: list[ContaminationHit]
) -> tuple[list[Problem], list[ContaminationHit]]:
    """Remove every training prompt that has at least one hit."""
    bad = {h.train_id for h in hits}
    return [p for p in train if p.id not in bad], hits
