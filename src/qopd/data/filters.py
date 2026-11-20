"""Answer parsability, text normalization and deduplication (PLAN 2.1).

Stages, applied in order by :func:`qopd.data.pool.build_pool`:

1. ``normalize`` — tidy whitespace in problem and answer; drop empty ones.
2. ``answer_type`` — keep answers that math-verify parses to a single real number with
   no free variables (``integer`` / ``number`` / ``closed_form``); this is the doc's
   "integer or simple closed-form answers so math-verify can check them".
3. ``self_check`` — the gold answer, written as ``\\boxed{...}``, must verify against
   itself, so the evaluation verifier can mark a correct student answer correct.
4. ``dedup_exact`` — exact duplicates of the case- and whitespace-folded problem text.
5. ``dedup_near`` — MinHash on word 5-grams (128 permutations), Jaccard >= 0.8.

Each stage returns the kept problems plus per-reason drop counts and the dropped ids.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace

from qopd.types import Problem

ANSWER_TYPES = ("integer", "number", "closed_form")


@dataclass
class FilterResult:
    stage: str
    n_in: int
    kept: list[Problem]
    dropped: dict[str, int] = field(default_factory=dict)  # reason -> count
    dropped_ids: list[tuple[str, str]] = field(default_factory=list)  # (id, reason)

    def drop(self, problem: Problem, reason: str) -> None:
        self.dropped[reason] = self.dropped.get(reason, 0) + 1
        self.dropped_ids.append((problem.id, reason))

    def summary(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "in": self.n_in,
            "kept": len(self.kept),
            "dropped": dict(sorted(self.dropped.items())),
        }


# --- 1. normalization -----------------------------------------------------------------

_HSPACE = re.compile("[ \t\f\v\u00a0]+")  # horizontal whitespace incl. NBSP
_MANY_NEWLINES = re.compile(r"\n{3,}")


def normalize_whitespace(text: str) -> str:
    """Collapse all whitespace runs to one space and strip (used for dedup keys)."""
    return " ".join(text.split())


def normalize_text(text: str) -> str:
    """Tidy whitespace but keep line structure (stored problem text)."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [_HSPACE.sub(" ", line).strip() for line in text.split("\n")]
    return _MANY_NEWLINES.sub("\n\n", "\n".join(lines)).strip()


def normalize(problems: list[Problem]) -> FilterResult:
    res = FilterResult("normalize", len(problems), [])
    for p in problems:
        q, a = normalize_text(p.problem), normalize_whitespace(p.answer)
        if not q:
            res.drop(p, "empty_problem")
        elif not a:
            res.drop(p, "empty_answer")
        else:
            res.kept.append(replace(p, problem=q, answer=a))
    return res


# --- 2./3. answer checks ----------------------------------------------------------------


def classify_answer(answer: str) -> str:
    """``integer`` | ``number`` | ``closed_form`` (kept) or the reason it is not simple."""
    import sympy  # noqa: PLC0415
    from math_verify import parse  # noqa: PLC0415

    try:
        parsed = parse(f"${answer}$")
    except Exception:  # noqa: BLE001 - parser failures are a drop reason, not a crash
        return "parse_error"
    if not parsed:
        return "unparsed"
    x = parsed[0]
    if isinstance(x, str):
        return "string"
    if isinstance(x, sympy.Integer):
        return "integer"
    if isinstance(x, sympy.Rational | sympy.Float):
        return "number"
    if isinstance(x, sympy.Expr) and x.is_number and not x.free_symbols and not x.has(sympy.I):
        return "closed_form"
    return f"other:{type(x).__name__}"


def keep_parsable_answers(
    problems: list[Problem], keep: tuple[str, ...] | list[str] = ANSWER_TYPES
) -> FilterResult:
    res = FilterResult("answer_type", len(problems), [])
    for p in problems:
        t = classify_answer(p.answer)
        if t in keep:
            res.kept.append(replace(p, meta={**p.meta, "answer_type": t}))
        else:
            res.drop(p, t)
    return res


def self_verifies(answer: str) -> bool:
    """The gold answer must verify against ``\\boxed{gold}`` (what a correct student writes)."""
    from math_verify import parse, verify  # noqa: PLC0415

    try:
        return bool(verify(parse(f"${answer}$"), parse(f"\\boxed{{{answer}}}")))
    except Exception:  # noqa: BLE001
        return False


def self_check(problems: list[Problem]) -> FilterResult:
    res = FilterResult("self_check", len(problems), [])
    for p in problems:
        if self_verifies(p.answer):
            res.kept.append(p)
        else:
            res.drop(p, "boxed_gold_does_not_verify")
    return res


# --- 4./5. deduplication ---------------------------------------------------------------------


def dedup_key(text: str) -> str:
    return normalize_whitespace(text).lower()


def answers_agree(a: str, b: str) -> bool:
    """Same text, or math-verify judges them equal (``1/2`` vs ``\\frac{1}{2}``)."""
    if normalize_whitespace(a) == normalize_whitespace(b):
        return True
    from math_verify import parse, verify  # noqa: PLC0415

    try:
        return bool(verify(parse(f"${a}$"), parse(f"${b}$")))
    except Exception:  # noqa: BLE001
        return False


def dedup_exact(problems: list[Problem], drop_conflicts: bool = True) -> FilterResult:
    """Keep the first occurrence of each case/whitespace-folded problem text.

    With ``drop_conflicts``, a group of identical problems whose gold answers disagree is
    dropped entirely (at least one label is wrong and we cannot tell which).
    """
    res = FilterResult("dedup_exact", len(problems), [])
    groups: dict[str, list[Problem]] = {}
    for p in problems:
        groups.setdefault(dedup_key(p.problem), []).append(p)
    conflicted = {
        k
        for k, g in groups.items()
        if drop_conflicts and len(g) > 1 and not all(answers_agree(g[0].answer, x.answer) for x in g)
    }
    seen: set[str] = set()
    for p in problems:
        k = dedup_key(p.problem)
        if k in conflicted:
            res.drop(p, "duplicate_with_conflicting_answers")
        elif k in seen:
            res.drop(p, "exact_duplicate")
        else:
            seen.add(k)
            res.kept.append(p)
    return res


def word_shingles(text: str, n: int) -> set[str]:
    words = dedup_key(text).split()
    if len(words) < n:
        return {" ".join(words)}
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def dedup_near(
    problems: list[Problem], threshold: float = 0.8, num_perm: int = 128, ngram: int = 5
) -> FilterResult:
    """Drop a problem if an earlier kept problem has MinHash Jaccard >= ``threshold``.

    LSH proposes candidates; each candidate is confirmed with the MinHash Jaccard estimate.
    """
    from datasketch import MinHash, MinHashLSH  # noqa: PLC0415

    res = FilterResult("dedup_near", len(problems), [])
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    sigs: dict[str, MinHash] = {}
    for p in problems:
        m = MinHash(num_perm=num_perm, seed=1)
        for sh in word_shingles(p.problem, ngram):
            m.update(sh.encode("utf-8"))
        if any(m.jaccard(sigs[c]) >= threshold for c in lsh.query(m)):
            res.drop(p, "near_duplicate")
            continue
        lsh.insert(p.id, m)
        sigs[p.id] = m
        res.kept.append(p)
    return res
