"""Answer verification (PLAN 3.1).

- math: math-verify on the last ``\\boxed{}`` (after the thinking block), numeric and
  closed-form answers.
- multiple choice (GPQA-Diamond): letter extraction.
"""

from __future__ import annotations

from dataclasses import dataclass

from qopd.types import Problem, ProblemKind


@dataclass(frozen=True)
class Verdict:
    correct: bool
    extracted: str | None  # None = no parsable answer found
    reason: str = ""


def strip_thinking(text: str) -> str:
    """Return the text after the last ``</think>`` (whole text if there is none)."""
    marker = "</think>"
    return text.rsplit(marker, 1)[1] if marker in text else text


def verify_math(response: str, gold: str) -> Verdict:
    raise NotImplementedError("step 3.1")


def verify_choice(response: str, gold_letter: str, letters: str = "ABCD") -> Verdict:
    raise NotImplementedError("step 3.1")


def verify(problem: Problem, response: str) -> Verdict:
    if problem.kind == ProblemKind.MULTIPLE_CHOICE:
        return verify_choice(response, problem.answer)
    return verify_math(response, problem.answer)
