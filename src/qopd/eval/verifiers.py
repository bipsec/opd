"""Answer verification (PLAN 3.1).

Policy (recorded in PROGRESS.md):

- Only the final answer is graded: the text after the last ``</think>``. A response that
  opened ``<think>`` but never closed it (e.g. truncated) has no final answer and is
  incorrect. Without think tags (non-thinking mode) the whole response is graded.
- Math: the last ``\\boxed{...}`` / ``\\fbox{...}`` in the final answer (balanced braces) is
  compared with the gold by math-verify. math-verify alone may merge several boxed
  answers into a set, so the last box is cut out first. With no box, math-verify's own
  extraction on the final-answer text is the fallback (``Verdict.method`` says which).
- Multiple choice (GPQA-Diamond): the last letter A–D found by the highest-priority
  pattern: ``"answer": "C"`` (the prompt's format) > ``\\boxed{C}`` > "answer is C" /
  "Answer: (C)" > a final line that is just the letter.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from qopd.types import Problem, ProblemKind

THINK_OPEN, THINK_CLOSE = "<think>", "</think>"


@dataclass(frozen=True)
class Verdict:
    correct: bool
    extracted: str | None  # None = no parsable answer found
    reason: str = ""  # correct | wrong | no_final_answer | no_answer | unparsable
    method: str = ""  # boxed | fallback | json | letter_pattern | ...


def strip_thinking(text: str) -> str | None:
    """Final-answer text: after the last ``</think>``; None if thinking never closed."""
    if THINK_CLOSE in text:
        return text.rsplit(THINK_CLOSE, 1)[1]
    if THINK_OPEN in text:
        return None
    return text


def last_boxed(text: str) -> str | None:
    """Content of the last ``\\boxed{...}`` or ``\\fbox{...}`` (balanced braces)."""
    starts = [m for m in re.finditer(r"\\(?:boxed|fbox)\s*", text)]
    for m in reversed(starts):
        i = m.end()
        if i < len(text) and text[i] == "{":
            depth = 0
            for j in range(i, len(text)):
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                    if depth == 0:
                        return text[i + 1 : j].strip()
            continue  # unbalanced: try an earlier box
        token = re.match(r"([^\s$]+)", text[i:])  # "\boxed 5"
        if token:
            return token.group(1)
    return None


def _parse_gold(gold: str) -> list:
    from math_verify import parse  # noqa: PLC0415

    return parse(f"${gold}$", parsing_timeout=None)


def verify_math(response: str, gold: str) -> Verdict:
    from math_verify import parse, verify  # noqa: PLC0415

    final = strip_thinking(response)
    if final is None:
        return Verdict(False, None, "no_final_answer")
    box = last_boxed(final)
    if box is not None:
        pred, method = parse(f"${box}$", parsing_timeout=None), "boxed"
        if not pred:
            return Verdict(False, box, "unparsable", method)
    else:
        pred, method = parse(final, parsing_timeout=None), "fallback"
        if not pred:
            return Verdict(False, None, "no_answer", method)
    extracted = str(pred[1]) if len(pred) > 1 else str(pred[0])
    ok = bool(verify(_parse_gold(gold), pred, timeout_seconds=None))
    return Verdict(ok, extracted, "correct" if ok else "wrong", method)


_L = r"\(?\s*([A-D])\s*\)?"
MC_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("json", re.compile(r"[\"']answer[\"']\s*:\s*[\"']?\s*\(?([A-D])\)?", re.IGNORECASE)),
    ("boxed", re.compile(r"\\boxed\s*\{\s*(?:\\text\s*\{\s*)?\(?([A-D])\)?\s*\}?\s*\}")),
    (
        "answer_is",
        re.compile(
            r"answer\s*(?:is|:)\s*(?:option\s*)?\**\s*" + _L + r"(?![A-Za-z])", re.IGNORECASE
        ),
    ),
    ("final_line", re.compile(r"(?:^|\n)\s*\**\(?([A-D])\)?\**\s*\.?\s*$")),
)


def extract_choice(final: str, letters: str = "ABCD") -> tuple[str | None, str]:
    for name, pat in MC_PATTERNS:
        found = [m.group(1).upper() for m in pat.finditer(final)]
        found = [f for f in found if f in letters]
        if found:
            return found[-1], name
    return None, ""


def verify_choice(response: str, gold_letter: str, letters: str = "ABCD") -> Verdict:
    final = strip_thinking(response)
    if final is None:
        return Verdict(False, None, "no_final_answer")
    letter, method = extract_choice(final, letters)
    if letter is None:
        return Verdict(False, None, "no_answer")
    ok = letter == gold_letter.strip().upper()
    return Verdict(ok, letter, "correct" if ok else "wrong", method)


def verify(problem: Problem, response: str) -> Verdict:
    if problem.kind == ProblemKind.MULTIPLE_CHOICE:
        return verify_choice(response, problem.answer)
    return verify_math(response, problem.answer)
