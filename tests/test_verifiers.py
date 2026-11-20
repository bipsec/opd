"""PLAN 3.1: hand-written correct / incorrect / malformed answers, thinking blocks, boxes."""

import re
from pathlib import Path

import pytest

from qopd.eval.verifiers import (
    extract_choice,
    last_boxed,
    strip_thinking,
    verify,
    verify_choice,
    verify_math,
)
from qopd.types import Problem, ProblemKind

T = "<think>\nLet me work this out. Maybe \\boxed{99}? No.\n</think>\n\n"


# --- thinking blocks ------------------------------------------------------------------


def test_strip_thinking():
    assert strip_thinking("<think>a</think>final") == "final"
    assert strip_thinking("<think>a</think>x<think>b</think>last") == "last"
    assert strip_thinking("<think>never closed \\boxed{3}") is None
    assert strip_thinking("no tags \\boxed{3}") == "no tags \\boxed{3}"


def test_box_inside_thinking_does_not_count():
    v = verify_math(T + "I am not sure.", "99")
    assert not v.correct and v.reason == "no_answer"


def test_truncated_thinking_is_no_answer():
    v = verify_math("<think>so the answer is \\boxed{42}", "42")
    assert (v.correct, v.reason) == (False, "no_final_answer")


# --- boxed extraction ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "box"),
    [
        ("\\boxed{42}", "42"),
        ("a \\boxed{1} b \\boxed{\\frac{1}{2}}", "\\frac{1}{2}"),
        ("\\boxed{\\left( 3, \\frac{\\pi}{2} \\right)}", "\\left( 3, \\frac{\\pi}{2} \\right)"),
        ("\\fbox{7}", "7"),
        ("\\boxed 5 is it", "5"),
        ("\\boxed{3} then broken \\boxed{4", "3"),
        ("nothing", None),
    ],
)
def test_last_boxed(text, box):
    assert last_boxed(text) == box


# --- math: correct --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("gold", "final"),
    [
        ("18", "She earns \\boxed{18} dollars."),
        ("27.0", "\\boxed{27}"),  # AMC golds are floats
        ("\\frac{1}{2}", "\\boxed{0.5}"),
        ("\\frac{1}{2}", "\\boxed{\\dfrac{1}{2}}"),
        ("\\sqrt{2} - 1", "\\boxed{-1 + \\sqrt{2}}"),
        ("\\left( 3, \\frac{\\pi}{2} \\right)", "\\boxed{(3, \\frac{\\pi}{2})}"),
        ("74^\\circ", "\\boxed{74^\\circ}"),
        ("1000", "\\boxed{1000}"),
        ("-\\frac{1}{21}", "\\boxed{-\\frac{1}{21}}"),
        ("3", "First guess \\boxed{1/2}, corrected: \\boxed{3}"),  # last box wins
    ],
)
def test_math_correct(gold, final):
    v = verify_math(T + final, gold)
    assert v.correct, v
    assert v.method == "boxed"


def test_math_fallback_without_box():
    v = verify_math(T + "The answer is 72.", "72")
    assert v.correct and v.method == "fallback"


# --- math: incorrect / malformed ------------------------------------------------------------


@pytest.mark.parametrize(
    ("gold", "final", "reason"),
    [
        ("18", "\\boxed{19}", "wrong"),
        ("\\frac{1}{2}", "\\boxed{\\frac{1}{3}}", "wrong"),
        ("\\frac{1}{2}", "\\boxed{1/2} but actually \\boxed{3}", "wrong"),  # last box is graded
        ("204", "I could not solve it.", "no_answer"),
        ("204", "", "no_answer"),
        ("5", "\\boxed{}", "unparsable"),
    ],
)
def test_math_incorrect(gold, final, reason):
    v = verify_math(T + final, gold)
    assert not v.correct
    assert v.reason == reason, v


# --- multiple choice ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("final", "letter", "method"),
    [
        ('{"answer": "C"}', "C", "json"),
        ('"answer": "(B)"', "B", "json"),
        ("Therefore \\boxed{D}", "D", "boxed"),
        ("\\boxed{\\text{A}}", "A", "boxed"),
        ("The answer is (B).", "B", "answer_is"),
        ("Answer: C", "C", "answer_is"),
        ("Reasoning...\n\nA", "A", "final_line"),
        ('A is wrong, B too. "answer": "D"', "D", "json"),  # format beats stray letters
    ],
)
def test_extract_choice(final, letter, method):
    assert extract_choice(final) == (letter, method)


def test_choice_verdicts():
    assert verify_choice(T + '"answer": "C"', "C").correct
    assert not verify_choice(T + '"answer": "A"', "C").correct
    assert verify_choice(T + "I think it is a tricky one.", "C").reason == "no_answer"
    assert verify_choice("<think>answer is C", "C").reason == "no_final_answer"
    assert verify_choice(T + "\\boxed{E}", "C").reason == "no_answer"  # E not an option


def test_dispatch_by_kind():
    mc = Problem("g/0", "q", "B", "gpqa", ProblemKind.MULTIPLE_CHOICE)
    assert verify(mc, T + '"answer": "B"').correct
    assert verify(Problem("m/0", "q", "7", "m"), T + "\\boxed{7}").correct


# --- real eval golds (skipped if the suites are not built) ----------------------------------


EVAL_DIR = Path(__file__).resolve().parents[1] / "data" / "eval"


@pytest.mark.parametrize(
    "suite", ["gsm8k", "math500", "amc23", "aime24", "aime25", "aime26", "hmmt_feb26"]
)
def test_every_gold_is_gradable(suite):
    from qopd.data.sources import read_parquet

    f = EVAL_DIR / f"{suite}.parquet"
    if not f.exists():
        pytest.skip(f"{suite} not built")
    items = read_parquet(f)
    bad = [
        p.id for p in items if not verify(p, T + f"So the answer is \\boxed{{{p.answer}}}.").correct
    ]
    assert bad == [], bad[:10]


@pytest.mark.parametrize("suite", ["gsm8k", "aime24", "aime25", "aime26", "math500"])
def test_off_by_one_is_wrong(suite):
    from qopd.data.sources import read_parquet

    f = EVAL_DIR / f"{suite}.parquet"
    if not f.exists():
        pytest.skip(f"{suite} not built")
    ints = [p for p in read_parquet(f) if re.fullmatch(r"-?\d+", p.answer)]  # plain digits only
    accepted = [p.id for p in ints if verify(p, T + f"\\boxed{{{int(p.answer) + 1}}}").correct]
    assert ints and accepted == [], accepted[:10]
