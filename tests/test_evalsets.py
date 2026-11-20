"""PLAN 2.2: suite converters, GPQA shuffling, config consistency, report rows (CPU)."""

import pytest

from qopd.config import CONFIG_ROOT, EvalCfg, load
from qopd.data.evalsets import (
    EVAL_SOURCES,
    LETTERS,
    annotate,
    gpqa_problem,
    gsm8k_answer,
    report_section,
)
from qopd.types import Problem, ProblemKind


def test_gsm8k_answer():
    assert gsm8k_answer("She makes 9 * 2 = 18.\n#### 18") == "18"
    assert gsm8k_answer("... #### 1,250") == "1250"
    with pytest.raises(ValueError):
        gsm8k_answer("no marker")


GPQA_ROW = {
    "Record ID": "rec123",
    "Question": "Which particle mediates the electromagnetic force?",
    "Correct Answer": "Photon",
    "Incorrect Answer 1": "Gluon",
    "Incorrect Answer 2": "W boson",
    "Incorrect Answer 3": "Higgs boson",
    "High-level domain": "Physics",
}


def test_gpqa_shuffle_deterministic_and_correct():
    a = gpqa_problem("gpqa_diamond", 0, GPQA_ROW)
    b = gpqa_problem("gpqa_diamond", 0, GPQA_ROW)
    assert a == b
    assert a.kind is ProblemKind.MULTIPLE_CHOICE
    assert a.answer in LETTERS
    assert f"{a.answer}) Photon" in a.problem
    for letter in LETTERS:
        assert f"\n{letter}) " in a.problem
    other = gpqa_problem("gpqa_diamond", 0, GPQA_ROW, seed=7)
    assert other.answer in LETTERS  # a different seed may reorder


def test_gpqa_letters_spread():
    rows = [{**GPQA_ROW, "Record ID": f"r{i}"} for i in range(200)]
    letters = {gpqa_problem("g", i, r).answer for i, r in enumerate(rows)}
    assert letters == set(LETTERS)


def test_converters_on_sample_rows():
    m = EVAL_SOURCES["math500"].convert(
        "math500",
        3,
        {
            "problem": " What  is 1+1? ",
            "answer": "2",
            "unique_id": "test/x.json",
            "subject": "Algebra",
            "level": "1",
        },
    )
    assert (m.id, m.problem, m.answer, m.meta["subject"]) == (
        "math500/0003",
        "What is 1+1?",
        "2",
        "Algebra",
    )
    g = EVAL_SOURCES["gsm8k"].convert("gsm8k", 0, {"question": "q", "answer": "x\n#### 7"})
    assert g.answer == "7"
    h = EVAL_SOURCES["hmmt_feb26"].convert(
        "hmmt_feb26", 0, {"problem_idx": "1", "problem": "p", "answer": "-\\frac{1}{21}"}
    )
    assert h.meta["problem_idx"] == "1"


def test_annotate_math_only():
    p = annotate(Problem("s/0", "q", "\\frac{1}{2}", "s"))
    assert p.meta == {"answer_type": "number", "self_verifies": True}
    mc = Problem("g/0", "q", "B", "g", ProblemKind.MULTIPLE_CHOICE)
    assert annotate(mc) is mc


def test_sources_match_eval_config():
    ev = load(CONFIG_ROOT / "eval" / "default.yaml", cls=EvalCfg)
    assert set(ev.suites) == set(EVAL_SOURCES)
    for name, suite in ev.suites.items():
        src = EVAL_SOURCES[name]
        assert suite.source == src.hf_repo
        assert suite.gated == src.gated
        assert (suite.kind == "mc") == (src.kind is ProblemKind.MULTIPLE_CHOICE)
        assert len(src.revision) == 40  # pinned full commit


def test_report_marks_blocked_suites():
    stats = {
        "suites": {
            "gpqa_diamond": {
                "status": "blocked",
                "reason": "gated",
                "hf_repo": "Idavidrein/gpqa",
                "revision": "8" * 40,
                "license": "cc-by-4.0",
            },
            "aime26": {
                "status": "ok",
                "n": 30,
                "hf_repo": "MathArena/aime_2026",
                "revision": "d" * 40,
                "license": "cc-by-nc-sa-4.0",
                "answer_types": {"integer": 30},
                "not_self_verifying": 0,
            },
        }
    }
    text = "\n".join(report_section(stats))
    assert "| gpqa_diamond | **blocked**: gated | TBD |" in text
    assert "| aime26 | ok | 30 |" in text
