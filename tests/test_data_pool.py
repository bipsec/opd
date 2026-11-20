"""PLAN 2.1: filters, schema I/O and report rendering (CPU, no downloads)."""

import pytest

from qopd.data import filters
from qopd.data.pool import write_report
from qopd.data.sources import (
    SOURCES,
    DatasetNotFoundError,
    get_source,
    read_parquet,
    to_problems,
    write_parquet,
)
from qopd.types import Problem
from qopd.utils import io


def P(i, q, a="1"):
    return Problem(f"t/{i}", q, a, "t")


def test_normalize_text_keeps_lines():
    raw = "  Find  x.\r\n\r\n\r\n\tIf x = 2  "
    assert filters.normalize_text(raw) == "Find x.\n\nIf x = 2"
    res = filters.normalize([P(0, raw, " 2 "), P(1, "   ", "1"), P(2, "q", "")])
    assert [p.answer for p in res.kept] == ["2"]
    assert res.dropped == {"empty_problem": 1, "empty_answer": 1}


@pytest.mark.parametrize(
    ("answer", "kind"),
    [
        ("42", "integer"),
        ("-8", "integer"),
        ("\\frac{79}{20}", "number"),
        ("-13.5", "number"),
        ("\\sqrt{2} - 1", "closed_form"),
        ("\\dfrac{5\\pi}{6}", "closed_form"),
    ],
)
def test_classify_kept(answer, kind):
    assert filters.classify_answer(answer) == kind


@pytest.mark.parametrize("answer", ["(-5, 6)", "\\{1, 2\\}", "x^2 + 1", "y=5", "3 - 2i", "C"])
def test_classify_dropped(answer):
    assert filters.classify_answer(answer) not in filters.ANSWER_TYPES


def test_answer_type_stage_records_type():
    res = filters.keep_parsable_answers([P(0, "q", "7"), P(1, "q", "(1,2)")])
    assert [p.meta["answer_type"] for p in res.kept] == ["integer"]
    assert res.n_in == len(res.kept) + sum(res.dropped.values())


def test_self_verifies():
    assert filters.self_verifies("\\frac{1}{2}")
    assert filters.self_verifies("35")


def test_dedup_exact_and_near():
    base = (
        "Let a and b be positive integers such that a plus b equals twenty. "
        "Find the largest product ab."
    )
    ps = [
        P(0, base),
        P(1, base.upper().replace(" ", "  ")),  # exact after case/space folding
        P(2, base.replace("largest", "maximum")),  # near duplicate
        P(3, "A completely different question about triangles and their inscribed circles today."),
    ]
    ex = filters.dedup_exact(ps)
    assert [p.id for p in ex.kept] == ["t/0", "t/2", "t/3"]
    near = filters.dedup_near(ex.kept, threshold=0.8, num_perm=128, ngram=5)
    assert near.dropped.get("near_duplicate", 0) <= 1
    assert "t/3" in [p.id for p in near.kept] and "t/0" in [p.id for p in near.kept]


def test_conflicting_duplicates_dropped():
    q = "Find the number of ordered pairs of integers with some property here."
    ps = [
        P(0, q, "312"),
        P(1, q, "179"),
        P(2, "Other problem text.", "\\frac{1}{2}"),
        P(3, "other problem   text.", "1/2"),
    ]
    res = filters.dedup_exact(ps)
    assert [p.id for p in res.kept] == ["t/2"]  # 1/2 == \frac{1}{2}: agree, dedup normally
    assert res.dropped == {"duplicate_with_conflicting_answers": 2, "exact_duplicate": 1}
    keep_all = filters.dedup_exact(ps, drop_conflicts=False)
    assert [p.id for p in keep_all.kept] == ["t/0", "t/2"]


def test_near_dedup_threshold_respected():
    a = " ".join(f"w{i}" for i in range(60))
    b = " ".join(f"w{i}" for i in range(59)) + " changed"  # Jaccard of 5-grams ~0.93
    c = " ".join(f"v{i}" for i in range(60))
    res = filters.dedup_near([P(0, a), P(1, b), P(2, c)], threshold=0.8)
    assert [p.id for p in res.kept] == ["t/0", "t/2"]
    assert res.dropped_ids == [("t/1", "near_duplicate")]


def test_parquet_round_trip(tmp_path):
    spec = SOURCES["deepscaler_preview"]
    ps = to_problems(spec, [{"problem": "q1", "answer": "1"}, {"problem": "q2", "answer": "2"}])
    assert ps[0].id == "deepscaler_preview/000000" and ps[1].meta == {"row": 1}
    sha = write_parquet(ps, tmp_path / "x.parquet")
    assert read_parquet(tmp_path / "x.parquet") == ps
    assert sha == io.sha256_file(tmp_path / "x.parquet")


def test_unknown_source():
    with pytest.raises(DatasetNotFoundError):
        get_source("not_a_dataset")


def test_report_renders_from_stats(tmp_path):
    stats = {
        "step": "2.1",
        "source": {
            "hf_repo": "org/ds",
            "revision": "abc",
            "file": "f.json",
            "file_sha256": "0" * 64,
            "license": "mit",
            "rows": 3,
            "notes": "n",
            "name": "ds",
        },
        "config": {
            "keep_answer_types": ["integer"],
            "near_dup": {"ngram": 5, "num_perm": 128, "threshold": 0.8},
        },
        "stages": [{"stage": "normalize", "in": 3, "kept": 2, "dropped": {"empty_answer": 1}}],
        "pool": {
            "file": "data/pool/pool.parquet",
            "sha256": "1" * 64,
            "n": 2,
            "by_source": {"ds": 2},
            "by_answer_type": {"integer": 2},
        },
    }
    io.write_json(tmp_path / "pool" / "pool_stats.json", stats)
    out = write_report(tmp_path, tmp_path / "data.md")
    text = out.read_text()
    assert "| normalize | 3 | 2 | empty_answer: 1 |" in text
    assert "**2** problems" in text
    assert "TBD (step 2.2)" in text
