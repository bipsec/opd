"""PLAN 7.4: fidelity JSONs + reports/e1.md from saved pools (CPU, synthetic pool)."""

import pytest

torch = pytest.importorskip("torch")

from qopd.fidelity.pool import build_pool, load_pool, score_pool, student_stats, write_student_stats  # noqa: E402
from qopd.fidelity.report import compute_all, load_all, write_report  # noqa: E402
from tests.test_opd_loop import PROBLEMS, FakeEngine, FakeTeacher, cfg, prompt_of, tiny  # noqa: E402


def test_e1_pipeline(tmp_path):
    root = tmp_path / "e1"
    pool = root / "S2" / "pool"
    student = tiny(0)
    build_pool(FakeEngine(student), PROBLEMS[:5], prompt_of, cfg().rollout, 2, pool)
    rs = load_pool(pool)
    write_student_stats(pool, rs, *student_stats(student, rs, 0, "cpu"))
    for rung, seed in (("T16", 99), ("T8", 99), ("T3", 5)):
        score_pool(FakeTeacher(seed), rs, 4, pool / "scores" / rung)
    (root / "S1" / "pool").mkdir(parents=True)  # S1 not scored yet: skipped
    out = compute_all(root)
    assert set(out) == {"S2"} and set(out["S2"]) == {"T8", "T3"}
    assert (root / "S2" / "fidelity" / "T8.json").exists()
    res = load_all(root)
    md = write_report(res, None, tmp_path / "e1.md", root).read_text()
    assert "## Student S2" in md and "| T8 |" in md and "| T3 |" in md
    assert "TBD (definition pending)" in md  # informativeness not implemented
    assert md.index("| T8 |") < md.index("| T3 |")  # rung order, not alphabetical


def test_empty_report(tmp_path):
    md = write_report({}, None, tmp_path / "e1.md", tmp_path / "e1").read_text()
    assert "TBD: no scored pools yet" in md
