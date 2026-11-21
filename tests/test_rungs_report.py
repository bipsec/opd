"""PLAN 4.x: rung table from bench/eval files; missing values stay TBD (CPU)."""

from qopd.quant.report import RUNGS, collect, write
from qopd.utils import io


def test_rungs_report(tmp_path):
    runs = tmp_path / "runs"
    io.write_json(
        runs / "teacher_bench" / "T16_n200" / "summary.json",
        {
            "load_s": 168.0,
            "load_delta_gb": 81.0,
            "kv_cache_gb": 16.0,
            "scored_tokens_per_s": 5318.0,
        },
    )
    io.write_json(
        runs / "eval" / "T16__math500__n1__t16384__s0" / "summary.json",
        {"suites": {"math500": {"pass_at_1": 0.9, "complete": True}}},
    )
    res = collect(runs)
    assert set(res) == set(RUNGS)
    t = res["T16"]
    assert (t["format"], t["load_s"], t["math500"], t["scored_tokens_per_s"]) == (
        "BF16",
        168.0,
        0.9,
        5318.0,
    )
    assert res["T8"]["load_s"] is None and res["T8"]["tool"] == "llmcompressor"
    md = write(res, tmp_path / "rungs.md", tmp_path / "rungs.json").read_text()
    assert "| T16 | BF16 |" in md and "90.0%" in md and "| T8 |" in md and "TBD" in md
    assert io.read_json(tmp_path / "rungs.json")["T16"]["load_delta_gb"] == 81.0
