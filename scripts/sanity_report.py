"""Thin wrapper: render reports/sanity_eval.md from the 3.3 run directories (PLAN 3.3)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qopd.eval.report import write_sanity_report  # noqa: E402

RUNS = {m: ROOT / "runs" / "eval" / f"{m}__gsm8k+math500__n1__t16384__s0" for m in ("S2", "S1")}
EXPECTED = {"gsm8k": 1319, "math500": 500}

PUBLISHED = ROOT / "reports" / "published" / "qwen3_tech_report.json"


def published() -> dict[tuple[str, str], str]:
    """(model, suite) -> cell text, read from reports/published/qwen3_tech_report.json."""
    import json

    if not PUBLISHED.exists():
        return {}
    data = json.loads(PUBLISHED.read_text())
    th = data["thinking_mode_post_trained"]
    out = {(m, "math500"): f"{v} (Table 19)" for m, v in th["MATH-500"].items()}
    out |= {(m, "gsm8k"): "n/a (base models only, Table 8)" for m in ("S1", "S2")}
    return out


def truncation_analysis() -> Path:
    """Per student, MATH-500: pass@1, truncation rate, accuracy among responses that finished
    within the token limit, and the bound if every truncated response were correct."""
    import json

    out = {}
    for m, d in RUNS.items():
        f = d / "samples.jsonl"
        if not f.exists():
            continue
        rows = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
        r = [x for x in rows if x["suite"] == "math500"]
        fin = [x for x in r if x["finish_reason"] != "length"]
        if not r or not fin:
            continue
        n_trunc = len(r) - len(fin)
        out[m] = {
            "n": len(r),
            "pass_at_1": sum(x["correct"] for x in r) / len(r),
            "truncated": n_trunc / len(r),
            "acc_finished": sum(x["correct"] for x in fin) / len(fin),
            "upper_bound": (sum(x["correct"] for x in r) + n_trunc) / len(r),
            "no_answer_among_finished": sum(x["reason"] not in ("correct", "wrong") for x in fin),
        }
    path = ROOT / "reports" / "sanity_truncation.json"
    path.write_text(
        json.dumps({"suite": "math500", "max_tokens": 16384, "students": out}, indent=2)
    )
    return path


if __name__ == "__main__":
    print(f"wrote {write_sanity_report(RUNS, EXPECTED, published())}")
    print(f"wrote {truncation_analysis()}")
