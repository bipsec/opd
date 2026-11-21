"""Thin wrapper: render reports/sanity_eval.md from the 3.3 run directories (PLAN 3.3)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qopd.eval.report import write_sanity_report  # noqa: E402

RUNS = {m: ROOT / "runs" / "eval" / f"{m}__gsm8k+math500__n1__t16384__s0" for m in ("S2", "S1")}
EXPECTED = {"gsm8k": 1319, "math500": 500}

if __name__ == "__main__":
    print(f"wrote {write_sanity_report(RUNS, EXPECTED)}")
