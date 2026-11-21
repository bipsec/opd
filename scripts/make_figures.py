"""Thin wrapper: render paper/figures/*.pdf from result files (PLAN 10.2).

Figures whose inputs do not exist yet are skipped; the paper shows a TBD box for them.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qopd.paper.figures import make_all  # noqa: E402

if __name__ == "__main__":
    for name, path in make_all().items():
        print(f"{name:9s} {path if path else 'skipped (inputs not available yet)'}")
