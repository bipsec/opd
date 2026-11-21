"""Thin wrapper: write reports/status.md (PLAN 8.4). Standard library only, so it also runs
on the host: python3 scripts/status_digest.py [experiment ...]"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qopd.orchestration.digest import write_digest  # noqa: E402

if __name__ == "__main__":
    print(
        f"wrote {write_digest(ROOT / 'runs', ROOT / 'reports' / 'status.md', sys.argv[1:] or None)}"
    )
