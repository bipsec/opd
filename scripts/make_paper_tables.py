"""Thin wrapper: regenerate paper/tables/*.tex and paper/tables/numbers.tex from files."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qopd.paper.tables import count_tbd, write_all  # noqa: E402

if __name__ == "__main__":
    paths = write_all()
    sections = sorted((ROOT / "paper" / "sections").glob("*.tex"))
    print(f"wrote {len(paths)} files to paper/tables/")
    print(f"TBD placeholders: tables {count_tbd(paths)}, sections {count_tbd(sections)}")
