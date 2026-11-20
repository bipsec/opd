"""Render ``reports/data.md`` from the stats files the data steps write (CLAUDE.md rule 6)."""

from __future__ import annotations

from pathlib import Path

from qopd.config import PROJECT_ROOT
from qopd.utils import io

REPORT = PROJECT_ROOT / "reports" / "data.md"


def write_report(ddir: Path, report: Path = REPORT) -> Path:
    """Render every section whose stats file exists; the rest say TBD."""
    from qopd.data import evalsets, pool  # noqa: PLC0415

    lines = [
        "# Data report",
        "",
        "Generated from `data/**/*_stats.json` by `qopd data report`; do not edit by hand.",
        "",
    ]
    sections = [
        ("Training pool (step 2.1)", ddir / "pool" / "pool_stats.json", pool.report_section),
        ("Evaluation suites (step 2.2)", ddir / "eval" / "eval_stats.json",
         evalsets.report_section),
        ("Contamination filter (step 2.3)", ddir / "contamination" / "contamination_stats.json",
         pool.contamination_report_section),
    ]
    for title, path, render in sections:
        if path.exists():
            lines += render(io.read_json(path))
        else:
            lines += [f"## {title}", "", "TBD (not built yet).", ""]
    for title in ("Splits and manifest (step 2.4)",):
        lines += [f"## {title}", "", f"TBD ({title.split('(')[1].rstrip(')')}).", ""]
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines))
    return report
