"""Thin wrapper around qopd.utils.env (PLAN step 0.1).

Usage:
    python scripts/env_probe.py [--label host|container] [--out-dir reports]

Writes reports/env.<label>.json, then re-renders reports/env.md from every
env.*.json present, so host and container probes appear side by side.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # works before the package is pip-installed

from qopd.utils.env import EnvReport, probe, render_markdown  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label", default=None)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "reports")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rep = probe(args.label)
    (args.out_dir / f"env.{rep.label}.json").write_text(json.dumps(rep.to_dict(), indent=2))

    reports = [
        EnvReport.from_dict(json.loads(p.read_text()))
        for p in sorted(args.out_dir.glob("env.*.json"))
    ]
    (args.out_dir / "env.md").write_text(render_markdown(reports))
    print(f"wrote {args.out_dir / 'env.md'} ({', '.join(r.label for r in reports)})")


if __name__ == "__main__":
    main()
