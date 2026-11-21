"""Host-side job queue (PLAN 8.1). Standard library only: run with the host's python3.

    python3 scripts/queue_host.py status e2 [--include-optional]
    python3 scripts/queue_host.py run e2 [--include-optional] [--dry-run]

Plan the queue first, inside the container: `qopd queue plan e2`. Start the supervisor in its
own tmux session so it survives logout:
    tmux new-session -d -s queue_e2 'python3 scripts/queue_host.py run e2'
Each run gets its own tmux session (name = run id); logs go to runs/<exp>/<run_id>/tmux.log.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qopd.orchestration.queue import (  # noqa: E402
    JobState,
    docker_prefix,
    load_queue,
    next_queued,
    status,
    supervise,
)


def print_status(experiment: str, runs_root: Path, include_optional: bool) -> None:
    specs = load_queue(runs_root, experiment)
    st = status(specs, runs_root, include_optional)
    for spec, state in st.rows:
        opt = " (optional)" if spec.optional else ""
        print(f"{state.value:8s} {spec.run_id}{opt}")
    counts = ", ".join(f"{s.value} {st.count(s)}" for s in JobState)
    eta = (
        "unknown (some runs have no estimate)" if st.eta_hours is None else f"{st.eta_hours:.1f} h"
    )
    print(f"\n{counts}; remaining ≈ {eta} ({st.eta_source})")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("action", choices=["status", "run"])
    ap.add_argument("experiment")
    ap.add_argument("--include-optional", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="show the next launch and exit")
    ap.add_argument("--poll", type=float, default=30.0, help="seconds between checks")
    ap.add_argument("--runs-root", type=Path, default=ROOT / "runs")
    ap.add_argument("--hf-cache", type=Path, default=Path.home() / ".cache" / "huggingface")
    args = ap.parse_args()

    if args.action == "status":
        print_status(args.experiment, args.runs_root, args.include_optional)
        return
    specs = load_queue(args.runs_root, args.experiment)
    prefix = docker_prefix(ROOT, args.hf_cache)
    if args.dry_run:
        nxt = next_queued(specs, args.runs_root, args.include_optional)
        print(f"next: {nxt.run_id if nxt else 'nothing'}\n{prefix} {nxt.command if nxt else ''}")
        return
    supervise(specs, args.runs_root, prefix, args.include_optional, args.poll)


if __name__ == "__main__":
    main()
