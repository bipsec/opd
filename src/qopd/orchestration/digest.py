"""Status digest (PLAN 8.4): reports/status.md from the queues and run directories.

Per planned experiment (``runs/<exp>/queue.json``): done / running / stalled / failed / queued
counts, an ETA (measured wall-clock per kind/student once runs finish, else the grid's
estimates), the latest metrics of running runs, and anomalies found in each run's files:

- non-finite loss or reward (NaN / inf);
- entropy collapse: entropy below ``ENTROPY_FLOOR`` nats or below ``ENTROPY_FRAC`` of round 1;
- repetition spike: repetition rate above ``REPETITION_MAX`` or ``REPETITION_X`` x round 1;
- truncation above the doc's alert (opd.truncation_alert, default 0.30);
- OOM restarts: out-of-memory messages in stdout.log / tmux.log, and resume count.

Running vs stalled uses the ``state.json`` heartbeat (tmux is only visible on the host).
Thresholds are operational alerts, not scientific choices; they are printed in the report.
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from qopd.orchestration.queue import RunSpec, list_queues, load_queue, run_dir

ENTROPY_FLOOR = 0.05
ENTROPY_FRAC = 0.25
REPETITION_MAX = 0.20
REPETITION_X = 3.0
TRUNCATION_ALERT = 0.30
STALL_MINUTES = 30
_OOM = re.compile(r"out of memory|OutOfMemoryError|CUDA_ERROR_OUT_OF_MEMORY", re.IGNORECASE)


@dataclass
class RunDigest:
    spec: RunSpec
    state: str  # done | running | stalled | failed | queued
    last: dict[str, Any] = field(default_factory=dict)
    anomalies: list[str] = field(default_factory=list)
    hours: float | None = None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _finite(x: Any) -> bool:
    return isinstance(x, int | float) and math.isfinite(x)


def anomalies(
    rows: list[dict[str, Any]], logs: str, truncation_alert: float = TRUNCATION_ALERT
) -> list[str]:
    out: list[str] = []
    bad = [
        r.get("round", r.get("step"))
        for r in rows
        if any(k in r and not _finite(r[k]) for k in ("loss", "reward_mean"))
    ]
    if bad:
        out.append(f"non-finite loss/reward at round(s) {bad[:5]}")
    ents = [(r.get("round"), r["entropy_mean"]) for r in rows if _finite(r.get("entropy_mean"))]
    if ents:
        first = ents[0][1]
        low = [
            rd for rd, e in ents if e < ENTROPY_FLOOR or (first > 0 and e < ENTROPY_FRAC * first)
        ]
        if low:
            out.append(
                f"entropy collapse from round {low[0]} (latest {ents[-1][1]:.3f} nats, "
                f"round 1 {first:.3f})"
            )
    reps = [
        (r.get("round"), r["repetition_rate"]) for r in rows if _finite(r.get("repetition_rate"))
    ]
    if reps:
        base = reps[0][1]
        spikes = [
            rd for rd, v in reps if v > REPETITION_MAX or (base > 0 and v > REPETITION_X * base)
        ]
        if spikes:
            out.append(f"repetition spike at round(s) {spikes[:5]} (latest {reps[-1][1]:.1%})")
    tr = [(r.get("round"), r["truncation_rate"]) for r in rows if _finite(r.get("truncation_rate"))]
    if tr and tr[0][1] > truncation_alert:
        out.append(
            f"round-1 truncation {tr[0][1]:.0%} > {truncation_alert:.0%} (doc: revisit limits)"
        )
    n_oom = len(_OOM.findall(logs))
    if n_oom:
        out.append(f"{n_oom} out-of-memory message(s) in logs")
    return out


def digest_run(spec: RunSpec, runs_root: Path, now: float | None = None) -> RunDigest:
    d = run_dir(runs_root, spec)
    now = now or time.time()
    started = d / "queue_started.json"
    state_f = d / "state.json"
    if (d / "DONE").exists():
        st = "done"
    elif started.exists():
        age = (now - state_f.stat().st_mtime) / 60 if state_f.exists() else math.inf
        st = "running" if age < STALL_MINUTES else "stalled"
    else:
        st = "queued"
    rows = _read_jsonl(d / "metrics.jsonl")
    logs = "".join(
        p.read_text(errors="replace") for p in (d / "stdout.log", d / "tmux.log") if p.exists()
    )
    an = anomalies(rows, logs)
    resumes = logs.count("resumed at round") + logs.count("resumed from")
    if resumes:
        an.append(f"resumed {resumes} time(s)")
    hours = None
    if started.exists() and (d / "DONE").exists():
        hours = ((d / "DONE").stat().st_mtime - json.loads(started.read_text())["t"]) / 3600
    last = rows[-1] if rows else {}
    keep = (
        "round",
        "step",
        "reward_mean",
        "entropy_mean",
        "repetition_rate",
        "truncation_rate",
        "round_seconds",
        "loss",
    )
    return RunDigest(spec, st, {k: last[k] for k in keep if k in last}, an, hours)


def digest_experiment(runs_root: Path, experiment: str, now: float | None = None) -> dict[str, Any]:
    specs = load_queue(runs_root, experiment)
    runs = [digest_run(s, runs_root, now) for s in specs]
    measured: dict[str, list[float]] = {}
    for r in runs:
        if r.hours is not None:
            measured.setdefault(r.spec.estimate_key, []).append(r.hours)
    eta, unknown = 0.0, False
    for r in runs:
        if r.spec.optional or r.state == "done":
            continue
        h = (
            sum(measured[r.spec.estimate_key]) / len(measured[r.spec.estimate_key])
            if r.spec.estimate_key in measured
            else r.spec.estimate_hours
        )
        if h is None:
            unknown = True
        else:
            eta += h
    counts: dict[str, int] = {}
    for r in runs:
        if not r.spec.optional or r.state != "queued":
            counts[r.state] = counts.get(r.state, 0) + 1
    return {
        "experiment": experiment,
        "runs": runs,
        "counts": counts,
        "eta_hours": None if unknown else eta,
        "measured_keys": sorted(measured),
    }


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.3g}"
    return str(v)


def render(digests: list[dict[str, Any]]) -> str:
    lines = [
        "# Run status digest (step 8.4)",
        "",
        f"Generated {time.strftime('%Y-%m-%d %H:%M %Z')} by `qopd report status` from "
        "`runs/<exp>/queue.json` and the run directories; do not edit by hand.",
        "",
    ]
    if not digests:
        lines += ["No planned experiments (`qopd queue plan <grid>`).", ""]
    for dg in digests:
        c = dg["counts"]
        eta = (
            "unknown (some runs have no estimate)"
            if dg["eta_hours"] is None
            else f"{dg['eta_hours']:.1f} h"
        )
        src = (
            "measured where available, else grid estimates"
            if dg["measured_keys"]
            else "grid estimates"
        )
        lines += [
            f"## {dg['experiment']}",
            "",
            "Counts: "
            + ", ".join(f"{k} {v}" for k, v in sorted(c.items()))
            + f". Remaining required work ≈ {eta} ({src}).",
            "",
        ]
        active = [r for r in dg["runs"] if r.state in ("running", "stalled")]
        if active:
            lines += ["| Run | state | latest metrics |", "| --- | --- | --- |"]
            for r in active:
                lines.append(
                    f"| {r.spec.run_id} | {r.state} | "
                    + ", ".join(f"{k}={_fmt(v)}" for k, v in r.last.items())
                    + " |"
                )
            lines.append("")
        flagged = [r for r in dg["runs"] if r.anomalies]
        lines += ["Anomalies: " + ("none" if not flagged else ""), ""]
        for r in flagged:
            lines.append(f"- **{r.spec.run_id}** ({r.state}): " + "; ".join(r.anomalies))
        done = [r for r in dg["runs"] if r.state == "done"]
        if done:
            lines += [
                "",
                f"Completed ({len(done)}): "
                + ", ".join(
                    f"{r.spec.run_id} ({r.hours:.1f} h)" if r.hours else r.spec.run_id for r in done
                ),
            ]
        lines.append("")
    lines += [
        "Alert thresholds (operational): entropy < "
        f"{ENTROPY_FLOOR} nats or < {ENTROPY_FRAC:.0%} of round 1; repetition > "
        f"{REPETITION_MAX:.0%} or > {REPETITION_X:g}x round 1; round-1 truncation > "
        f"{TRUNCATION_ALERT:.0%}; stalled = no state.json update for {STALL_MINUTES} min.",
        "",
    ]
    return "\n".join(lines)


def write_digest(
    runs_root: Path, report: Path, experiments: list[str] | None = None, now: float | None = None
) -> Path:
    exps = experiments or (list_queues(runs_root) if runs_root.exists() else [])
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(render([digest_experiment(runs_root, e, now) for e in exps]))
    return report
