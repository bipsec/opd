"""PLAN 8.4: status digest states, ETA and anomaly detection (CPU, synthetic run dirs)."""

import json
import math
import os
import time

from qopd.orchestration.digest import anomalies, digest_experiment, render, write_digest
from qopd.orchestration.queue import RunSpec, run_id, write_queue


def spec(i, kind="opd", student="S2", est=6.0, optional=False):
    return RunSpec(
        "e",
        run_id(kind, student, "T16", i),
        kind,
        student,
        "T16",
        i,
        "cmd",
        "spec",
        optional=optional,
        estimate_key=f"{kind}:{student}",
        estimate_hours=est,
    )


def rows(n, **over):
    out = []
    for r in range(1, n + 1):
        row = {
            "round": r,
            "loss": 0.1,
            "reward_mean": -0.2,
            "entropy_mean": 0.8,
            "repetition_rate": 0.01,
            "truncation_rate": 0.05,
            "round_seconds": 600.0,
        }
        for k, f in over.items():
            row[k] = f(r)
        out.append(row)
    return out


def test_anomaly_rules():
    assert anomalies(rows(5), "") == []
    a = anomalies(rows(5, loss=lambda r: math.nan if r == 4 else 0.1), "")
    assert any("non-finite" in x and "[4]" in x for x in a)
    assert any(
        "entropy collapse from round 3" in x
        for x in anomalies(rows(5, entropy_mean=lambda r: 0.8 if r < 3 else 0.1), "")
    )
    assert any(
        "repetition spike" in x
        for x in anomalies(rows(5, repetition_rate=lambda r: 0.01 if r < 5 else 0.3), "")
    )
    assert any(
        "round-1 truncation" in x for x in anomalies(rows(2, truncation_rate=lambda r: 0.4), "")
    )
    assert any(
        "2 out-of-memory" in x
        for x in anomalies([], "CUDA error: out of memory\\n...torch.OutOfMemoryError")
    )


def test_states_eta_and_render(tmp_path):
    specs = [spec(0), spec(1), spec(2), spec(3), spec(4, optional=True)]
    write_queue(specs, tmp_path, "e", {})
    now = time.time()
    for i, s in enumerate(specs[:3]):
        d = tmp_path / "e" / s.run_id
        d.mkdir(parents=True)
        (d / "queue_started.json").write_text(json.dumps({"t": now - 3 * 3600}))
        with (d / "metrics.jsonl").open("w") as f:
            for r in rows(3):
                f.write(json.dumps(r) + "\n")
        (d / "state.json").write_text("{}")
        if i == 0:
            (d / "DONE").write_text("x")  # done, took ~3 h
        if i == 2:
            os.utime(d / "state.json", (now - 3600, now - 3600))  # no heartbeat for 1 h
    dg = digest_experiment(tmp_path, "e", now)
    states = {r.spec.run_id: r.state for r in dg["runs"]}
    assert list(states.values())[:4] == ["done", "running", "stalled", "queued"]
    assert dg["counts"] == {"done": 1, "running": 1, "stalled": 1, "queued": 1}
    assert abs(dg["eta_hours"] - 3 * 3.0) < 0.05  # measured 3 h replaces the 6 h estimate
    md = render([dg])
    assert "stalled" in md and "Anomalies: none" in md and "Alert thresholds" in md
    out = write_digest(tmp_path, tmp_path / "status.md", now=now)
    assert out.read_text().startswith("# Run status digest")


def test_no_queues(tmp_path):
    assert "No planned experiments" in write_digest(tmp_path, tmp_path / "s.md").read_text()
