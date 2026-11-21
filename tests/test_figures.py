"""PLAN 10.2: figures render from (synthetic) result files and skip when inputs are missing."""

import random

import pytest

pytest.importorskip("matplotlib")

from qopd.paper.figures import RUNG_ORDER, make_all
from qopd.utils import io

BUCKETS = ["[0,0.1)", "[0.1,0.5)", "[0.5,1)", "[1,2)", "[2,inf)"]


def synthetic(tmp_path):
    """SYNTHETIC numbers for rendering tests only - never written outside tmp_path."""
    rnd = random.Random(0)
    e1 = tmp_path / "e1"
    for st in ("S1", "S2"):
        for i, r in enumerate(RUNG_ORDER[1:]):
            m = {
                "pearson": 1 - 0.05 * i,
                "mae": 0.02 * (i + 1),
                "topk_kl": 0.01 * (i + 1),
                "top1_agreement": 1 - 0.03 * i,
                "sign_flip_rate": 0.01 * i,
            }
            m["by_entropy_bucket"] = {
                b: {"mae": 0.01 * (i + 1) * (j + 1)} for j, b in enumerate(BUCKETS)
            }
            io.write_json(e1 / st / "fidelity" / f"{r}.json", m)
    res = tmp_path / "results"
    scores = []
    for st in ("S1", "S2"):
        for t in (*RUNG_ORDER[:5], "D16", "D8", "D4"):
            for seed in (0, 1):
                scores.append(
                    {
                        "student": st,
                        "teacher": t,
                        "kind": "opd",
                        "seed": seed,
                        "math500": 0.6 + rnd.random() * 0.05,
                    }
                )
        for kind in ("coldstart", "sft", "rlvr"):
            scores.append(
                {
                    "student": st,
                    "teacher": None,
                    "kind": kind,
                    "seed": 0,
                    "math500": 0.55 + rnd.random() * 0.05,
                }
            )
    io.write_json(res / "student_scores.json", scores)
    io.write_json(
        res / "rungs.json",
        {
            r: {"memory_gb": 60 / (2**i)}
            for i, r in enumerate(("T16", "T8", "T4-int", "T3", "T2", "D16"))
        },
    )
    return e1, res


def test_all_figures_render(tmp_path):
    e1, res = synthetic(tmp_path)
    made = make_all(tmp_path / "figs", res, e1)
    assert all(made.values()), made
    for p in made.values():
        assert p.exists() and p.stat().st_size > 1000 and p.with_suffix(".png").exists()


def test_missing_inputs_are_skipped(tmp_path):
    made = make_all(tmp_path / "figs", tmp_path / "nothing", tmp_path / "nothing_e1")
    assert made == {"fidelity": None, "bits": None, "entropy": None, "memory": None}


def test_spread_enforces_gap():
    from qopd.paper.figures import spread

    out = spread([1.0, 1.01, 5.0], 0.5)
    s = sorted(out)
    assert s[1] - s[0] >= 0.5 - 1e-9 and s[2] - s[1] >= 0.5 - 1e-9
    assert spread([0.0, 10.0], 1.0) == [0.0, 10.0]
