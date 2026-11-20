"""Expand an experiment YAML (e.g. configs/experiments/e2.yaml) into run specs (PLAN 8.1).

students × rungs × seeds → one :class:`RunSpec` each, with a deterministic run id.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunSpec:
    experiment: str
    run_id: str  # e.g. "S1__T4-int__s0"
    student: str
    teacher: str
    seed: int
    overrides: dict[str, Any] = field(default_factory=dict)
    node: str = "local"  # which Spark runs it (configs/cluster.yaml)


def run_id(student: str, teacher: str, seed: int) -> str:
    return f"{student}__{teacher}__s{seed}"


def expand(experiment_yaml: Path) -> list[RunSpec]:
    raise NotImplementedError("step 8.1")
