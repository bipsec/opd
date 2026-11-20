"""OPD round loop (PLAN 6.2): sample → score → one optimizer step → sync weights.

Modes:
- ``co_resident``: student engine, trainer and teacher live in memory together.
- ``phase_alternating``: load/unload the teacher and student engine each round.
- (two-Spark, once the peer is free) teacher scoring on the peer node.

Per-round metrics: reward mean/std/percentiles, student entropy, response length,
truncation rate, repetition rate (ends in a repeated 8-gram), time per stage, peak
memory, power. Checkpoint every N rounds; resume from ``state.json``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qopd.config import ExperimentCfg
    from qopd.utils.run import RunDir


class LoopMode(StrEnum):
    CO_RESIDENT = "co_resident"
    PHASE_ALTERNATING = "phase_alternating"


@dataclass
class RoundMetrics:
    round: int
    reward_mean: float
    reward_std: float
    reward_pcts: dict[str, float] = field(default_factory=dict)
    entropy: float = float("nan")
    response_len_mean: float = float("nan")
    truncation_rate: float = float("nan")
    repetition_rate: float = float("nan")
    seconds: dict[str, float] = field(default_factory=dict)  # stage -> seconds
    peak_mem_gb: float = float("nan")
    power_w_mean: float = float("nan")


def repetition_rate(responses: list[list[int]], n: int = 8) -> float:
    """Share of responses "ending in a repeated n-gram" (PLAN 6.2).

    Exact definition (last n-gram repeated immediately vs anywhere earlier) is settled in
    step 6.2 and recorded in PROGRESS.md.
    """
    raise NotImplementedError("step 6.2")


def run_opd(cfg: ExperimentCfg, run: RunDir) -> None:
    """Run (or resume) the OPD loop described by ``cfg`` inside ``run``."""
    raise NotImplementedError("step 6.2")
