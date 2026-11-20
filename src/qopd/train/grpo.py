"""RLVR baseline with GRPO, verifier reward only, wall-clock matched (PLAN 8.2).

Group size 4; reuses ``qopd.rollout.engine`` for sampling.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from qopd.config import ExperimentCfg
    from qopd.utils.run import RunDir


def group_advantages(rewards: torch.Tensor, group_size: int = 4) -> torch.Tensor:
    """(r − mean_group) / (std_group + eps) for rewards laid out as [n_groups * group_size]."""
    raise NotImplementedError("step 8.2")


def run_grpo(cfg: ExperimentCfg, run: RunDir) -> None:
    raise NotImplementedError("step 8.2")
