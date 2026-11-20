"""Seed every RNG we use (python, numpy, torch if available)."""

from __future__ import annotations

import os
import random


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed python, numpy and torch. ``deterministic`` also asks torch for deterministic kernels."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np  # noqa: PLC0415

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
