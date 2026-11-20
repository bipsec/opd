"""3-bit GPTQ, group 128, via GPTQModel (rung T3) (PLAN 4.2)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from qopd.quant.recipes import RecipeResult


class GPTQ3Recipe:
    format = "gptq_w3a16_g128"
    tool = "gptqmodel"

    def build(self, src: str, calibration: Path, out_dir: Path, **kwargs: Any) -> RecipeResult:
        raise NotImplementedError("step 4.2")
