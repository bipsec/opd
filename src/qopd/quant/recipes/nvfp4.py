"""NVFP4 via ModelOpt or llm-compressor (rung T4-fp) (PLAN 4.2)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from qopd.quant.recipes import RecipeResult


class NVFP4Recipe:
    format = "nvfp4"
    tool = "llmcompressor|modelopt"

    def build(self, src: str, calibration: Path, out_dir: Path, **kwargs: Any) -> RecipeResult:
        raise NotImplementedError("step 4.2")
