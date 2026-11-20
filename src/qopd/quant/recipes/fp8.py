"""FP8 W8A8 via llm-compressor (rung T8 / D8) (PLAN 4.2)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from qopd.quant.recipes import RecipeResult


class FP8Recipe:
    format = "fp8_w8a8"
    tool = "llmcompressor"

    def build(self, src: str, calibration: Path, out_dir: Path, **kwargs: Any) -> RecipeResult:
        raise NotImplementedError("step 4.2")
