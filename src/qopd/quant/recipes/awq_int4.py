"""INT4 AWQ W4A16, group 128, via llm-compressor (rung T4-int / D4) (PLAN 4.2).

AutoAWQ is archived; its AWQ support moved into llm-compressor.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from qopd.quant.recipes import RecipeResult


class AWQInt4Recipe:
    format = "awq_w4a16_g128"
    tool = "llmcompressor"

    def build(self, src: str, calibration: Path, out_dir: Path, **kwargs: Any) -> RecipeResult:
        raise NotImplementedError("step 4.2")
