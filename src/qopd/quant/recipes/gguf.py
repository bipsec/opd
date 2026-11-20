"""2-bit GGUF Q2_K via llama.cpp, optional (rung T2) (PLAN 4.2)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from qopd.quant.recipes import RecipeResult


class GGUFQ2KRecipe:
    format = "gguf_q2_k"
    tool = "llama.cpp"

    def build(self, src: str, calibration: Path, out_dir: Path, **kwargs: Any) -> RecipeResult:
        raise NotImplementedError("step 4.2")
