"""Build one ladder rung and write ``recipe.json`` (PLAN 4.2).

One rung per approved job: quantizing the 30B teacher can take hours (CLAUDE.md rule 4).
"""

from __future__ import annotations

from pathlib import Path

from qopd.quant.recipes import Recipe, RecipeResult
from qopd.quant.recipes.awq_int4 import AWQInt4Recipe
from qopd.quant.recipes.fp8 import FP8Recipe
from qopd.quant.recipes.gguf import GGUFQ2KRecipe
from qopd.quant.recipes.gptq3 import GPTQ3Recipe
from qopd.quant.recipes.nvfp4 import NVFP4Recipe

RECIPES: dict[str, type[Recipe]] = {
    "fp8_w8a8": FP8Recipe,
    "nvfp4": NVFP4Recipe,
    "awq_w4a16_g128": AWQInt4Recipe,
    "gptq_w3a16_g128": GPTQ3Recipe,
    "gguf_q2_k": GGUFQ2KRecipe,
}


def build_rung(rung: str, calibration: Path, out_root: Path | None = None) -> RecipeResult:
    """Look up ``rung`` in the registry, run its recipe, write ``recipe.json``."""
    raise NotImplementedError("step 4.2")


def write_recipe_json(result: RecipeResult) -> Path:
    raise NotImplementedError("step 4.2")
