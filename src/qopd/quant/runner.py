"""Build one ladder rung and write ``recipe.json`` (PLAN 4.2); validate a quantized checkpoint.

One rung per approved job: quantizing the 30B teacher can take hours (CLAUDE.md rule 4).
Quantization runs in the ``qopd-quant`` image; loading/scoring the result runs in ``qopd-dev``
(its vLLM keeps its own compressed-tensors), so :func:`validate` also proves the checkpoint
format is readable by the serving stack.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from qopd.config import CONFIG_ROOT, PROJECT_ROOT, ModelCfg, load
from qopd.quant.recipes import Recipe, RecipeResult
from qopd.quant.recipes.awq_int4 import AWQInt4Recipe
from qopd.quant.recipes.fp8 import FP8Recipe
from qopd.quant.recipes.gguf import GGUFQ2KRecipe
from qopd.quant.recipes.gptq3 import GPTQ3Recipe
from qopd.quant.recipes.nvfp4 import NVFP4Recipe
from qopd.utils import io
from qopd.utils.logging import get_logger

log = get_logger("quant")
CALIBRATION = PROJECT_ROOT / "data" / "calibration" / "t16_traces.jsonl"

RECIPES: dict[str, type[Any]] = {
    "fp8_w8a8": FP8Recipe,
    "nvfp4": NVFP4Recipe,
    "awq_w4a16_g128": AWQInt4Recipe,
    "gptq_w3a16_g128": GPTQ3Recipe,
    "gguf_q2_k": GGUFQ2KRecipe,
}


def write_recipe_json(result: RecipeResult) -> Path:
    path = result.out_dir / "recipe.json"
    io.write_json(path, result.to_json())
    return path


def build_model(
    base_id: str,
    fmt: str,
    out_dir: Path,
    calibration: Path = CALIBRATION,
    rung: str = "",
    **kwargs: Any,
) -> RecipeResult:
    """Quantize registry model ``base_id`` with format ``fmt`` into ``out_dir``."""
    if (out_dir / "recipe.json").exists():
        log.info("%s exists; skipping", out_dir)
        d = io.read_json(out_dir / "recipe.json")
        return RecipeResult(**{**d, "out_dir": Path(d["out_dir"])})
    if not calibration.exists():
        raise FileNotFoundError(f"{calibration} missing: run `qopd quant calib-traces` (4.1)")
    from qopd.models.registry import resolve  # noqa: PLC0415

    base = resolve(base_id)
    recipe: Recipe = RECIPES[fmt]()
    res = recipe.build(
        base.path, calibration, out_dir, revision=base.revision, rung=rung or fmt, **kwargs
    )
    write_recipe_json(res)
    return res


def build_rung(rung: str, calibration: Path = CALIBRATION, **kwargs: Any) -> RecipeResult:
    """Build registry rung ``rung`` (e.g. ``T8``) into its configured local path."""
    cfg = load(CONFIG_ROOT / "models" / f"{rung}.yaml", cls=ModelCfg)
    if cfg.quant_format is None or cfg.base is None:
        raise ValueError(f"{rung} is not a quantized rung")
    return build_model(cfg.base, cfg.quant_format, Path(cfg.path), calibration, rung=rung, **kwargs)


def validate(
    base_id: str,
    quantized: Path,
    n: int = 20,
    top_k: int = 20,
    kv_cache_gb: float = 4.0,
    gpu_memory_utilization: float = 0.2,
) -> dict[str, Any]:
    """Score the same exact-id rollouts with the BF16 model and the quantized checkpoint in vLLM;
    write and return the comparison (``<quantized>/validation.json``)."""
    from qopd.models.registry import resolve  # noqa: PLC0415
    from qopd.teacher.agreement import agreement_metrics  # noqa: PLC0415
    from qopd.teacher.vllm_backend import VLLMTeacher  # noqa: PLC0415
    from qopd.types import Rollout  # noqa: PLC0415

    pool = PROJECT_ROOT / "runs" / "agreement" / f"{base_id}_vllm_vs_hf_s0" / "rollouts.jsonl"
    if not pool.exists():
        raise FileNotFoundError(f"{pool} missing (step 1.3 agreement run)")
    rollouts = [Rollout.from_dict(d) for d in io.iter_jsonl(pool)][:n]
    max_len = max(len(r.input_ids) for r in rollouts) + 16
    base = resolve(base_id)
    scores = {}
    for tag, path, rev in (("bf16", base.path, base.revision), ("quant", str(quantized), None)):
        t = VLLMTeacher(
            path,
            rev,
            kv_cache_gb=kv_cache_gb,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_len,
            max_logprobs=max(20, top_k),
        )
        t.start()
        try:
            scores[tag] = t.score(rollouts, top_k)
        finally:
            t.stop()
    out = {
        "base": base_id,
        "quantized": str(quantized),
        "n_sequences": len(rollouts),
        **agreement_metrics(scores["bf16"], scores["quant"]),
    }
    io.write_json(quantized / "validation.json", out)
    return out
