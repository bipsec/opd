"""One module per quantization format; all implement :class:`Recipe` (PLAN 4.2).

Every recipe calibrates on the same texts - the 512 T16 calibration traces (prompt + trace,
at most 2,048 tokens, exact token ids; ``data/calibration/t16_traces.jsonl``) - and writes
``models/quant/<rung>/recipe.json`` with the tool, its version, the scheme, the modules left in
full precision and the sha256 of the calibration file.

Never quantized: ``lm_head`` and the mixture-of-experts router gates (``mlp.gate``), whose
outputs choose experts rather than producing features. For MoE teachers every expert sees the
calibration tokens (llm-compressor ``moe_calibrate_all_experts``), so rarely routed experts still
get calibrated scales; the routing itself is unchanged.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from qopd.utils import io

IGNORE_DEFAULT: tuple[str, ...] = ("lm_head", "re:.*mlp.gate$")


@dataclass
class RecipeResult:
    rung: str
    out_dir: Path
    format: str
    tool: str
    tool_version: str
    calibration_sha256: str
    seconds: float
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["out_dir"] = str(self.out_dir)
        return d


class Recipe(Protocol):
    """A quantization recipe for one format."""

    format: str
    tool: str

    def build(self, src: str, calibration: Path, out_dir: Path, **kwargs: Any) -> RecipeResult:
        """Quantize ``src`` (HF repo/path) with the calibration texts into ``out_dir``."""
        ...


def calibration_dataset(path: Path, max_len: int = 2048, n: int | None = None) -> Any:
    """``datasets.Dataset`` of exact input_ids/attention_mask from the calibration traces."""
    from datasets import Dataset  # noqa: PLC0415

    rows = []
    for r in io.iter_jsonl(path):
        ids = (r["prompt_ids"] + r["response_ids"])[:max_len]
        rows.append({"input_ids": ids, "attention_mask": [1] * len(ids)})
        if n is not None and len(rows) >= n:
            break
    return Dataset.from_list(rows)


class LLMCompressorRecipe:
    """Shared driver for llm-compressor ``oneshot`` recipes."""

    format = ""
    tool = "llmcompressor"
    scheme = ""
    # Every MoE expert sees all calibration tokens (default); a recipe can opt out (AWQ).
    calibrate_all_experts = True

    def _all_experts(self, kwargs: dict[str, Any]) -> bool:
        return bool(kwargs.get("moe_calibrate_all_experts", self.calibrate_all_experts))

    def modifiers(self, ignore: list[str]) -> list[Any]:
        raise NotImplementedError

    def build(self, src: str, calibration: Path, out_dir: Path, **kwargs: Any) -> RecipeResult:
        import llmcompressor  # noqa: PLC0415
        from llmcompressor import oneshot  # noqa: PLC0415
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

        from qopd.utils.memory import track_memory  # noqa: PLC0415

        revision = kwargs.get("revision")
        ignore = list(kwargs.get("ignore", IGNORE_DEFAULT))
        max_len = int(kwargs.get("max_seq_length", 2048))
        n = kwargs.get("num_calibration_samples")
        t0 = time.perf_counter()
        with track_memory(f"load {src} for {self.format}"):
            model = AutoModelForCausalLM.from_pretrained(src, revision=revision, dtype="auto")
        tok = AutoTokenizer.from_pretrained(src, revision=revision)
        ds = calibration_dataset(calibration, max_len, n)
        oneshot(
            model=model,
            moe_calibrate_all_experts=self._all_experts(kwargs),
            dataset=ds,
            recipe=self.modifiers(ignore),
            max_seq_length=max_len,
            num_calibration_samples=len(ds),
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(out_dir, save_compressed=True)
        tok.save_pretrained(out_dir)
        return RecipeResult(
            rung=kwargs.get("rung", ""),
            out_dir=out_dir,
            format=self.format,
            tool=self.tool,
            tool_version=llmcompressor.__version__,
            calibration_sha256=io.sha256_file(calibration),
            seconds=time.perf_counter() - t0,
            extra={
                "moe_calibrate_all_experts": self._all_experts(kwargs),
                "scheme": self.scheme,
                "ignore": ignore,
                "max_seq_length": max_len,
                "num_calibration_samples": len(ds),
                "source": src,
                "revision": revision,
            },
        )
