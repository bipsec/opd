"""FP8 W8A8 via llm-compressor, static activation scales from calibration (rungs T8, D8)."""

from __future__ import annotations

from typing import Any

from qopd.quant.recipes import LLMCompressorRecipe


class FP8Recipe(LLMCompressorRecipe):
    format = "fp8_w8a8"
    scheme = "FP8"  # weights and activations FP8, static per-tensor activation scales

    def modifiers(self, ignore: list[str]) -> list[Any]:
        from llmcompressor.modifiers.quantization import QuantizationModifier  # noqa: PLC0415

        return [QuantizationModifier(targets="Linear", scheme=self.scheme, ignore=ignore)]
