"""NVFP4 (rung T4-fp) via llm-compressor (the doc allows ModelOpt or llm-compressor; llm-compressor
chosen by the user, PROGRESS.md 2026-11-21: same toolchain and checkpoint format as T8/T4-int)."""

from __future__ import annotations

from typing import Any

from qopd.quant.recipes import LLMCompressorRecipe


class NVFP4Recipe(LLMCompressorRecipe):
    format = "nvfp4"
    scheme = "NVFP4"

    def modifiers(self, ignore: list[str]) -> list[Any]:
        from llmcompressor.modifiers.quantization import QuantizationModifier  # noqa: PLC0415

        return [QuantizationModifier(targets="Linear", scheme=self.scheme, ignore=ignore)]
