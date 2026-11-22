"""3-bit GPTQ, group 128 (rung T3), via llm-compressor's GPTQModifier (scheme W3A16).

The doc names GPTQModel, which cannot be installed next to the NGC torch/numpy/transformers in
any release; by the user's decision (PROGRESS.md, 2026-11-21) the same GPTQ algorithm is run with
llm-compressor. vLLM serving of the 3-bit compressed-tensors checkpoint is validated on S2 first.
"""

from __future__ import annotations

from typing import Any

from qopd.quant.recipes import LLMCompressorRecipe


class GPTQ3Recipe(LLMCompressorRecipe):
    format = "gptq_w3a16_g128"
    scheme = "W3A16"  # 3-bit symmetric weights, group 128, 16-bit activations

    def modifiers(self, ignore: list[str]) -> list[Any]:
        from llmcompressor.modifiers.quantization import GPTQModifier  # noqa: PLC0415

        return [GPTQModifier(targets="Linear", scheme=self.scheme, ignore=ignore)]
