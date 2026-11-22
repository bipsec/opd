"""INT4 AWQ W4A16, group 128, via llm-compressor (rungs T4-int, D4).

AutoAWQ is archived; its AWQ support moved into llm-compressor.
"""

from __future__ import annotations

from typing import Any

from qopd.quant.recipes import LLMCompressorRecipe


class AWQInt4Recipe(LLMCompressorRecipe):
    format = "awq_w4a16_g128"
    scheme = "W4A16"  # 4-bit symmetric weights, group size 128, 16-bit activations

    def modifiers(self, ignore: list[str]) -> list[Any]:
        from llmcompressor.modifiers.awq import AWQModifier  # noqa: PLC0415

        return [AWQModifier(targets=["Linear"], scheme=self.scheme, ignore=ignore)]
