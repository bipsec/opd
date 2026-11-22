"""INT4 AWQ W4A16, group 128, via llm-compressor (rungs T4-int, D4).

AutoAWQ is archived; its AWQ support moved into llm-compressor.
"""

from __future__ import annotations

from typing import Any

from qopd.quant.recipes import LLMCompressorRecipe


class AWQInt4Recipe(LLMCompressorRecipe):
    format = "awq_w4a16_g128"
    scheme = "W4A16"  # 4-bit symmetric weights, group size 128, 16-bit activations
    # Routed-only calibration: each expert's AWQ scales come from the tokens the router sends it.
    # With all 128 experts calibrated, AWQ's cached activations grow ~128x and the 30B-A3B build
    # ran out of memory (PROGRESS.md, 2026-11-22; user decision). Other rungs calibrate all experts.
    calibrate_all_experts = False

    def modifiers(self, ignore: list[str]) -> list[Any]:
        from llmcompressor.modifiers.awq import AWQModifier  # noqa: PLC0415

        return [AWQModifier(targets=["Linear"], scheme=self.scheme, ignore=ignore)]
