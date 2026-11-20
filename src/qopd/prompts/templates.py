"""Chat prompts in Qwen3 thinking mode, built directly as token ids (PLAN 1.2).

Rule: the teacher scores exactly ``prompt_ids + response_ids`` from the student's
rollout; nothing is decoded and re-encoded in between.
"""

from __future__ import annotations

from typing import Any

from qopd.types import Problem, Rollout

DEFAULT_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."


def format_messages(
    problem: Problem, instruction: str = DEFAULT_INSTRUCTION
) -> list[dict[str, str]]:
    """Single-turn user message for ``problem`` (instruction appended)."""
    return [{"role": "user", "content": f"{problem.problem}\n\n{instruction}"}]


def prompt_ids(tokenizer: Any, problem: Problem, enable_thinking: bool = True) -> list[int]:
    """Apply the chat template with ``add_generation_prompt=True`` and return token ids."""
    raise NotImplementedError("step 1.2")


def teacher_input_ids(rollout: Rollout) -> list[int]:
    """The exact teacher-scoring input: ``prompt_ids + response_ids`` (no re-tokenization)."""
    return rollout.input_ids
