"""Chat prompts in Qwen3 thinking mode, built directly as token ids (PLAN 1.2).

Rules
- A prompt is produced once, as token ids, by the tokenizer's chat template with
  ``add_generation_prompt=True``. In thinking mode the prompt ends with
  ``<|im_start|>assistant\\n`` and the model writes ``<think>...</think>`` itself.
- The teacher scores exactly ``rollout.prompt_ids + rollout.response_ids``: the ids the
  student was prompted with and the ids it generated (including the final ``<|im_end|>``
  when it stopped on EOS). Nothing is decoded and re-encoded in between; re-tokenizing
  can merge or split tokens and silently shift every per-token reward.
- Teacher and student share the tokenizer (checked by ``qopd.models.tokenizer_check``),
  so the teacher's prompt ids are the student's prompt ids.

Instruction wording follows the Qwen3 model card's recommendation for math and
multiple-choice evaluation (recorded as a decision in PROGRESS.md).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from qopd.types import Problem, ProblemKind, Rollout

MATH_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
MC_INSTRUCTION = (
    'Please show your choice in the answer field with only the choice letter, e.g., "answer": "C".'
)
ASSISTANT_PREFIX = "<|im_start|>assistant\n"


class TokenDriftError(AssertionError):
    """The teacher-scoring input is not exactly prompt_ids + response_ids."""


def instruction_for(problem: Problem) -> str:
    return MC_INSTRUCTION if problem.kind == ProblemKind.MULTIPLE_CHOICE else MATH_INSTRUCTION


def format_messages(problem: Problem, instruction: str | None = None) -> list[dict[str, str]]:
    """Single-turn user message for ``problem`` with the instruction appended."""
    text = instruction if instruction is not None else instruction_for(problem)
    return [{"role": "user", "content": f"{problem.problem}\n\n{text}"}]


def _as_ids(out: Any) -> list[int]:
    """``apply_chat_template(tokenize=True)`` returns a list (transformers 4) or a
    BatchEncoding (transformers 5); normalise to a plain list of ints."""
    ids = out["input_ids"] if hasattr(out, "keys") else out
    if ids and isinstance(ids[0], list):  # batched form
        if len(ids) != 1:
            raise ValueError("expected a single conversation")
        ids = ids[0]
    return [int(i) for i in ids]


def prompt_ids(
    tokenizer: Any,
    problem: Problem,
    enable_thinking: bool = True,
    instruction: str | None = None,
) -> list[int]:
    """Chat-template token ids for ``problem``, ending at the assistant turn."""
    ids = _as_ids(
        tokenizer.apply_chat_template(
            format_messages(problem, instruction),
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    )
    if not ids:
        raise ValueError(f"empty prompt for problem {problem.id}")
    return ids


def prompt_text(
    tokenizer: Any,
    problem: Problem,
    enable_thinking: bool = True,
    instruction: str | None = None,
) -> str:
    """The same prompt as text, for logging and inspection only (never for scoring)."""
    return tokenizer.apply_chat_template(
        format_messages(problem, instruction),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def batch_prompt_ids(
    tokenizer: Any, problems: Sequence[Problem], enable_thinking: bool = True
) -> list[list[int]]:
    return [prompt_ids(tokenizer, p, enable_thinking) for p in problems]


def teacher_input_ids(rollout: Rollout) -> list[int]:
    """The exact teacher-scoring input: ``prompt_ids + response_ids`` (no re-tokenization)."""
    return rollout.input_ids


def check_teacher_input(rollout: Rollout, scored_ids: Sequence[int]) -> None:
    """Raise :class:`TokenDriftError` unless ``scored_ids`` is exactly the rollout's ids.

    Backends call this on whatever sequence they actually sent to the teacher.
    """
    expected = rollout.input_ids
    got = [int(i) for i in scored_ids]
    if got != expected:
        n = min(len(got), len(expected))
        first = next((i for i in range(n) if got[i] != expected[i]), n)
        raise TokenDriftError(
            f"teacher input differs from prompt_ids + response_ids at position {first} "
            f"(len {len(got)} vs {len(expected)})"
        )


def retokenization_drift(tokenizer: Any, response_ids: Sequence[int]) -> bool:
    """Diagnostic only: True if decode -> encode would change ``response_ids``.

    Shows why scoring must use the generated ids: a sampled sequence is not always the
    canonical tokenization of its own text.
    """
    text = tokenizer.decode(list(response_ids), skip_special_tokens=False)
    return tokenizer.encode(text, add_special_tokens=False) != list(response_ids)
