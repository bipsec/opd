"""Shared data types passed between modules.

Kept dependency-free (plain lists / tuples) so that data, rollout, teacher, train and
fidelity code can exchange them without importing torch. Tensor conversion happens at
the edges (``train/``).

Token handling rule (PLAN 1.2): a rollout is always ``prompt_ids + response_ids`` exactly
as generated. Text is never re-tokenized before scoring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class ProblemKind(StrEnum):
    MATH = "math"  # free-form numeric / closed-form answer (math-verify)
    MULTIPLE_CHOICE = "mc"  # letter answer (GPQA-Diamond)


@dataclass(frozen=True)
class Problem:
    """Unified data schema (PLAN 2.1): ``{id, problem, answer, source, kind}``."""

    id: str
    problem: str
    answer: str
    source: str
    kind: ProblemKind = ProblemKind.MATH
    meta: dict[str, Any] = field(default_factory=dict)


class FinishReason(StrEnum):
    STOP = "stop"  # EOS / stop token
    LENGTH = "length"  # hit max_tokens (truncated)
    ABORT = "abort"


@dataclass
class Rollout:
    """One student sample with the exact token ids it produced."""

    prompt_ids: list[int]
    response_ids: list[int]
    sampled_logprobs: list[float]  # student log p of each response token at sampling time
    finish_reason: FinishReason = FinishReason.STOP
    problem_id: str = ""
    sample_idx: int = 0

    def __post_init__(self) -> None:
        if len(self.sampled_logprobs) not in (0, len(self.response_ids)):
            raise ValueError(
                f"sampled_logprobs has {len(self.sampled_logprobs)} entries, "
                f"response_ids has {len(self.response_ids)}"
            )

    @property
    def input_ids(self) -> list[int]:
        """The exact sequence a teacher scores: ``prompt_ids + response_ids``."""
        return self.prompt_ids + self.response_ids

    @property
    def response_slice(self) -> slice:
        """Positions in ``input_ids`` whose tokens are response tokens."""
        return slice(len(self.prompt_ids), len(self.prompt_ids) + len(self.response_ids))

    @property
    def truncated(self) -> bool:
        return self.finish_reason == FinishReason.LENGTH


@dataclass
class TeacherScores:
    """Teacher log-probs on a rollout's response tokens (one entry per response token)."""

    token_logprobs: list[float]  # log p_teacher(realized token)
    topk_ids: list[list[int]] = field(default_factory=list)  # [T][k]
    topk_logprobs: list[list[float]] = field(default_factory=list)  # [T][k]

    def __len__(self) -> int:
        return len(self.token_logprobs)
