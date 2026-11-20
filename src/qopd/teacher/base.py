"""TeacherScorer interface (PLAN 1.3).

``score(rollouts, top_k)`` returns, for each rollout, the teacher log-prob of every
realized response token plus the teacher's top-k ids and log-probs at each position.
Scoring input is always ``rollout.input_ids`` (exact ids; never re-tokenized).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from qopd.types import Rollout, TeacherScores


class TeacherScorer(ABC):
    """Common interface for every teacher backend."""

    backend: str = "base"

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path

    @abstractmethod
    def start(self) -> None:
        """Load the teacher (memory logged before and after)."""

    @abstractmethod
    def stop(self) -> None:
        """Release the teacher's memory."""

    @abstractmethod
    def score(self, rollouts: Sequence[Rollout], top_k: int) -> list[TeacherScores]:
        """Score the response tokens of each rollout."""

    def __enter__(self) -> TeacherScorer:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()


def check_scores(rollouts: Sequence[Rollout], scores: Sequence[TeacherScores], top_k: int) -> None:
    """Shape check shared by all backends: one score per response token, k per position."""
    if len(rollouts) != len(scores):
        raise ValueError(f"{len(rollouts)} rollouts but {len(scores)} scores")
    for i, (r, s) in enumerate(zip(rollouts, scores, strict=True)):
        if len(s.token_logprobs) != len(r.response_ids):
            raise ValueError(
                f"rollout {i}: {len(r.response_ids)} response tokens, "
                f"{len(s.token_logprobs)} teacher log-probs"
            )
        if top_k and any(len(row) != top_k for row in s.topk_ids):
            raise ValueError(f"rollout {i}: top-k rows are not all length {top_k}")
