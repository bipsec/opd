"""HF transformers teacher backend: plain forward pass, the slow reference (PLAN 1.3).

Used to validate the vLLM backend: mean |Δ logprob| < 0.02 nats/token on 20 sequences.
"""

from __future__ import annotations

from collections.abc import Sequence

from qopd.teacher.base import TeacherScorer
from qopd.types import Rollout, TeacherScores


class HFTeacher(TeacherScorer):
    backend = "hf"

    def __init__(self, model_path: str, batch_size: int = 1) -> None:
        super().__init__(model_path)
        self.batch_size = batch_size

    def start(self) -> None:
        raise NotImplementedError("step 1.3")

    def stop(self) -> None:
        raise NotImplementedError("step 1.3")

    def score(self, rollouts: Sequence[Rollout], top_k: int) -> list[TeacherScores]:
        raise NotImplementedError("step 1.3")
