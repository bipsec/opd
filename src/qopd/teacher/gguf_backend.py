"""llama.cpp teacher backend for GGUF rungs (stub until Phase 4.3, only if needed)."""

from __future__ import annotations

from collections.abc import Sequence

from qopd.teacher.base import TeacherScorer
from qopd.types import Rollout, TeacherScores


class GGUFTeacher(TeacherScorer):
    backend = "gguf"

    def __init__(self, model_path: str, server_url: str = "http://127.0.0.1:8081") -> None:
        super().__init__(model_path)
        self.server_url = server_url

    def start(self) -> None:
        raise NotImplementedError("step 4.3")

    def stop(self) -> None:
        raise NotImplementedError("step 4.3")

    def score(self, rollouts: Sequence[Rollout], top_k: int) -> list[TeacherScores]:
        raise NotImplementedError("step 4.3")
