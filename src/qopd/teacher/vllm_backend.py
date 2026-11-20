"""vLLM teacher backend (PLAN 1.3).

Feed ``prompt_ids + response_ids`` as the prompt with ``prompt_logprobs=top_k`` and
``max_tokens=1``, then slice out the response positions. Note: the resident vLLM server
is started with ``--max-logprobs 20``; our own engine sets ``max_logprobs >= top_k``.
"""

from __future__ import annotations

from collections.abc import Sequence

from qopd.teacher.base import TeacherScorer
from qopd.types import Rollout, TeacherScores


class VLLMTeacher(TeacherScorer):
    backend = "vllm"

    def __init__(
        self, model_path: str, gpu_memory_utilization: float, max_model_len: int | None = None
    ) -> None:
        super().__init__(model_path)
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len

    def start(self) -> None:
        raise NotImplementedError("step 1.3")

    def stop(self) -> None:
        raise NotImplementedError("step 1.3")

    def score(self, rollouts: Sequence[Rollout], top_k: int) -> list[TeacherScores]:
        raise NotImplementedError("step 1.3")
