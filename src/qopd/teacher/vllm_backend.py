"""vLLM teacher backend (PLAN 1.3).

Each rollout's ``prompt_ids + response_ids`` is sent as a token prompt with
``prompt_logprobs=top_k`` and ``max_tokens=1``; the response positions are sliced out.
vLLM returns, for every prompt position j >= 1, the log-prob of the actual token
``input_ids[j]`` plus the top-k alternatives. ``logprobs_mode="raw_logprobs"`` means the
values come from the raw logits (no temperature / top-k / top-p processing).

Memory (GB10 unified memory): the KV cache is sized in bytes
(``kv_cache_memory_bytes``) and ``gpu_memory_utilization`` is a small explicit share, so
the engine stays small next to other processes (vLLM's startup check needs that share to
fit in free memory). Prefix caching is off so every position's logits are computed.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from qopd.prompts.templates import check_teacher_input
from qopd.teacher.base import TeacherScorer, check_scores
from qopd.types import Rollout, TeacherScores
from qopd.utils.logging import get_logger

log = get_logger("teacher.vllm")


def _position_items(pos: Any) -> list[tuple[int, Any]]:
    """(token_id, Logprob) pairs of one prompt position (dict or flat-logprobs view)."""
    if pos is None:
        return []
    return list(pos.items())


def scores_from_prompt_logprobs(
    prompt_logprobs: Sequence[Any], input_ids: Sequence[int], n_prompt: int, top_k: int
) -> TeacherScores:
    """Turn vLLM ``prompt_logprobs`` for one sequence into :class:`TeacherScores`."""
    token_lp: list[float] = []
    topk_ids: list[list[int]] = []
    topk_lp: list[list[float]] = []
    for j in range(n_prompt, len(input_ids)):
        items = _position_items(prompt_logprobs[j])
        by_id = dict(items)
        actual = int(input_ids[j])
        if actual not in by_id:
            raise RuntimeError(f"vLLM returned no log-prob for the realized token at {j}")
        token_lp.append(float(by_id[actual].logprob))
        if top_k:
            ranked = sorted(
                ((lp.rank, tid, lp.logprob) for tid, lp in items if lp.rank is not None),
                key=lambda x: x[0],
            )[:top_k]
            topk_ids.append([int(t) for _, t, _ in ranked])
            topk_lp.append([float(v) for _, _, v in ranked])
    return TeacherScores(token_lp, topk_ids, topk_lp)


class VLLMTeacher(TeacherScorer):
    backend = "vllm"

    def __init__(
        self,
        model_path: str,
        revision: str | None = None,
        kv_cache_gb: float = 4.0,
        gpu_memory_utilization: float = 0.12,
        max_model_len: int | None = None,
        max_logprobs: int = 20,
        dtype: str = "bfloat16",
        enforce_eager: bool = False,
        seed: int = 0,
        quantization: str | None = None,
    ) -> None:
        super().__init__(model_path)
        self.revision = revision
        self.kv_cache_gb = kv_cache_gb
        # vLLM's startup check compares this share of total (unified) memory with free
        # memory even when the KV cache is sized in bytes; keep it small and explicit.
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.max_logprobs = max_logprobs
        self.dtype = dtype
        self.enforce_eager = enforce_eager
        self.seed = seed
        self.quantization = quantization
        self.llm: Any = None

    def start(self) -> None:
        if self.llm is not None:
            return
        from qopd.models.loading import build_llm  # noqa: PLC0415

        self.llm = build_llm(
            self.model_path,
            revision=self.revision,
            kv_cache_gb=self.kv_cache_gb,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            max_logprobs=self.max_logprobs,
            dtype=self.dtype,
            seed=self.seed,
            enforce_eager=self.enforce_eager,
            quantization=self.quantization,
        )

    def stop(self) -> None:
        if self.llm is None:
            return
        from qopd.models.loading import shutdown_llm  # noqa: PLC0415

        llm, self.llm = self.llm, None
        shutdown_llm(llm, f"vLLM teacher {self.model_path}")

    def score(self, rollouts: Sequence[Rollout], top_k: int) -> list[TeacherScores]:
        if self.llm is None:
            raise RuntimeError("call start() first")
        if top_k > self.max_logprobs:
            raise ValueError(f"top_k={top_k} exceeds max_logprobs={self.max_logprobs}")
        from vllm import SamplingParams  # noqa: PLC0415
        from vllm.inputs import TokensPrompt  # noqa: PLC0415

        params = SamplingParams(
            max_tokens=1, prompt_logprobs=max(top_k, 1), temperature=1.0, detokenize=False
        )
        prompts = [TokensPrompt(prompt_token_ids=r.input_ids) for r in rollouts]
        outputs = self.llm.generate(prompts, params, use_tqdm=False)
        scores = []
        for r, out in zip(rollouts, outputs, strict=True):
            check_teacher_input(r, out.prompt_token_ids)
            scores.append(
                scores_from_prompt_logprobs(
                    out.prompt_logprobs, out.prompt_token_ids, len(r.prompt_ids), top_k
                )
            )
        check_scores(rollouts, scores, top_k)
        return scores
