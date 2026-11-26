"""Student rollout engine (PLAN 1.4) and weight sync (PLAN 1.4 / 6.3).

``RolloutEngine.generate`` returns, per sample, the exact prompt and response token ids,
the sampled-token log-probs, the finish reason and lengths. Every sampling parameter is
passed explicitly (the model's generation_config.json defaults, e.g. Qwen3's top_k=20,
are never inherited), and the stop tokens come from the model's own generation config.

Weight sync starts with ``ReloadFromCheckpoint`` (save safetensors, restart the engine
from that directory); an in-place strategy (vLLM ``update_weights``) is timed in 6.3.
"""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from qopd.types import FinishReason, Rollout
from qopd.utils.logging import get_logger
from qopd.utils.memory import PeakMemory

if TYPE_CHECKING:
    from qopd.config import SamplingCfg

log = get_logger("rollout")


@dataclass
class GenerationStats:
    n_sequences: int
    n_tokens: int  # response tokens generated
    seconds: float
    peak_mem_gb: float  # system memory in use (unified), peak during generation
    min_available_gb: float = float("nan")
    finish_reasons: dict[str, int] = field(default_factory=dict)

    @property
    def tokens_per_s(self) -> float:
        return self.n_tokens / self.seconds if self.seconds > 0 else 0.0


def _finish_reason(reason: str | None) -> FinishReason:
    if reason == "length":
        return FinishReason.LENGTH
    if reason in ("stop", None):
        return FinishReason.STOP
    return FinishReason.ABORT


def sampling_params(
    sampling: SamplingCfg,
    n: int,
    stop_token_ids: list[int],
    seed: int | None,
    max_tokens: int | None = None,
) -> Any:
    """vLLM ``SamplingParams`` with every field that affects the distribution set."""
    from vllm import SamplingParams  # noqa: PLC0415

    return SamplingParams(
        n=n,
        temperature=sampling.temperature,
        top_p=sampling.top_p,
        top_k=sampling.top_k,
        min_p=0.0,
        repetition_penalty=1.0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        max_tokens=max_tokens if max_tokens is not None else sampling.max_tokens,
        stop_token_ids=stop_token_ids,
        seed=seed,
        logprobs=0,  # log-prob of the sampled token
        detokenize=False,  # ids only; text is never used for scoring
    )


class RolloutEngine:
    """vLLM-backed generator for one student model."""

    def __init__(
        self,
        model_path: str,
        revision: str | None = None,
        kv_cache_gb: float = 6.0,
        gpu_memory_utilization: float = 0.12,
        max_model_len: int = 5120,
        seed: int = 0,
        enforce_eager: bool = False,
        enable_prefix_caching: bool = False,
    ) -> None:
        self.model_path = model_path
        self.revision = revision
        self.kv_cache_gb = kv_cache_gb
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.seed = seed
        self.enforce_eager = enforce_eager
        self.enable_prefix_caching = enable_prefix_caching  # eval: repeated prompts
        self.stop_ids: list[int] = []
        self.load_seconds = float("nan")
        self._llm: Any = None

    @property
    def running(self) -> bool:
        return self._llm is not None

    def start(self) -> None:
        """Start the vLLM engine (memory logged before and after)."""
        if self._llm is not None:
            return
        from qopd.models.loading import build_llm, stop_token_ids  # noqa: PLC0415

        t0 = time.perf_counter()
        self.stop_ids = stop_token_ids(self.model_path, self.revision)
        self._llm = build_llm(
            self.model_path,
            revision=self.revision,
            kv_cache_gb=self.kv_cache_gb,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            seed=self.seed,
            enforce_eager=self.enforce_eager,
            enable_prefix_caching=self.enable_prefix_caching,
        )
        self.load_seconds = time.perf_counter() - t0

    def stop(self) -> None:
        """Shut down the engine and free its memory."""
        if self._llm is None:
            return
        from qopd.models.loading import shutdown_llm  # noqa: PLC0415

        llm, self._llm = self._llm, None
        shutdown_llm(llm, f"rollout engine {self.model_path}")

    def restart(self, model_path: str, revision: str | None = None) -> float:
        """Point the engine at new weights; restart it only if it was running.

        (Phase-alternating OPD keeps the engine stopped between rounds; the next ``start``
        then loads the new weights.) Returns seconds taken.
        """
        t0 = time.perf_counter()
        was_running = self.running
        self.stop()
        self.model_path, self.revision = model_path, revision
        if was_running:
            self.start()
        return time.perf_counter() - t0

    def generate(
        self,
        prompt_ids: list[list[int]],
        sampling: SamplingCfg,
        n: int = 1,
        problem_ids: list[str] | None = None,
        seed: int | None = None,
        seeds: list[int] | None = None,
        max_tokens: list[int] | None = None,
    ) -> tuple[list[Rollout], GenerationStats]:
        """Sample ``n`` responses per prompt.

        Request i uses ``seeds[i]`` if given, else ``seed + i`` (or ``sampling.seed + i``).
        """
        if self._llm is None:
            raise RuntimeError("call start() first")
        from vllm.inputs import TokensPrompt  # noqa: PLC0415

        if problem_ids is not None and len(problem_ids) != len(prompt_ids):
            raise ValueError("problem_ids and prompt_ids differ in length")
        if max_tokens is not None and len(max_tokens) != len(prompt_ids):
            raise ValueError("max_tokens and prompt_ids differ in length")
        budgets = max_tokens or [sampling.max_tokens] * len(prompt_ids)  # per-request limits
        longest = max(len(p) + b for p, b in zip(prompt_ids, budgets, strict=True))
        if longest > self.max_model_len:
            raise ValueError(
                f"prompt + max_tokens = {longest} > max_model_len {self.max_model_len}"
            )
        if seeds is not None and len(seeds) != len(prompt_ids):
            raise ValueError("seeds and prompt_ids differ in length")
        base_seed = seed if seed is not None else sampling.seed
        per_request = seeds or [
            None if base_seed is None else base_seed + i for i in range(len(prompt_ids))
        ]
        params = [
            sampling_params(sampling, n, self.stop_ids, s, b)
            for s, b in zip(per_request, budgets, strict=True)
        ]
        prompts = [TokensPrompt(prompt_token_ids=list(p)) for p in prompt_ids]
        t0 = time.perf_counter()
        with PeakMemory() as pm:
            outputs = self._llm.generate(prompts, params, use_tqdm=False)
        seconds = time.perf_counter() - t0

        rollouts: list[Rollout] = []
        reasons: dict[str, int] = {}
        for i, (p, out) in enumerate(zip(prompt_ids, outputs, strict=True)):
            if list(out.prompt_token_ids) != list(p):
                raise RuntimeError(f"engine changed prompt {i} token ids")
            for k, c in enumerate(out.outputs):
                ids = [int(t) for t in c.token_ids]
                lps = [float(d[t].logprob) for d, t in zip(c.logprobs, ids, strict=True)]
                reason = _finish_reason(c.finish_reason)
                reasons[reason.value] = reasons.get(reason.value, 0) + 1
                pid = problem_ids[i] if problem_ids is not None else f"p{i}"
                rollouts.append(Rollout(list(p), ids, lps, reason, pid, k))
        stats = GenerationStats(
            n_sequences=len(rollouts),
            n_tokens=sum(len(r.response_ids) for r in rollouts),
            seconds=seconds,
            peak_mem_gb=pm.peak_used_gb,
            min_available_gb=pm.min_available_gb,
            finish_reasons=reasons,
        )
        log.info(
            "generated %d seqs, %d tokens in %.1fs (%.0f tok/s), peak used %.1f GB",
            stats.n_sequences,
            stats.n_tokens,
            stats.seconds,
            stats.tokens_per_s,
            stats.peak_mem_gb,
        )
        return rollouts, stats


class WeightSync(Protocol):
    """Moves trained student weights into the running inference engine."""

    name: str

    def sync(self, engine: RolloutEngine, model: Any, step: int) -> float: ...


class ReloadFromCheckpoint:
    """Save safetensors (+ tokenizer) to ``ckpt_dir/sync_<step>``, restart the engine there.

    Keeps the last ``keep_last`` sync directories; returns the seconds the sync took.
    """

    name = "reload_from_checkpoint"

    def __init__(self, ckpt_dir: Path, tokenizer: Any, keep_last: int = 1) -> None:
        self.ckpt_dir = Path(ckpt_dir)
        self.tokenizer = tokenizer
        self.keep_last = keep_last
        self.last_save_s = float("nan")
        self.last_restart_s = float("nan")

    def sync(self, engine: RolloutEngine, model: Any, step: int) -> float:
        t0 = time.perf_counter()
        out = self.ckpt_dir / f"sync_{step:06d}"
        tmp = out.with_name(out.name + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        from qopd.models.loading import save_hf_bf16  # noqa: PLC0415

        save_hf_bf16(model, self.tokenizer, tmp)  # vLLM serves BF16; trainer keeps FP32
        shutil.rmtree(out, ignore_errors=True)
        tmp.rename(out)  # a crash mid-save never leaves a half-written sync dir
        self.last_save_s = time.perf_counter() - t0
        self.last_restart_s = engine.restart(str(out))
        old = sorted(self.ckpt_dir.glob("sync_[0-9]*"))
        for d in old[: max(0, len(old) - self.keep_last)]:
            shutil.rmtree(d, ignore_errors=True)
        total = time.perf_counter() - t0
        log.info(
            "weight sync step %d: save %.1fs, restart %.1fs",
            step,
            self.last_save_s,
            self.last_restart_s,
        )
        return total
