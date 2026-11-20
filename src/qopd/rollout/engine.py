"""Student rollout engine (PLAN 1.4) and weight sync (PLAN 1.4 / 6.3).

``RolloutEngine.generate`` returns exact token ids, sampled-token log-probs, finish
reasons and lengths. Weight sync starts with ``ReloadFromCheckpoint`` (save safetensors,
restart the engine); an in-place strategy is evaluated in step 6.3.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from qopd.types import Rollout

if TYPE_CHECKING:
    from qopd.config import SamplingCfg


@dataclass
class GenerationStats:
    n_sequences: int
    n_tokens: int
    seconds: float
    peak_mem_gb: float

    @property
    def tokens_per_s(self) -> float:
        return self.n_tokens / self.seconds if self.seconds > 0 else 0.0


class WeightSync(Protocol):
    """Moves trained student weights into the running inference engine."""

    name: str

    def sync(self, engine: RolloutEngine, model: Any, step: int) -> None: ...


class ReloadFromCheckpoint:
    """Save safetensors to ``ckpt_dir``, then restart the vLLM engine from it."""

    name = "reload_from_checkpoint"

    def __init__(self, ckpt_dir: Path) -> None:
        self.ckpt_dir = ckpt_dir

    def sync(self, engine: RolloutEngine, model: Any, step: int) -> None:
        raise NotImplementedError("step 1.4")


class RolloutEngine:
    """vLLM-backed generator for one student model."""

    def __init__(self, model_path: str, gpu_memory_utilization: float, seed: int = 0) -> None:
        self.model_path = model_path
        self.gpu_memory_utilization = gpu_memory_utilization
        self.seed = seed
        self._llm: Any = None

    def start(self) -> None:
        """Start the vLLM engine (memory logged before and after)."""
        raise NotImplementedError("step 1.4")

    def stop(self) -> None:
        """Shut down the engine and free its memory."""
        raise NotImplementedError("step 1.4")

    def generate(
        self,
        prompt_ids: list[list[int]],
        sampling: SamplingCfg,
        n: int = 1,
        problem_ids: list[str] | None = None,
    ) -> tuple[list[Rollout], GenerationStats]:
        """Sample ``n`` responses per prompt; ``problem_ids`` tag each rollout."""
        raise NotImplementedError("step 1.4")
