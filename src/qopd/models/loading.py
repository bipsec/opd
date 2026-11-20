"""Model loaders: HF transformers (training) and vLLM (inference).

PLAN 1.1/1.3/1.4. Every loader wraps the load in :func:`qopd.utils.memory.track_memory`
(CLAUDE.md: log memory before and after each model load). torch/vllm import lazily.
"""

from __future__ import annotations

from typing import Any

from qopd.config import ModelCfg, SamplingCfg


def load_hf_model(cfg: ModelCfg, **kwargs: Any) -> Any:
    """Load an ``AutoModelForCausalLM`` in BF16 for training or reference scoring."""
    raise NotImplementedError("step 1.3")


def load_tokenizer(cfg: ModelCfg) -> Any:
    """Load the tokenizer for ``cfg`` (HF ``AutoTokenizer``)."""
    raise NotImplementedError("step 1.1")


def load_vllm_engine(
    cfg: ModelCfg, gpu_memory_utilization: float, sampling: SamplingCfg | None = None, **kwargs: Any
) -> Any:
    """Start a ``vllm.LLM`` for ``cfg`` with an explicit ``gpu_memory_utilization``."""
    raise NotImplementedError("step 1.4")
