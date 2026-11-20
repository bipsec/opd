"""Model loaders: HF transformers (training) and vLLM (inference).

PLAN 1.1/1.3/1.4. Every loader wraps the load in :func:`qopd.utils.memory.track_memory`
(CLAUDE.md: log memory before and after each model load). torch/vllm import lazily.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from qopd.config import ModelCfg, SamplingCfg
from qopd.utils.logging import get_logger

log = get_logger("models")


def load_hf_model(cfg: ModelCfg, **kwargs: Any) -> Any:
    """Load an ``AutoModelForCausalLM`` in BF16 for training or reference scoring."""
    raise NotImplementedError("step 1.3")


def load_tokenizer(cfg: ModelCfg, local_files_only: bool = False) -> Any:
    """Load the fast tokenizer for ``cfg`` at its pinned revision (HF ``AutoTokenizer``)."""
    from transformers import AutoTokenizer  # noqa: PLC0415

    kwargs: dict[str, Any] = {"use_fast": True, "local_files_only": local_files_only}
    if cfg.source_type == "hf":
        kwargs["revision"] = cfg.revision
    return AutoTokenizer.from_pretrained(cfg.path, **kwargs)


def download(cfg: ModelCfg, tokenizer_only: bool = False) -> tuple[Path, str]:
    """Download ``cfg`` into the HF cache; return (local snapshot dir, commit sha).

    Uses the pinned ``revision`` if set, else the default branch (the caller pins the sha).
    """
    if cfg.source_type != "hf":
        raise ValueError(f"{cfg.id} is a local model ({cfg.source}); nothing to download")
    from huggingface_hub import snapshot_download  # noqa: PLC0415

    patterns = ["*.json", "*.txt", "*.jinja", "*.model", "*.tiktoken"] if tokenizer_only else None
    path = Path(
        snapshot_download(cfg.source, revision=cfg.revision, allow_patterns=patterns)
    )
    sha = path.name  # snapshot dirs are named by commit sha
    log.info("downloaded %s (%s) -> %s", cfg.id, cfg.source, path)
    return path, sha


def load_vllm_engine(
    cfg: ModelCfg, gpu_memory_utilization: float, sampling: SamplingCfg | None = None, **kwargs: Any
) -> Any:
    """Start a ``vllm.LLM`` for ``cfg`` with an explicit ``gpu_memory_utilization``."""
    raise NotImplementedError("step 1.4")
