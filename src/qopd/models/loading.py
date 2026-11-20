"""Model loaders: HF transformers (training) and vLLM (inference).

PLAN 1.1/1.3/1.4. Every loader wraps the load in :func:`qopd.utils.memory.track_memory`
(CLAUDE.md: log memory before and after each model load). torch/vllm import lazily.
"""

from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Any

from qopd.config import ModelCfg
from qopd.utils.logging import get_logger
from qopd.utils.memory import track_memory

log = get_logger("models")


def load_hf_model(cfg: ModelCfg | str, revision: str | None = None, device: str = "cuda") -> Any:
    """Load an ``AutoModelForCausalLM`` in BF16 (training / reference scoring)."""
    import torch  # noqa: PLC0415
    from transformers import AutoModelForCausalLM  # noqa: PLC0415

    path = cfg.path if isinstance(cfg, ModelCfg) else cfg
    rev = cfg.revision if isinstance(cfg, ModelCfg) and cfg.source_type == "hf" else revision
    with track_memory(f"load HF {path}", log):
        return AutoModelForCausalLM.from_pretrained(path, revision=rev, dtype=torch.bfloat16).to(
            device
        )


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


def build_llm(
    model_path: str,
    *,
    revision: str | None = None,
    kv_cache_gb: float = 4.0,
    gpu_memory_utilization: float = 0.12,
    max_model_len: int | None = None,
    max_logprobs: int = 20,
    dtype: str = "bfloat16",
    seed: int = 0,
    enforce_eager: bool = False,
    quantization: str | None = None,
    enable_prefix_caching: bool = False,
    tokenizer: str | None = None,
) -> Any:
    """Start a ``vllm.LLM`` configured for GB10 unified memory (lessons from step 1.3).

    - worker processes use ``spawn`` (a CUDA context in the parent breaks ``fork``);
    - the KV cache is sized in bytes, and ``gpu_memory_utilization`` is a small explicit
      share (vLLM checks it against *free* memory at startup);
    - ``max_model_len`` should be set explicitly (Qwen3's default is 40,960 tokens);
    - log-probs are raw (unprocessed logits); the model's generation_config.json is not
      applied as default sampling parameters, so every request sets them explicitly.
    """
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    from vllm import LLM  # noqa: PLC0415

    with track_memory(f"load vLLM {model_path}", log):
        return LLM(
            model=model_path,
            tokenizer=tokenizer,
            revision=revision,
            tokenizer_revision=revision if tokenizer is None else None,
            dtype=dtype,
            seed=seed,
            max_model_len=max_model_len,
            max_logprobs=max_logprobs,
            logprobs_mode="raw_logprobs",
            kv_cache_memory_bytes=int(kv_cache_gb * 1024**3),
            gpu_memory_utilization=gpu_memory_utilization,
            enable_prefix_caching=enable_prefix_caching,
            enforce_eager=enforce_eager,
            quantization=quantization,
            generation_config="vllm",
        )


def shutdown_llm(llm: Any, tag: str = "vLLM") -> None:
    """Stop the engine-core process and release its memory."""
    with track_memory(f"unload {tag}", log):
        core = getattr(getattr(llm, "llm_engine", None), "engine_core", None)
        if core is not None and callable(getattr(core, "shutdown", None)):
            core.shutdown()
        del llm
        gc.collect()
        try:
            import torch  # noqa: PLC0415

            torch.cuda.empty_cache()
        except ImportError:
            pass


def load_vllm_engine(cfg: ModelCfg, **kwargs: Any) -> Any:
    """:func:`build_llm` for a registry model at its pinned revision."""
    return build_llm(cfg.path, revision=cfg.revision, **kwargs)


def stop_token_ids(model_path: str, revision: str | None = None) -> list[int]:
    """EOS ids from the model's generation_config.json (Qwen3: <|im_end|>, <|endoftext|>)."""
    from transformers import GenerationConfig  # noqa: PLC0415

    eos = GenerationConfig.from_pretrained(model_path, revision=revision).eos_token_id
    if eos is None:
        return []
    return [int(e) for e in (eos if isinstance(eos, list) else [eos])]
