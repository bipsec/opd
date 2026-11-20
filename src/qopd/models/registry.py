"""Model registry: short ids -> :class:`qopd.config.ModelCfg` (PLAN 1.1).

The single source of truth is ``configs/models/<id>.yaml``. HF repo ids for S1, S2, T16
and D16 come from PLAN.md; quantized rungs are local directories built in Phase 4
(``models/quant/<rung>/``). :func:`verify_exists` checks a repo id against the Hub
before first use (step 1.1).
"""

from __future__ import annotations

import re
from enum import StrEnum
from functools import cache
from pathlib import Path

from qopd.config import CONFIG_ROOT, Backend, ModelCfg, Role, load

__all__ = [
    "Backend",
    "ModelId",
    "ModelNotFoundError",
    "Role",
    "load_registry",
    "pin_revision",
    "resolve",
    "teachers",
    "verify_exists",
]

MODELS_DIR = CONFIG_ROOT / "models"


class ModelId(StrEnum):
    S1 = "S1"  # Qwen3-1.7B student
    S2 = "S2"  # Qwen3-0.6B student
    T16 = "T16"  # Qwen3-30B-A3B BF16
    T8 = "T8"  # FP8 W8A8
    T4_FP = "T4-fp"  # NVFP4
    T4_INT = "T4-int"  # INT4 AWQ W4A16 g128
    T3 = "T3"  # 3-bit GPTQ g128
    T2 = "T2"  # 2-bit GGUF Q2_K (optional)
    D16 = "D16"  # Qwen3-8B BF16 (equal-memory study, E4)
    D8 = "D8"
    D4 = "D4"


@cache
def load_registry(models_dir: Path = MODELS_DIR) -> dict[ModelId, ModelCfg]:
    """Load every ``<id>.yaml``; file stem, ``id`` field and :class:`ModelId` must agree."""
    reg: dict[ModelId, ModelCfg] = {}
    for path in sorted(models_dir.glob("*.yaml")):
        cfg = load(path, cls=ModelCfg)
        if cfg.id != path.stem:
            raise ValueError(f"{path}: id {cfg.id!r} does not match the file name")
        reg[ModelId(cfg.id)] = cfg
    missing = set(ModelId) - set(reg)
    if missing:
        raise ValueError(f"no config for model ids: {sorted(m.value for m in missing)}")
    for cfg in reg.values():
        if cfg.base is not None and ModelId(cfg.base) not in reg:
            raise ValueError(f"{cfg.id}: unknown base {cfg.base!r}")
    return reg


def resolve(model_id: str | ModelId) -> ModelCfg:
    """Look up a model id (e.g. ``"T4-int"``); raises KeyError for unknown ids."""
    try:
        key = ModelId(model_id)
    except ValueError as e:
        raise KeyError(f"unknown model id {model_id!r}; known: {[m.value for m in ModelId]}") from e
    return load_registry()[key]


def teachers() -> list[ModelCfg]:
    return [c for c in load_registry().values() if c.role == Role.TEACHER]


class ModelNotFoundError(RuntimeError):
    """A model named in the plan cannot be found (CLAUDE.md rule 8: stop, do not substitute)."""


def verify_exists(cfg: ModelCfg) -> str:
    """Check that the HF repo (at the pinned revision) or the local directory exists.

    Returns the HF commit sha, or ``"local"``. Raises :class:`ModelNotFoundError` listing
    the options instead of picking a replacement.
    """
    if cfg.source_type == "local":
        path = Path(cfg.path)
        if not (path / "config.json").exists() and not any(path.glob("*.gguf")):
            raise ModelNotFoundError(
                f"{cfg.id}: local model {path} not built yet "
                "(options: build it in Phase 4, or remove the rung)"
            )
        return "local"
    from huggingface_hub import HfApi  # noqa: PLC0415
    from huggingface_hub.utils import HfHubHTTPError  # noqa: PLC0415

    try:
        info = HfApi().model_info(cfg.source, revision=cfg.revision)
    except HfHubHTTPError as e:
        raise ModelNotFoundError(
            f"{cfg.id}: {cfg.source}@{cfg.revision or 'main'} not found on the Hub ({e}). "
            "Options: (1) check the repo id / revision, (2) log in if it is gated, "
            "(3) ask the user for a replacement."
        ) from e
    return str(info.sha)


_REV_LINE = re.compile(r"^revision:[^\n#]*(#.*)?$", re.MULTILINE)


def pin_revision(model_id: str | ModelId, sha: str, models_dir: Path = MODELS_DIR) -> Path:
    """Write ``revision: <sha>`` into configs/models/<id>.yaml, keeping comments."""
    path = models_dir / f"{ModelId(model_id).value}.yaml"
    text = path.read_text()
    new, n = _REV_LINE.subn(f"revision: {sha}          # pinned commit", text)
    if n != 1:
        raise ValueError(f"{path}: expected exactly one top-level 'revision:' line")
    path.write_text(new)
    load_registry.cache_clear()
    return path
