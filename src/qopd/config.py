"""Typed configuration: pydantic models, YAML loading, ``key=value`` overrides (PLAN 0.3).

Conventions
- Every model forbids unknown keys, so a typo in a YAML file or an override fails loudly.
- ``None`` means "not decided yet" (TBD): either a later step decides it (e.g. the OPD
  learning rate from the 6.5 sweep) or it is defined only in the experiment-plan doc.
- Experiment files may compose other files with a ``defaults:`` mapping
  (``section: path/relative/to/configs.yaml``); inline values override them.
- Overrides use dotted keys and YAML scalars: ``opd.rounds=5 eval.sampling.top_p=0.9``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = PROJECT_ROOT / "configs"


class Cfg(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


# --- models -------------------------------------------------------------------


class Backend(StrEnum):
    VLLM = "vllm"
    HF = "hf"
    GGUF = "gguf"


class Role(StrEnum):
    STUDENT = "student"
    TEACHER = "teacher"


class ModelCfg(Cfg):
    """One model id from the registry (configs/models/<id>.yaml)."""

    id: str
    source: str  # HF repo id, or a path relative to the project root
    source_type: Literal["hf", "local"] = "hf"
    revision: str | None = None  # pinned HF commit (filled at download time)
    backend: Backend = Backend.VLLM
    role: Role
    bits: float = Field(gt=0, le=16)
    base: str | None = None  # BF16 parent id for quantized rungs
    quant_format: str | None = None  # name of configs/quant/<format>.yaml
    dtype: str = "bfloat16"

    @model_validator(mode="after")
    def _quantized_needs_parent(self) -> ModelCfg:
        if (self.quant_format is None) != (self.base is None):
            raise ValueError("quant_format and base must both be set (quantized) or both unset")
        return self

    @property
    def path(self) -> str:
        """Absolute path for local models, repo id for HF models."""
        return str(PROJECT_ROOT / self.source) if self.source_type == "local" else self.source


# --- quantization -------------------------------------------------------------


class CalibrationCfg(Cfg):
    """Doc: 512 calibration prompts, each paired with a BF16-teacher reasoning trace in chat
    format, cut at 2,048 tokens; the same texts for every rung (generated in PLAN 4.1)."""

    split: str = "calibration"
    n_samples: int = Field(512, gt=0)
    text_source: Literal["t16_traces"] = "t16_traces"
    chat_format: bool = True
    max_seq_len: int = Field(2048, gt=0)


class QuantCfg(Cfg):
    """One quantization format (configs/quant/<format>.yaml)."""

    format: str
    scheme: str  # human-readable, e.g. "W4A16"
    tool: str | None  # None = decided in step 4.2 (PLAN offers alternatives)
    weight_bits: float = Field(gt=0, le=16)
    group_size: int | None = None  # None = per-channel or defined by the format itself
    calibration: CalibrationCfg | None = Field(default_factory=CalibrationCfg)
    optional: bool = False
    params: dict[str, Any] = Field(default_factory=dict)  # tool-specific knobs


# --- data -----------------------------------------------------------------------


class DataCfg(Cfg):
    seed: int = 0
    train_source: str | None = None  # DeepScaleR-Preview or NuminaMath; decided in 2.1/2.4
    n_train: int = Field(16_000, gt=0)
    n_coldstart: int = Field(5_000, gt=0)  # subset of train
    n_calibration: int = Field(512, gt=0)  # disjoint from train
    n_probe: int = Field(500, gt=0)  # disjoint from train (E1)
    contamination_ngram: int = Field(13, gt=0)
    # Doc: MinHash on word 5-grams, 128 permutations, Jaccard >= 0.8 is a duplicate.
    near_dup_ngram: int = Field(5, gt=0)
    near_dup_num_perm: int = Field(128, gt=0)
    near_dup_threshold: float = Field(0.8, gt=0, le=1)
    data_dir: str = "data"

    @model_validator(mode="after")
    def _coldstart_subset(self) -> DataCfg:
        if self.n_coldstart > self.n_train:
            raise ValueError("n_coldstart must not exceed n_train")
        return self


# --- sampling -------------------------------------------------------------------


class SamplingCfg(Cfg):
    temperature: float = Field(ge=0)
    top_p: float = Field(1.0, gt=0, le=1)
    top_k: int = -1  # -1 = disabled (vLLM convention)
    max_tokens: int = Field(gt=0)
    enable_thinking: bool = True
    seed: int | None = None


# --- training -------------------------------------------------------------------


class OptimizerCfg(Cfg):
    """Doc (settled in 0.3): AdamW, betas 0.9/0.999, eps 1e-8, no weight decay, clip 1.0."""

    name: Literal["adamw"] = "adamw"
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = Field(1e-8, gt=0)
    weight_decay: float = Field(0.0, ge=0)
    grad_clip_norm: float | None = Field(1.0, gt=0)


class LoopMode(StrEnum):
    CO_RESIDENT = "co_resident"
    PHASE_ALTERNATING = "phase_alternating"


class OPDLossCfgModel(Cfg):
    clip_tau: float | None = Field(None, gt=0)  # None = no reward clipping
    top_k_kl: bool = False  # top-k KL variant (teacher top-k, renormalized)
    hybrid_weight: float = Field(0.0, ge=0)  # weight of the verifier reward


class OPDCfg(Cfg):
    prompts_per_round: int = Field(64, gt=0)
    rollouts_per_prompt: int = Field(4, gt=0)
    rounds: int = Field(100, gt=0)
    optimizer_steps_per_round: int = Field(1, gt=0)
    teacher_top_k: int = Field(20, gt=0)
    checkpoint_every: int = Field(20, gt=0)
    lr: float | None = Field(None, gt=0)  # set by the 6.5 sweep, then frozen
    optimizer: OptimizerCfg = Field(default_factory=OptimizerCfg)
    # Doc: all 256 rollouts form one batch -> one optimizer step via gradient accumulation.
    micro_batch_size: int | None = Field(None, gt=0)  # memory knob only; set in the pilot
    loss_averaging: Literal["batch_tokens"] = "batch_tokens"  # mean over all response tokens
    warmup_rounds: int = Field(5, ge=0)  # linear warmup, then constant
    schedule: Literal["warmup_constant"] = "warmup_constant"
    truncation_alert: float = Field(0.30, gt=0, le=1)  # doc: revisit limits if round-0 > 30%
    mode: LoopMode | None = None  # chosen at the 6.4 gate
    weight_sync: str = "reload_from_checkpoint"
    loss: OPDLossCfgModel = Field(default_factory=OPDLossCfgModel)
    rollout: SamplingCfg


class SFTCfg(Cfg):
    lr: float = Field(1e-5, gt=0)
    epochs: float = Field(1, gt=0)
    optimizer: OptimizerCfg = Field(default_factory=OptimizerCfg)
    dtype: str = "bfloat16"
    gradient_checkpointing: bool = True
    response_only_loss: bool = True
    loss_averaging: Literal["batch_tokens"] = "batch_tokens"
    global_batch_size: int = Field(32, gt=0)  # sequences per optimizer step
    micro_batch_size: int | None = Field(None, gt=0)  # memory knob only; set in step 5.2
    # Doc: linear warmup over 3% of steps, cosine decay to 10% of peak.
    schedule: Literal["warmup_cosine"] = "warmup_cosine"
    warmup_ratio: float = Field(0.03, ge=0, lt=1)
    min_lr_ratio: float = Field(0.1, ge=0, le=1)
    # Doc: 8,192 tokens prompt + response; longer traces dropped (not truncated); no packing.
    max_seq_len: int = Field(8192, gt=0)
    drop_overlong: bool = True
    packing: bool = False


# --- evaluation -----------------------------------------------------------------


class SuiteCfg(Cfg):
    n_samples: int = Field(gt=0)
    kind: Literal["math", "mc"] = "math"
    gated: bool = False  # needs the user's HF access
    enabled: bool = True
    source: str | None = None  # dataset id, pinned in step 2.2


class EvalCfg(Cfg):
    sampling: SamplingCfg
    suites: dict[str, SuiteCfg]


# --- cluster ----------------------------------------------------------------------


class NodeCfg(Cfg):
    name: str
    host: str
    user: str
    hostname: str = ""
    role: Literal["primary", "secondary"] = "secondary"


class ClusterCfg(Cfg):
    env_file: str = ".env"
    nodes: list[NodeCfg]

    @model_validator(mode="after")
    def _one_primary(self) -> ClusterCfg:
        if sum(n.role == "primary" for n in self.nodes) != 1:
            raise ValueError("exactly one node must have role: primary")
        return self


# --- experiments ------------------------------------------------------------------


class ExperimentCfg(Cfg):
    name: str
    student: str
    teacher: str | None = None
    seeds: list[int] = Field(default_factory=lambda: [0], min_length=1)
    node: str = "local"
    data: DataCfg | None = None
    opd: OPDCfg | None = None
    sft: SFTCfg | None = None
    eval: EvalCfg | None = None
    # Resource-driven deviations from the design (CLAUDE.md); shown in every report.
    amendments: list[str] = Field(default_factory=list)
    notes: str = ""


# --- loading ----------------------------------------------------------------------

C = TypeVar("C", bound=BaseModel)

_DIR_KINDS: dict[str, type[BaseModel]] = {
    "models": ModelCfg,
    "quant": QuantCfg,
    "data": DataCfg,
    "eval": EvalCfg,
    "experiments": ExperimentCfg,
}
_FILE_KINDS: dict[str, type[BaseModel]] = {
    "train/opd": OPDCfg,
    "train/sft": SFTCfg,
    "cluster": ClusterCfg,
}


def infer_kind(path: Path) -> type[BaseModel]:
    """Pick the config model from the file's place under ``configs/``."""
    path = Path(path).resolve()
    root = _configs_root(path)
    rel = path.relative_to(root).with_suffix("").as_posix()
    if rel in _FILE_KINDS:
        return _FILE_KINDS[rel]
    top = rel.split("/", 1)[0]
    if top in _DIR_KINDS:
        return _DIR_KINDS[top]
    raise ValueError(f"cannot infer config type for {path}; pass cls=")


def _configs_root(path: Path) -> Path:
    for parent in path.parents:
        if parent.name == "configs":
            return parent
    return CONFIG_ROOT


def deep_merge(base: Mapping[str, Any], over: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _read_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text())
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    return data


def _resolve_defaults(raw: dict[str, Any], path: Path) -> dict[str, Any]:
    defaults = raw.pop("defaults", None)
    if not defaults:
        return raw
    if not isinstance(defaults, Mapping):
        raise ValueError(f"{path}: defaults must map section -> file")
    root = _configs_root(Path(path).resolve())
    merged: dict[str, Any] = {}
    for section, rel in defaults.items():
        sub = _resolve_defaults(_read_yaml(root / rel), root / rel)
        merged[section] = sub
    return deep_merge(merged, raw)


_FLOAT_RE = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$")


def parse_value(value: str) -> Any:
    """YAML scalar parsing, plus floats like ``2e-5`` that YAML 1.1 reads as strings."""
    if value == "":
        return ""
    parsed = yaml.safe_load(value)
    if isinstance(parsed, str) and _FLOAT_RE.match(parsed.strip()):
        return float(parsed)
    return parsed


def parse_override(item: str) -> tuple[list[str], Any]:
    """``"a.b=3"`` -> (["a", "b"], 3). Values are parsed as YAML scalars."""
    key, sep, value = item.partition("=")
    if not sep or not key or any(not part for part in key.split(".")):
        raise ValueError(f"bad override {item!r}; expected dotted.key=value")
    return key.split("."), parse_value(value)


def apply_overrides(raw: dict[str, Any], overrides: Iterable[str]) -> dict[str, Any]:
    out = deep_merge(raw, {})
    for item in overrides:
        keys, value = parse_override(item)
        node = out
        for k in keys[:-1]:
            child = node.get(k)
            if child is None:
                child = node[k] = {}
            elif not isinstance(child, dict):
                raise ValueError(f"override {item!r}: {k!r} is not a mapping")
            else:
                child = node[k] = dict(child)
            node = child
        node[keys[-1]] = value
    return out


def load_raw(path: str | Path, overrides: Iterable[str] = ()) -> dict[str, Any]:
    """YAML -> dict with ``defaults`` composed and overrides applied (not validated)."""
    path = Path(path)
    return apply_overrides(_resolve_defaults(_read_yaml(path), path), overrides)


def load(path: str | Path, overrides: Iterable[str] = (), cls: type[C] | None = None) -> C:
    """Load, compose, override and validate a config file."""
    model = cls or infer_kind(Path(path))
    return model.model_validate(load_raw(path, overrides))  # type: ignore[return-value]


def to_dict(cfg: BaseModel) -> dict[str, Any]:
    return cfg.model_dump(mode="json")


def dump(cfg: BaseModel, path: str | Path) -> Path:
    """Write the fully resolved config as YAML."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(to_dict(cfg), sort_keys=False, allow_unicode=True))
    return path


def write_resolved(cfg: BaseModel, run_dir: str | Path) -> Path:
    """Write ``config.yaml`` (the resolved config) into a run directory."""
    return dump(cfg, Path(run_dir) / "config.yaml")


def all_config_files(root: Path = CONFIG_ROOT) -> list[Path]:
    return sorted(p for p in root.rglob("*.yaml") if p.is_file())
