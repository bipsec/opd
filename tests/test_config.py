"""Config round-trip (PLAN 0.3 "done when"): load → override → dump → reload, every file."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from qopd.config import (
    CONFIG_ROOT,
    ClusterCfg,
    DataCfg,
    EvalCfg,
    ExperimentCfg,
    ModelCfg,
    OPDCfg,
    QuantCfg,
    SFTCfg,
    all_config_files,
    apply_overrides,
    dump,
    infer_kind,
    load,
    parse_override,
    write_resolved,
)
from qopd.models.registry import ModelId, load_registry
from qopd.utils.run import create_run_dir

FILES = all_config_files()

# One override per config type that changes a real value.
OVERRIDES = {
    ModelCfg: ["revision=abc123"],
    QuantCfg: ["calibration.n_samples=256"],
    DataCfg: ["seed=7"],
    OPDCfg: ["rounds=5", "rollout.max_tokens=512"],
    SFTCfg: ["lr=2e-5"],
    EvalCfg: ["sampling.top_p=0.9", "suites.math500.n_samples=2"],
    ClusterCfg: ["env_file=other.env"],
    ExperimentCfg: ["seeds=[0,1]", "notes=round-trip"],
}


def test_found_every_config_dir():
    rel = {p.relative_to(CONFIG_ROOT).parts[0] for p in FILES}
    assert {"models", "quant", "data", "train", "eval", "experiments", "cluster.yaml"} <= rel


@pytest.mark.parametrize("path", FILES, ids=lambda p: str(p.relative_to(CONFIG_ROOT)))
def test_round_trip(path: Path, tmp_path: Path):
    cls = infer_kind(path)
    overrides = OVERRIDES[cls]
    cfg = load(path, overrides)
    out = dump(cfg, tmp_path / "resolved.yaml")
    again = load(out, cls=cls)
    assert again == cfg
    # overrides really landed
    for item in overrides:
        keys, value = parse_override(item)
        node = again.model_dump(mode="json")
        for k in keys:
            node = node[k]
        assert node == value


@pytest.mark.parametrize("path", FILES, ids=lambda p: str(p.relative_to(CONFIG_ROOT)))
def test_unknown_key_rejected(path: Path):
    with pytest.raises(ValidationError):
        load(path, ["definitely_not_a_field=1"])


def test_override_value_parsing():
    assert parse_override("lr=2e-5") == (["lr"], 2e-5)
    assert parse_override("a.b=3") == (["a", "b"], 3)
    assert parse_override("x=null") == (["x"], None)
    assert parse_override("x=[0,1]") == (["x"], [0, 1])
    assert parse_override("x=co_resident") == (["x"], "co_resident")
    assert parse_override("x=1.5") == (["x"], 1.5)


def test_bad_override_syntax():
    for bad in ["rounds", "=5", "a..b=1", ".a=1"]:
        with pytest.raises(ValueError):
            parse_override(bad)
    with pytest.raises(ValueError):
        apply_overrides({"a": 1}, ["a.b=2"])  # a is not a mapping


def test_override_type_validation():
    with pytest.raises(ValidationError):
        load(CONFIG_ROOT / "train" / "opd.yaml", ["rounds=-1"])
    with pytest.raises(ValidationError):
        load(CONFIG_ROOT / "eval" / "default.yaml", ["sampling.top_p=1.5"])


def test_plan_defaults():
    opd = load(CONFIG_ROOT / "train" / "opd.yaml")
    assert (opd.prompts_per_round, opd.rollouts_per_prompt, opd.rounds) == (64, 4, 100)
    assert (opd.teacher_top_k, opd.checkpoint_every) == (20, 20)
    assert (opd.rollout.temperature, opd.rollout.max_tokens) == (1.0, 4096)
    assert opd.lr is None  # set by the 6.5 sweep
    ev = load(CONFIG_ROOT / "eval" / "default.yaml")
    assert (ev.sampling.temperature, ev.sampling.top_p, ev.sampling.enable_thinking) == (
        0.6,
        0.95,
        True,
    )
    assert {k: s.n_samples for k, s in ev.suites.items()} == {
        "gsm8k": 1,
        "math500": 4,
        "amc23": 8,
        "aime24": 16,
        "aime25": 16,
        "aime26": 16,
        "hmmt_feb26": 16,
        "gpqa_diamond": 4,
    }
    data = load(CONFIG_ROOT / "data" / "default.yaml")
    assert (data.n_train, data.n_coldstart, data.n_calibration, data.n_probe) == (
        16000,
        5000,
        512,
        500,
    )
    sft = load(CONFIG_ROOT / "train" / "sft.yaml")
    assert (sft.lr, sft.epochs) == (1e-5, 1)


def test_doc_training_details():
    """EXPERIMENT_PLAN "Training details settled in build step 0.3"."""
    opd = load(CONFIG_ROOT / "train" / "opd.yaml")
    sft = load(CONFIG_ROOT / "train" / "sft.yaml")
    for opt in (opd.optimizer, sft.optimizer):
        assert opt.name == "adamw"
        assert opt.betas == (0.9, 0.999)
        assert opt.eps == 1e-8
        assert opt.weight_decay == 0.0
        assert opt.grad_clip_norm == 1.0
    assert opd.loss_averaging == sft.loss_averaging == "batch_tokens"
    assert (opd.warmup_rounds, opd.schedule, opd.truncation_alert) == (5, "warmup_constant", 0.30)
    assert (sft.global_batch_size, sft.warmup_ratio, sft.min_lr_ratio) == (32, 0.03, 0.1)
    assert (sft.max_seq_len, sft.drop_overlong, sft.packing) == (8192, True, False)
    data = load(CONFIG_ROOT / "data" / "default.yaml")
    assert (data.near_dup_ngram, data.near_dup_num_perm, data.near_dup_threshold) == (5, 128, 0.8)
    for path in (CONFIG_ROOT / "quant").glob("*.yaml"):
        cal = load(path).calibration
        assert cal is not None
        assert (cal.n_samples, cal.max_seq_len, cal.text_source) == (512, 2048, "t16_traces")


def test_experiment_defaults_compose():
    exp = load(CONFIG_ROOT / "experiments" / "e0_pilot.yaml")
    assert exp.opd is not None and exp.opd.rounds == 20  # inline wins
    assert exp.opd.prompts_per_round == 64  # from train/opd.yaml
    assert exp.eval is not None and "math500" in exp.eval.suites
    exp2 = load(CONFIG_ROOT / "experiments" / "e0_pilot.yaml", ["opd.loss.clip_tau=2"])
    assert exp2.opd is not None and exp2.opd.loss.clip_tau == 2


def test_registry_matches_model_files():
    reg = load_registry()
    assert set(reg) == set(ModelId)
    assert reg[ModelId.S1].source == "Qwen/Qwen3-1.7B"
    assert reg[ModelId.S2].source == "Qwen/Qwen3-0.6B"
    assert reg[ModelId.T16].source == "Qwen/Qwen3-30B-A3B"
    assert reg[ModelId.D16].source == "Qwen/Qwen3-8B"
    quant_formats = {p.stem for p in (CONFIG_ROOT / "quant").glob("*.yaml")}
    for cfg in reg.values():
        if cfg.quant_format is not None:
            assert cfg.quant_format in quant_formats, cfg.id


def test_quantized_requires_parent():
    with pytest.raises(ValidationError):
        ModelCfg(id="X", source="x", role="teacher", bits=4, quant_format="nvfp4")


def test_write_resolved_into_run_dir(tmp_path: Path):
    exp = load(CONFIG_ROOT / "experiments" / "e0_pilot.yaml")
    rd = create_run_dir("e0_pilot", "test", exp, root=tmp_path)
    assert load(rd.config, cls=ExperimentCfg) == exp
    write_resolved(exp, tmp_path / "other")
    assert yaml.safe_load((tmp_path / "other" / "config.yaml").read_text())["name"] == "e0_pilot"
