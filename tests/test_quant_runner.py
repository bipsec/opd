"""PLAN 4.2: recipe registry, calibration dataset from exact ids, recipe.json (CPU, no tools)."""

import pytest

from qopd.config import CONFIG_ROOT, ModelCfg, load
from qopd.quant.recipes import IGNORE_DEFAULT, RecipeResult, calibration_dataset
from qopd.quant.runner import RECIPES, build_rung, write_recipe_json
from qopd.utils import io


def test_every_quantized_rung_has_a_recipe():
    for f in (CONFIG_ROOT / "models").glob("*.yaml"):
        m = load(f, cls=ModelCfg)
        if m.quant_format:
            assert m.quant_format in RECIPES, m.id
    assert "lm_head" in IGNORE_DEFAULT and any("mlp.gate" in x for x in IGNORE_DEFAULT)


def test_calibration_dataset_exact_ids(tmp_path):
    pytest.importorskip("datasets")
    p = tmp_path / "c.jsonl"
    io.write_jsonl(
        p,
        [
            {"prompt_ids": [1, 2], "response_ids": [3, 4, 5]},
            {"prompt_ids": [6], "response_ids": list(range(10))},
        ],
    )
    ds = calibration_dataset(p, max_len=6)
    assert ds[0]["input_ids"] == [1, 2, 3, 4, 5] and ds[0]["attention_mask"] == [1] * 5
    assert len(ds[1]["input_ids"]) == 6
    assert len(calibration_dataset(p, n=1)) == 1


def test_recipe_json_round_trip(tmp_path):
    r = RecipeResult(
        "T8", tmp_path, "fp8_w8a8", "llmcompressor", "0.15.0", "ab" * 32, 12.0, {"scheme": "FP8"}
    )
    d = io.read_json(write_recipe_json(r))
    assert (d["tool_version"], d["extra"]["scheme"], d["out_dir"]) == (
        "0.15.0",
        "FP8",
        str(tmp_path),
    )


def test_build_rung_guards(tmp_path):
    with pytest.raises(ValueError, match="not a quantized rung"):
        build_rung("T16")
    with pytest.raises(FileNotFoundError, match="calib-traces"):
        build_rung("T3", calibration=tmp_path / "missing.jsonl")  # T3 not built yet
