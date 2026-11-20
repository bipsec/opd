"""Every qopd module must import cheaply (no torch / vllm at import time)."""

import importlib
import pkgutil
import subprocess
import sys

import pytest

import qopd

MODULES = sorted(
    m.name
    for m in pkgutil.walk_packages(qopd.__path__, prefix="qopd.")
    if m.name != "qopd.__main__"
)


def test_module_list_covers_map():
    expected = {
        "qopd.cli",
        "qopd.types",
        "qopd.models.registry",
        "qopd.models.loading",
        "qopd.models.tokenizer_check",
        "qopd.prompts.templates",
        "qopd.data.sources",
        "qopd.data.filters",
        "qopd.data.contamination",
        "qopd.data.splits",
        "qopd.quant.runner",
        "qopd.quant.recipes.fp8",
        "qopd.quant.recipes.nvfp4",
        "qopd.quant.recipes.awq_int4",
        "qopd.quant.recipes.gptq3",
        "qopd.quant.recipes.gguf",
        "qopd.rollout.engine",
        "qopd.teacher.base",
        "qopd.teacher.vllm_backend",
        "qopd.teacher.hf_backend",
        "qopd.teacher.gguf_backend",
        "qopd.train.sft",
        "qopd.train.opd_loss",
        "qopd.train.opd_loop",
        "qopd.train.grpo",
        "qopd.eval.verifiers",
        "qopd.eval.runner",
        "qopd.eval.suites",
        "qopd.fidelity.pool",
        "qopd.fidelity.metrics",
        "qopd.orchestration.grid",
        "qopd.orchestration.queue",
        "qopd.stats.intervals",
        "qopd.stats.tests",
        "qopd.utils.run",
        "qopd.utils.monitor",
        "qopd.utils.seed",
        "qopd.utils.io",
        "qopd.utils.logging",
        "qopd.utils.memory",
        "qopd.utils.env",
        "qopd.utils.cluster",
    }
    assert expected <= set(MODULES), sorted(expected - set(MODULES))


@pytest.mark.parametrize("name", MODULES)
def test_imports(name):
    importlib.import_module(name)


def test_no_heavy_imports():
    """Importing every module in a fresh interpreter must not pull in vllm/transformers."""
    code = (
        "import importlib, sys\n"
        f"for m in {MODULES!r}: importlib.import_module(m)\n"
        "print(','.join(m for m in ('vllm', 'transformers') if m in sys.modules))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "", out.stdout
