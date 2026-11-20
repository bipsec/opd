"""Benchmark suite definitions (PLAN 2.2, 3.2).

Suites and samples per problem live in ``configs/eval/default.yaml`` (:class:`EvalCfg`).
Source datasets are pinned in step 2.2; a suite that cannot be found under its expected
name stops the step (no substitution).
"""

from __future__ import annotations

from pathlib import Path

from qopd.config import CONFIG_ROOT, EvalCfg, SuiteCfg, load

DEFAULT_EVAL = CONFIG_ROOT / "eval" / "default.yaml"


def load_suites(path: Path = DEFAULT_EVAL) -> dict[str, SuiteCfg]:
    return load(path, cls=EvalCfg).suites


def get_suite(name: str, cfg: EvalCfg | None = None) -> SuiteCfg:
    suites = cfg.suites if cfg is not None else load_suites()
    try:
        return suites[name]
    except KeyError as e:
        raise KeyError(f"unknown suite {name!r}; known: {sorted(suites)}") from e


def enabled_suites(cfg: EvalCfg) -> list[str]:
    return [name for name, s in cfg.suites.items() if s.enabled]
