"""Run directory contract (CLAUDE.md).

Every training, scoring or evaluation job writes to ``runs/<experiment>/<run_id>/``::

    config.yaml  git_sha.txt  env.json  stdout.log  metrics.jsonl  monitor.csv
    state.json   checkpoints/  DONE (only when finished successfully)

Re-opening an existing run directory resumes it; the stored config must match.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from qopd.utils import io
from qopd.utils.env import snapshot

CONTRACT_FILES = (
    "config.yaml",
    "git_sha.txt",
    "env.json",
    "stdout.log",
    "metrics.jsonl",
    "monitor.csv",
    "state.json",
)
CONTRACT_DIRS = ("checkpoints",)
DONE = "DONE"
PROJECT_ROOT = Path(__file__).resolve().parents[3]


class RunConfigMismatchError(RuntimeError):
    """An existing run directory holds a different config than the one requested."""


def _to_mapping(cfg: Any) -> dict[str, Any]:
    if hasattr(cfg, "model_dump"):  # pydantic model (qopd.config)
        return cfg.model_dump(mode="json")
    if isinstance(cfg, Mapping):
        return dict(cfg)
    raise TypeError(f"config must be a mapping or pydantic model, got {type(cfg).__name__}")


def git_sha(repo: Path = PROJECT_ROOT) -> str:
    """HEAD sha (``+dirty`` if the tree has changes) or ``nogit`` outside a repository."""
    try:
        sha = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "nogit"
    return f"{sha}+dirty" if dirty else sha


def default_runs_root() -> Path:
    return Path(os.environ.get("QOPD_RUNS", PROJECT_ROOT / "runs"))


@dataclass
class RunDir:
    path: Path

    # --- paths -------------------------------------------------------------
    @property
    def config(self) -> Path:
        return self.path / "config.yaml"

    @property
    def metrics(self) -> Path:
        return self.path / "metrics.jsonl"

    @property
    def monitor(self) -> Path:
        return self.path / "monitor.csv"

    @property
    def state_file(self) -> Path:
        return self.path / "state.json"

    @property
    def stdout(self) -> Path:
        return self.path / "stdout.log"

    @property
    def checkpoints(self) -> Path:
        return self.path / "checkpoints"

    # --- state -------------------------------------------------------------
    def load_config(self) -> dict[str, Any]:
        return yaml.safe_load(self.config.read_text()) or {}

    def log_metrics(self, row: Mapping[str, Any]) -> None:
        io.append_jsonl(self.metrics, dict(row))

    def load_state(self) -> dict[str, Any]:
        return io.read_json(self.state_file)

    def save_state(self, **updates: Any) -> dict[str, Any]:
        """Merge ``updates`` into state.json atomically and return the new state."""
        state = self.load_state() if self.state_file.exists() else {}
        state.update(updates)
        state["updated"] = datetime.now(UTC).isoformat(timespec="seconds")
        io.write_json(self.state_file, state)
        return state

    @property
    def is_done(self) -> bool:
        return (self.path / DONE).exists()

    def mark_done(self) -> None:
        self.save_state(status="done")
        (self.path / DONE).write_text(datetime.now(UTC).isoformat(timespec="seconds") + "\n")

    def missing(self) -> list[str]:
        """Contract entries that do not exist (empty list = contract satisfied)."""
        names = [f for f in CONTRACT_FILES if not (self.path / f).is_file()]
        return names + [d for d in CONTRACT_DIRS if not (self.path / d).is_dir()]


def create_run_dir(
    experiment: str,
    run_id: str,
    cfg: Any,
    root: Path | None = None,
) -> RunDir:
    """Create (or resume) ``<root>/<experiment>/<run_id>`` with every contract file."""
    rd = RunDir((root or default_runs_root()) / experiment / run_id)
    cfg_map = _to_mapping(cfg)
    if rd.config.exists():
        if rd.load_config() != cfg_map:
            raise RunConfigMismatchError(
                f"{rd.path} exists with a different config.yaml; use a new run_id"
            )
        rd.save_state(resumed=True)
        return rd

    rd.path.mkdir(parents=True, exist_ok=True)
    rd.checkpoints.mkdir(exist_ok=True)
    rd.config.write_text(yaml.safe_dump(cfg_map, sort_keys=False))
    (rd.path / "git_sha.txt").write_text(git_sha() + "\n")
    io.write_json(rd.path / "env.json", snapshot())
    for name in ("stdout.log", "metrics.jsonl"):
        (rd.path / name).touch()
    if not rd.monitor.exists():
        from qopd.utils.monitor import write_header  # noqa: PLC0415

        write_header(rd.monitor)
    rd.save_state(status="created", experiment=experiment, run_id=run_id)
    return rd
