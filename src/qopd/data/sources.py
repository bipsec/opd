"""Dataset download and conversion to the unified schema (PLAN 2.1, 2.2).

Training pool: a public math set with checkable answers (DeepScaleR-Preview or
NuminaMath, to be confirmed at the 2.4 checkpoint). Evaluation suites: see
``qopd.eval.suites``. If a dataset is not found under its expected name, stop and ask
(CLAUDE.md rule 8); never substitute.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from qopd.types import Problem


@dataclass(frozen=True)
class SourceSpec:
    name: str  # short name used in Problem.source
    hf_repo: str
    split: str
    problem_field: str
    answer_field: str
    config: str | None = None
    gated: bool = False


class DatasetNotFoundError(RuntimeError):
    """Raised instead of silently substituting a dataset."""


def download(spec: SourceSpec, cache_dir: Path | None = None) -> Path:
    """Download ``spec`` into the HF cache and return the local path."""
    raise NotImplementedError("step 2.1")


def to_problems(spec: SourceSpec, rows: Iterable[dict[str, object]]) -> list[Problem]:
    """Map raw rows to :class:`Problem` with stable ids (``<source>/<index or hash>``)."""
    raise NotImplementedError("step 2.1")


def write_parquet(problems: list[Problem], path: Path) -> str:
    """Write problems to parquet; return the file's sha256."""
    raise NotImplementedError("step 2.1")


def read_parquet(path: Path) -> list[Problem]:
    raise NotImplementedError("step 2.1")
