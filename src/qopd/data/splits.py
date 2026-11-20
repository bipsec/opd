"""Fixed-seed splits and the data manifest (PLAN 2.4).

From the filtered pool: train (16k), cold-start subset of train (5k), calibration (512,
disjoint from train) and probe (500, disjoint from train) for E1. ``data/manifest.json``
records counts and sha256 of every split file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from qopd.types import Problem


@dataclass
class Splits:
    train: list[Problem]
    coldstart: list[Problem]  # subset of train
    calibration: list[Problem]  # disjoint from train
    probe: list[Problem]  # disjoint from train


def make_splits(
    pool: list[Problem],
    seed: int,
    n_train: int,
    n_coldstart: int,
    n_calibration: int,
    n_probe: int,
) -> Splits:
    raise NotImplementedError("step 2.4")


def check_disjoint(splits: Splits) -> None:
    """Raise if calibration or probe overlap train, or cold-start is not a subset of train."""
    raise NotImplementedError("step 2.4")


def write_manifest(splits: Splits, out_dir: Path) -> Path:
    """Write each split to parquet and ``manifest.json`` with counts and hashes."""
    raise NotImplementedError("step 2.4")


def verify_manifest(manifest: Path) -> None:
    """Raise if any split file's hash differs from the manifest."""
    raise NotImplementedError("step 2.4")
