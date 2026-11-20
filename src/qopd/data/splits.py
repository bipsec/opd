"""Fixed-seed splits and the data manifest (PLAN 2.4).

From the decontaminated pool, ordered by ``sha256(f"{seed}:{id}")`` (a seeded random order
that is stable under removals: dropping a problem later, e.g. after decontaminating against
GPQA-Diamond, does not reshuffle the others):

- ``train``        the first ``n_train`` problems
- ``coldstart``    the first ``n_coldstart`` of ``train`` (a uniform subset; train order is random)
- ``calibration``  the next ``n_calibration`` problems (disjoint from train)
- ``probe``        the next ``n_probe`` problems (disjoint from train and calibration; E1)

The rest of the pool is unused (``reserve``, counted in the manifest). ``data/manifest.json``
records counts, file sha256 and a sha256 over the sorted ids of every split, plus the
evaluation suite files, so later steps can verify they read exactly these data.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from qopd.config import DataCfg, project_relative
from qopd.data.sources import read_parquet, write_parquet
from qopd.types import Problem
from qopd.utils import io
from qopd.utils.io import sha256_file

SPLITS = ("train", "coldstart", "calibration", "probe")
MANIFEST_VERSION = 1


class ManifestMismatchError(RuntimeError):
    """A data file does not match the hash recorded in data/manifest.json."""


@dataclass
class Splits:
    train: list[Problem]
    coldstart: list[Problem]  # subset of train
    calibration: list[Problem]  # disjoint from train
    probe: list[Problem]  # disjoint from train and calibration
    n_reserve: int = 0

    def as_dict(self) -> dict[str, list[Problem]]:
        return {name: getattr(self, name) for name in SPLITS}


def make_splits(
    pool: list[Problem],
    seed: int,
    n_train: int,
    n_coldstart: int,
    n_calibration: int,
    n_probe: int,
) -> Splits:
    need = n_train + n_calibration + n_probe
    if len(pool) < need:
        raise ValueError(f"pool has {len(pool)} problems, splits need {need}")
    if n_coldstart > n_train:
        raise ValueError("n_coldstart must not exceed n_train")
    if len({p.id for p in pool}) != len(pool):
        raise ValueError("pool ids are not unique")
    order = sorted(pool, key=lambda p: split_key(seed, p.id))  # input-order independent
    train = order[:n_train]
    cal = order[n_train : n_train + n_calibration]
    probe = order[n_train + n_calibration : need]
    s = Splits(train, train[:n_coldstart], cal, probe, n_reserve=len(pool) - need)
    check_disjoint(s)
    return s


def split_key(seed: int, pid: str) -> str:
    return hashlib.sha256(f"{seed}:{pid}".encode()).hexdigest()


def check_disjoint(splits: Splits) -> None:
    """Raise if calibration/probe overlap train or each other, or cold-start is not in train."""
    ids = {name: {p.id for p in ps} for name, ps in splits.as_dict().items()}
    for name, ps in splits.as_dict().items():
        if len(ids[name]) != len(ps):
            raise ValueError(f"{name} has duplicate ids")
    if not ids["coldstart"] <= ids["train"]:
        raise ValueError("coldstart is not a subset of train")
    for a, b in (("train", "calibration"), ("train", "probe"), ("calibration", "probe")):
        if overlap := ids[a] & ids[b]:
            raise ValueError(f"{a} and {b} share {len(overlap)} ids")


def ids_sha256(problems: list[Problem]) -> str:
    return hashlib.sha256("\n".join(sorted(p.id for p in problems)).encode()).hexdigest()


def _entry(path: Path, problems: list[Problem], sha: str | None = None) -> dict[str, Any]:
    return {
        "file": project_relative(path),
        "n": len(problems),
        "sha256": sha or sha256_file(path),
        "ids_sha256": ids_sha256(problems),
    }


def write_manifest(
    splits: Splits,
    out_dir: Path,
    manifest: Path,
    cfg: DataCfg,
    pool_file: Path,
    eval_files: dict[str, Path] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write each split to parquet and ``manifest.json`` with counts and hashes."""
    entries = {}
    for name, ps in splits.as_dict().items():
        path = out_dir / f"{name}.parquet"
        entries[name] = _entry(path, ps, write_parquet(ps, path))
    evals = {}
    for suite, f in sorted((eval_files or {}).items()):
        evals[suite] = _entry(f, read_parquet(f))
    data = {
        "version": MANIFEST_VERSION,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "seed": cfg.seed,
        "pool": {"file": project_relative(pool_file), "sha256": sha256_file(pool_file)},
        "splits": entries,
        "reserve": splits.n_reserve,
        "eval": evals,
        **(extra or {}),
    }
    io.write_json(manifest, data)
    return data


def verify_manifest(manifest: Path, root: Path | None = None) -> dict[str, Any]:
    """Raise :class:`ManifestMismatchError` if any listed file is missing or changed."""
    from qopd.config import PROJECT_ROOT  # noqa: PLC0415

    base = root or PROJECT_ROOT
    data = io.read_json(manifest)
    entries = [("pool", data["pool"])]
    entries += [(f"split {k}", v) for k, v in data["splits"].items()]
    entries += [(f"eval {k}", v) for k, v in data.get("eval", {}).items()]
    for label, e in entries:
        path = Path(e["file"])
        path = path if path.is_absolute() else base / path
        if not path.exists():
            raise ManifestMismatchError(f"{label}: {path} is missing")
        if sha256_file(path) != e["sha256"]:
            raise ManifestMismatchError(f"{label}: {path} sha256 differs from the manifest")
        if "ids_sha256" in e and ids_sha256(read_parquet(path)) != e["ids_sha256"]:
            raise ManifestMismatchError(f"{label}: ids differ from the manifest")
    return data


def build_splits(cfg: DataCfg, report: Path | None = None) -> dict[str, Any]:
    """PLAN 2.4 end to end: splits from pool_clean, manifest, report."""
    from qopd.data.evalsets import EVAL_SOURCES, eval_dir  # noqa: PLC0415
    from qopd.data.pool import data_dir  # noqa: PLC0415
    from qopd.data.report import REPORT, write_report  # noqa: PLC0415

    ddir = data_dir(cfg)
    pool_file = ddir / "pool" / "pool_clean.parquet"
    if not pool_file.exists():
        raise FileNotFoundError(f"{pool_file} missing; run `qopd data decontaminate` first")
    decon_stats = io.read_json(ddir / "contamination" / "contamination_stats.json")
    if decon_stats.get("sha256") != sha256_file(pool_file):
        raise ManifestMismatchError("pool_clean.parquet changed since decontamination; re-run it")
    splits = make_splits(
        read_parquet(pool_file), cfg.seed, cfg.n_train, cfg.n_coldstart, cfg.n_calibration,
        cfg.n_probe,
    )
    eval_files = {s: eval_dir(cfg) / f"{s}.parquet" for s in EVAL_SOURCES}
    built = {s: f for s, f in eval_files.items() if f.exists()}
    data = write_manifest(
        splits, ddir / "splits", ddir / "manifest.json", cfg, pool_file, built,
        extra={
            "eval_not_built": sorted(set(eval_files) - set(built)),
            "decontaminated_against": decon_stats["suites_checked"],
        },
    )
    verify_manifest(ddir / "manifest.json")
    write_report(ddir, report or REPORT)
    return data


def report_section(m: dict[str, Any]) -> list[str]:
    lines = [
        "## Splits and manifest (step 2.4)",
        "",
        f"From `{m['pool']['file']}` (sha256 `{m['pool']['sha256'][:16]}`), seed {m['seed']}: "
        "ordered by sha256(seed:id), then cut in order train → calibration → probe; cold-start is "
        "the first part of train. Calibration and probe are disjoint from train and from each "
        f"other. Unused (reserve): {m['reserve']}.",
        "",
        "| Split | Problems | File | sha256 |",
        "| --- | ---: | --- | --- |",
    ]
    for name, e in m["splits"].items():
        lines.append(f"| {name} | {e['n']} | `{e['file']}` | `{e['sha256'][:16]}` |")
    lines += ["", "Evaluation files recorded in the manifest:", "",
              "| Suite | Items | sha256 |", "| --- | ---: | --- |"]
    for name, e in m.get("eval", {}).items():
        lines.append(f"| {name} | {e['n']} | `{e['sha256'][:16]}` |")
    if m.get("eval_not_built"):
        lines += ["", f"**Not built (TBD):** {', '.join(m['eval_not_built'])} — the pool was "
                  f"decontaminated against {', '.join(m['decontaminated_against'])} only. After "
                  "building a missing suite, re-run `qopd data decontaminate` and "
                  "`qopd data splits`."]
    lines += ["", "`data/manifest.json` (tracked in git) holds counts, file sha256 and a sha256 "
              "over sorted ids for every file; `qopd data verify` checks them.", ""]
    return lines
