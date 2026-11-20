"""Dataset download and conversion to the unified schema (PLAN 2.1, 2.2).

Unified schema: ``{id, problem, answer, source, kind}`` (+ ``meta`` as JSON). Every source
is pinned to a Hub commit. If a dataset is not found under its expected name, stop and
ask (CLAUDE.md rule 8); never substitute.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qopd.types import Problem, ProblemKind
from qopd.utils.io import sha256_file
from qopd.utils.logging import get_logger

log = get_logger("data.sources")


@dataclass(frozen=True)
class SourceSpec:
    name: str  # short name used in Problem.source and ids
    hf_repo: str
    revision: str  # pinned dataset commit
    filename: str  # file inside the dataset repo
    problem_field: str
    answer_field: str
    license: str
    kind: ProblemKind = ProblemKind.MATH
    gated: bool = False
    notes: str = ""


SOURCES: dict[str, SourceSpec] = {
    "deepscaler_preview": SourceSpec(
        name="deepscaler_preview",
        hf_repo="agentica-org/DeepScaleR-Preview-Dataset",
        revision="b6ae8c60f5c1f2b594e2140b91c49c9ad0949e29",
        filename="deepscaler.json",
        problem_field="problem",
        answer_field="answer",
        license="mit",
        notes="AIME 1984-2023, AMC before 2023, Omni-MATH, STILL (per dataset README)",
    ),
}


class DatasetNotFoundError(RuntimeError):
    """Raised instead of silently substituting a dataset."""


def get_source(name: str) -> SourceSpec:
    try:
        return SOURCES[name]
    except KeyError as e:
        raise DatasetNotFoundError(f"unknown source {name!r}; known: {sorted(SOURCES)}") from e


def download(spec: SourceSpec) -> Path:
    """Download ``spec.filename`` at the pinned revision into the HF cache."""
    from huggingface_hub import hf_hub_download  # noqa: PLC0415
    from huggingface_hub.utils import HfHubHTTPError  # noqa: PLC0415

    try:
        path = hf_hub_download(
            spec.hf_repo, spec.filename, repo_type="dataset", revision=spec.revision
        )
    except HfHubHTTPError as e:
        raise DatasetNotFoundError(
            f"{spec.hf_repo}@{spec.revision} / {spec.filename}: {e}. Options: (1) check the "
            "repo id and revision, (2) log in if it is gated, (3) ask the user."
        ) from e
    return Path(path)


def load_rows(spec: SourceSpec, path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".json":
        rows = json.loads(path.read_text())
    elif path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    elif path.suffix == ".parquet":
        import pyarrow.parquet as pq  # noqa: PLC0415

        rows = pq.read_table(path).to_pylist()
    else:
        raise ValueError(f"unsupported file type {path.suffix}")
    if not isinstance(rows, list):
        raise ValueError(f"{path}: expected a list of records")
    return rows


def to_problems(spec: SourceSpec, rows: Iterable[dict[str, Any]]) -> list[Problem]:
    """Map raw rows to :class:`Problem`; ids are ``<source>/<row index>`` (stable per revision)."""
    out = []
    for i, row in enumerate(rows):
        out.append(
            Problem(
                id=f"{spec.name}/{i:06d}",
                problem=str(row.get(spec.problem_field) or ""),
                answer=str(row.get(spec.answer_field) or ""),
                source=spec.name,
                kind=spec.kind,
                meta={"row": i},
            )
        )
    return out


def fetch(name: str) -> tuple[list[Problem], dict[str, Any]]:
    """Download + convert a source; returns problems and a provenance record."""
    spec = get_source(name)
    path = download(spec)
    rows = load_rows(spec, path)
    info = {
        "name": spec.name,
        "hf_repo": spec.hf_repo,
        "revision": spec.revision,
        "file": spec.filename,
        "file_sha256": sha256_file(path),
        "license": spec.license,
        "rows": len(rows),
        "notes": spec.notes,
    }
    log.info("%s: %d rows from %s@%s", name, len(rows), spec.hf_repo, spec.revision[:8])
    return to_problems(spec, rows), info


# --- parquet I/O -------------------------------------------------------------------

_COLUMNS = ("id", "problem", "answer", "source", "kind", "meta")


def write_parquet(problems: list[Problem], path: Path) -> str:
    """Write problems to parquet (meta as JSON text); return the file's sha256."""
    import pyarrow as pa  # noqa: PLC0415
    import pyarrow.parquet as pq  # noqa: PLC0415

    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "id": [p.id for p in problems],
            "problem": [p.problem for p in problems],
            "answer": [p.answer for p in problems],
            "source": [p.source for p in problems],
            "kind": [p.kind.value for p in problems],
            "meta": [json.dumps(p.meta, sort_keys=True) for p in problems],
        }
    )
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(table, tmp)
    tmp.replace(path)
    return sha256_file(path)


def read_parquet(path: Path) -> list[Problem]:
    import pyarrow.parquet as pq  # noqa: PLC0415

    rows = pq.read_table(path, columns=list(_COLUMNS)).to_pylist()
    return [
        Problem(
            id=r["id"],
            problem=r["problem"],
            answer=r["answer"],
            source=r["source"],
            kind=ProblemKind(r["kind"]),
            meta=json.loads(r["meta"] or "{}"),
        )
        for r in rows
    ]
