"""Expand a grid YAML (configs/grids/<name>.yaml) into run specs (PLAN 8.1).

students × teachers × seeds per block -> one :class:`RunSpec` each, with a deterministic run id
``<kind>__<student>__<teacher|none>__s<seed>``. Each run's fully resolved
:class:`~qopd.config.ExperimentCfg` is written to ``runs/<experiment>/specs/<run_id>.yaml`` and
the ordered queue to ``runs/<experiment>/queue.json`` (read by the host-side queue).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from qopd.config import (
    CONFIG_ROOT,
    ExperimentCfg,
    GridCfg,
    dump,
    load,
    project_relative,
)
from qopd.models.registry import resolve
from qopd.orchestration.queue import RunSpec, run_id, write_queue
from qopd.utils.io import sha256_file
from qopd.utils.run import default_runs_root, git_sha

__all__ = ["RunSpec", "expand", "plan", "run_id"]


def expand(grid: GridCfg, specs_dir: Path) -> list[RunSpec]:
    """Build the run list (writes one resolved spec file per run into ``specs_dir``)."""
    specs: list[RunSpec] = []
    seen: set[str] = set()
    for block in grid.blocks:
        for student in block.students:
            resolve(student)  # unknown ids fail here, not hours later
            for teacher in block.teachers:
                if teacher is not None:
                    resolve(teacher)
                for seed in block.seeds:
                    rid = run_id(block.kind.value, student, teacher, seed)
                    if rid in seen:
                        raise ValueError(f"duplicate run {rid} in grid {grid.name}")
                    seen.add(rid)
                    overrides = [
                        f"student={student}",
                        f"seeds=[{seed}]",
                        f"name={grid.name}",
                        *block.overrides,
                    ]
                    overrides.append(f"teacher={teacher}" if teacher else "teacher=null")
                    cfg = load(CONFIG_ROOT / grid.base, overrides, cls=ExperimentCfg)
                    spec_path = specs_dir / f"{rid}.yaml"
                    dump(cfg, spec_path)
                    key_specific = f"{block.kind.value}:{student}"
                    key = key_specific if key_specific in grid.estimate_hours else block.kind.value
                    specs.append(
                        RunSpec(
                            experiment=grid.name,
                            run_id=rid,
                            kind=block.kind.value,
                            student=student,
                            teacher=teacher,
                            seed=seed,
                            command=grid.commands[block.kind].format(
                                spec=project_relative(spec_path)
                            ),
                            spec_file=project_relative(spec_path),
                            optional=block.optional,
                            estimate_key=key,
                            estimate_hours=grid.estimate_hours.get(key),
                            note=block.note,
                            overrides=overrides,
                        )
                    )
    return specs


def plan(grid_name: str, runs_root: Path | None = None) -> tuple[Path, list[RunSpec]]:
    """Expand ``configs/grids/<grid_name>.yaml`` and write specs + queue.json."""
    grid_file = CONFIG_ROOT / "grids" / f"{grid_name}.yaml"
    grid = load(grid_file, cls=GridCfg)
    root = runs_root or default_runs_root()
    specs = expand(grid, root / grid.name / "specs")
    meta = {
        "grid_file": project_relative(grid_file),
        "grid_sha256": sha256_file(grid_file),
        "git_sha": git_sha(),
        "planned": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    return write_queue(specs, root, grid.name, meta), specs
