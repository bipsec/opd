"""Training prompt pool (PLAN 2.1): fetch the source, run the filter stages, write outputs.

Outputs under ``<data_dir>/pool/``: ``pool.parquet``, ``pool_stats.json`` (provenance and
per-stage counts) and ``dropped.jsonl`` (every dropped id with its stage and reason).
``reports/data.md`` is rendered from the stats files (CLAUDE.md rule 6).
"""

from __future__ import annotations

import collections
import time
from pathlib import Path
from typing import Any

from qopd.config import PROJECT_ROOT, DataCfg
from qopd.data import filters
from qopd.data.sources import fetch, read_parquet, write_parquet
from qopd.utils import io
from qopd.utils.logging import get_logger

log = get_logger("data.pool")
REPORT = PROJECT_ROOT / "reports" / "data.md"


def data_dir(cfg: DataCfg) -> Path:
    d = Path(cfg.data_dir)
    return d if d.is_absolute() else PROJECT_ROOT / d


def build_pool(cfg: DataCfg) -> dict[str, Any]:
    """Run 2.1 end to end; returns the stats written to ``pool_stats.json``."""
    out = data_dir(cfg) / "pool"
    t0 = time.perf_counter()
    problems, provenance = fetch(cfg.train_source)
    stages = [
        lambda ps: filters.normalize(ps),
        lambda ps: filters.keep_parsable_answers(ps, cfg.keep_answer_types),
        lambda ps: filters.self_check(ps),
        lambda ps: filters.dedup_exact(ps, cfg.drop_conflicting_duplicates),
        lambda ps: filters.dedup_near(
            ps, cfg.near_dup_threshold, cfg.near_dup_num_perm, cfg.near_dup_ngram
        ),
    ]
    results = []
    dropped_rows = []
    current = problems
    for stage in stages:
        res = stage(current)
        log.info("%-12s in=%d kept=%d dropped=%s", res.stage, res.n_in, len(res.kept), res.dropped)
        results.append(res.summary())
        dropped_rows += [{"id": i, "stage": res.stage, "reason": r} for i, r in res.dropped_ids]
        current = res.kept

    sha = write_parquet(current, out / "pool.parquet")
    io.write_jsonl(out / "dropped.jsonl", dropped_rows)
    stats = {
        "step": "2.1",
        "source": provenance,
        "config": {
            "keep_answer_types": list(cfg.keep_answer_types),
            "drop_conflicting_duplicates": cfg.drop_conflicting_duplicates,
            "near_dup": {
                "ngram": cfg.near_dup_ngram,
                "num_perm": cfg.near_dup_num_perm,
                "threshold": cfg.near_dup_threshold,
            },
        },
        "stages": results,
        "pool": {
            "file": str((out / "pool.parquet").relative_to(PROJECT_ROOT)),
            "sha256": sha,
            "n": len(current),
            "by_source": dict(collections.Counter(p.source for p in current)),
            "by_answer_type": dict(collections.Counter(p.meta["answer_type"] for p in current)),
        },
        "seconds": round(time.perf_counter() - t0, 1),
    }
    io.write_json(out / "pool_stats.json", stats)
    write_report(data_dir(cfg))
    return stats


def load_pool(cfg: DataCfg) -> list:
    return read_parquet(data_dir(cfg) / "pool" / "pool.parquet")


# --- report ------------------------------------------------------------------------------


def _pool_section(stats: dict[str, Any]) -> list[str]:
    src, pool = stats["source"], stats["pool"]
    lines = [
        "## Training pool (step 2.1)",
        "",
        "Source: [`{repo}`](https://huggingface.co/datasets/{repo}) @ `{rev}`, file `{file}` "
        "(sha256 `{sha}`), license `{lic}`. {notes}.".format(
            repo=src["hf_repo"],
            rev=src["revision"],
            file=src["file"],
            sha=src["file_sha256"][:16],
            lic=src["license"],
            notes=src["notes"],
        ),
        "",
        f"Raw rows: {src['rows']}.",
        "",
        "| Stage | In | Kept | Dropped (by reason) |",
        "| --- | ---: | ---: | --- |",
    ]
    for st in stats["stages"]:
        reasons = ", ".join(f"{k}: {v}" for k, v in st["dropped"].items()) or "—"
        lines.append(f"| {st['stage']} | {st['in']} | {st['kept']} | {reasons} |")
    lines += [
        "",
        f"Pool: **{pool['n']}** problems in `{pool['file']}` (sha256 `{pool['sha256'][:16]}`).",
        "",
        "| Answer type | Count |",
        "| --- | ---: |",
    ]
    lines += [f"| {k} | {v} |" for k, v in sorted(pool["by_answer_type"].items())]
    lines += ["", "| Source | Count |", "| --- | ---: |"]
    lines += [f"| {k} | {v} |" for k, v in sorted(pool["by_source"].items())]
    cfg = stats["config"]
    lines += [
        "",
        f"Kept answer types: {', '.join(cfg['keep_answer_types'])} (math-verify parses the "
        "answer to one real number with no free variables). Near-duplicates: MinHash on word "
        f"{cfg['near_dup']['ngram']}-grams, {cfg['near_dup']['num_perm']} permutations, "
        f"Jaccard >= {cfg['near_dup']['threshold']} (doc setting; it can also merge distinct "
        "variants that differ in one symbol). Identical problems whose gold answers disagree "
        f"are all dropped: {cfg.get('drop_conflicting_duplicates', False)}. Every dropped id "
        "and reason is in `data/pool/dropped.jsonl`.",
        "",
    ]
    return lines


def write_report(ddir: Path, report: Path = REPORT) -> Path:
    """Render ``reports/data.md`` from whichever stats files exist."""
    lines = [
        "# Data report",
        "",
        "Generated from `data/**/..._stats.json` by `qopd data report`; do not edit by hand.",
        "",
    ]
    pool_stats = ddir / "pool" / "pool_stats.json"
    if pool_stats.exists():
        lines += _pool_section(io.read_json(pool_stats))
    else:
        lines += ["## Training pool (step 2.1)", "", "TBD (not built yet).", ""]
    lines += [
        "## Evaluation suites (step 2.2)",
        "",
        "TBD (step 2.2).",
        "",
        "## Contamination filter (step 2.3)",
        "",
        "TBD (step 2.3).",
        "",
        "## Splits and manifest (step 2.4)",
        "",
        "TBD (step 2.4).",
        "",
    ]
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines))
    return report
