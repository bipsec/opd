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

from qopd.config import PROJECT_ROOT, DataCfg, project_relative
from qopd.data import filters
from qopd.data.sources import fetch, read_parquet, write_parquet
from qopd.utils import io
from qopd.utils.logging import get_logger

log = get_logger("data.pool")


def data_dir(cfg: DataCfg) -> Path:
    d = Path(cfg.data_dir)
    return d if d.is_absolute() else PROJECT_ROOT / d


def build_pool(cfg: DataCfg, report: Path | None = None) -> dict[str, Any]:
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
            "file": project_relative(out / "pool.parquet"),
            "sha256": sha,
            "n": len(current),
            "by_source": dict(collections.Counter(p.source for p in current)),
            "by_answer_type": dict(collections.Counter(p.meta["answer_type"] for p in current)),
        },
        "seconds": round(time.perf_counter() - t0, 1),
    }
    io.write_json(out / "pool_stats.json", stats)
    from qopd.data.report import REPORT, write_report  # noqa: PLC0415

    write_report(data_dir(cfg), report or REPORT)
    return stats


def load_pool(cfg: DataCfg) -> list:
    return read_parquet(data_dir(cfg) / "pool" / "pool.parquet")


# --- report ------------------------------------------------------------------------------


def report_section(stats: dict[str, Any]) -> list[str]:
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


# --- contamination (PLAN 2.3) ------------------------------------------------------------

# Diagnostic only (everything with >= 1 shared n-gram is dropped): a pair is a "near copy"
# when it shares at least this share of the eval item's n-grams.
NEAR_COPY_SHARE = 0.5


def decontaminate(cfg: DataCfg, report: Path | None = None) -> dict[str, Any]:
    """Drop pool prompts sharing a 13-gram with any built eval suite -> pool_clean.parquet."""
    from qopd.data.contamination import build_index, drop_contaminated, find_contamination  # noqa: PLC0415
    from qopd.data.evalsets import EVAL_SOURCES, eval_dir  # noqa: PLC0415

    ddir = data_dir(cfg)
    out = ddir / "contamination"
    pool = load_pool(cfg)
    evals, checked, missing = [], [], []
    for suite in EVAL_SOURCES:
        f = eval_dir(cfg) / f"{suite}.parquet"
        if f.exists():
            evals += read_parquet(f)
            checked.append(suite)
        else:
            missing.append(suite)
    t0 = time.perf_counter()
    index = build_index(evals, cfg.contamination_ngram)
    hits = find_contamination(pool, index)
    clean, _ = drop_contaminated(pool, hits)

    by_train: dict[str, list] = collections.defaultdict(list)
    for h in hits:
        by_train[h.train_id].append(h)
    eval_source = {p.id: p.source for p in evals}
    dropped_rows = [
        {
            "id": tid,
            "reason": f"shares a {cfg.contamination_ngram}-gram with an evaluation item",
            "eval_ids": [h.eval_id for h in hs],
            "suites": sorted({eval_source[h.eval_id] for h in hs}),
            "max_shared_ngrams": max(h.n_shared for h in hs),
            "example_ngram": hs[0].ngram,
        }
        for tid, hs in by_train.items()
    ]
    io.write_jsonl(out / "dropped.jsonl", dropped_rows)
    io.write_jsonl(out / "hits.jsonl", (h.__dict__ for h in hits))
    sha = write_parquet(clean, ddir / "pool" / "pool_clean.parquet")

    from qopd.data.contamination import ngrams  # noqa: PLC0415

    eval_ngrams = {p.id: len(ngrams(p.problem, cfg.contamination_ngram)) for p in evals}
    near_copy = [h for h in hits if h.n_shared >= NEAR_COPY_SHARE * max(eval_ngrams[h.eval_id], 1)]
    from qopd.data.contamination import tokens  # noqa: PLC0415

    short = {tuple(tokens(p.problem)): p for p in evals if eval_ngrams[p.id] == 0}
    short_exact = [(t.id, short[k].id) for t in pool if (k := tuple(tokens(t.problem))) in short]
    per_suite: dict[str, dict[str, int]] = {}
    for s in checked:
        hs = [h for h in hits if eval_source[h.eval_id] == s]
        nc = [h for h in near_copy if eval_source[h.eval_id] == s]
        per_suite[s] = {
            "eval_items": index.n_items.get(s, 0),
            "too_short": index.too_short.get(s, 0),
            "eval_items_hit": len({h.eval_id for h in hs}),
            "train_dropped": len({h.train_id for h in hs}),
            "near_copy_eval_items": len({h.eval_id for h in nc}),
            "near_copy_train": len({h.train_id for h in nc}),
            "short_exact_train": sum(1 for _, e in short_exact if eval_source[e] == s),
        }
    stats = {
        "step": "2.3",
        "ngram": cfg.contamination_ngram,
        "tokenization": "lower-cased [a-z0-9]+ word tokens",
        "suites_checked": checked,
        "suites_not_checked": missing,
        "pool_in": len(pool),
        "pool_out": len(clean),
        "dropped": len(dropped_rows),
        "pairs": len(hits),
        "near_copy_share": NEAR_COPY_SHARE,
        "short_exact_matches": [{"train_id": t, "eval_id": e} for t, e in short_exact],
        "per_suite": per_suite,
        "output": project_relative(ddir / "pool" / "pool_clean.parquet"),
        "sha256": sha,
        "seconds": round(time.perf_counter() - t0, 1),
    }
    io.write_json(out / "contamination_stats.json", stats)
    from qopd.data.report import REPORT, write_report  # noqa: PLC0415

    write_report(ddir, report or REPORT)
    return stats


def contamination_report_section(stats: dict[str, Any]) -> list[str]:
    lines = [
        "## Contamination filter (step 2.3)",
        "",
        f"Training prompts sharing any {stats['ngram']}-gram ({stats['tokenization']}) with an "
        "evaluation item are dropped. Pool: "
        f"{stats['pool_in']} → **{stats['pool_out']}** (dropped {stats['dropped']}; "
        f"{stats['pairs']} train–eval pairs) in `{stats['output']}` "
        f"(sha256 `{stats['sha256'][:16]}`).",
        "",
        "| Suite | Eval items | Too short to match | Eval items hit | Training prompts dropped "
        "| Near copies: eval items / training prompts |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for s, r in stats["per_suite"].items():
        lines.append(f"| {s} | {r['eval_items']} | {r['too_short']} | {r['eval_items_hit']} | "
                     f"{r['train_dropped']} | {r['near_copy_eval_items']} / {r['near_copy_train']} |")
    n_short = len(stats.get("short_exact_matches", []))
    by_suite = [
        f"{k}: {v['short_exact_train']}"
        for k, v in stats["per_suite"].items()
        if v.get("short_exact_train")
    ]
    lines += [
        "",
        f"Diagnostic, not dropped: {n_short} training prompts exactly equal (same tokens) an "
        f"evaluation item too short for a 13-gram ({', '.join(by_suite) or 'none'}).",
    ]
    if stats["suites_not_checked"]:
        lines += ["", f"**Not checked (suite not built):** {', '.join(stats['suites_not_checked'])}. "
                  "Re-run `qopd data decontaminate` after building them."]
    lines += ["", f"Near copy: the pair shares >= {stats['near_copy_share']:.0%} of the evaluation "
              "item's n-grams (likely the same problem). Other hits are mostly shared contest "
              "boilerplate (e.g. \"... where m and n are relatively prime positive integers. "
              "Find m+n\"); per the doc they are dropped too. "
              "A prompt matching several suites counts once per suite. Every drop, with its "
              "evaluation ids and an example n-gram, is in `data/contamination/dropped.jsonl`.", ""]
    return lines
