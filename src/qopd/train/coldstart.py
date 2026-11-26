"""Cold-start traces (PLAN 5.1): T16 answers each cold-start prompt once; keep verified-correct.

Generated once and reused by every condition. Thinking mode, temperature 0.6, top-p 0.95 (the doc
does not specify sampling; same as the calibration traces), and a per-request budget
``max_tokens = max_total - len(prompt)`` with ``max_total`` = the SFT limit (8,192): a trace
that reaches the limit has no final answer and is dropped (doc: longer traces are dropped, not
truncated). Exact token ids are stored, so SFT never re-tokenizes.

Resumable: prompts are processed in chunks; every verdict is appended to ``all.jsonl`` and a
restart skips problems already answered. ``finalize`` writes ``traces.jsonl`` (correct only, the
SFT input) and ``coldstart_stats.json``.
"""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any

from qopd.config import PROJECT_ROOT, project_relative
from qopd.utils import io
from qopd.utils.logging import get_logger

log = get_logger("train.coldstart")
OUT_DIR = PROJECT_ROOT / "data" / "coldstart"


def done_ids(path: Path) -> set[str]:
    return {r["id"] for r in io.iter_jsonl(path)} if path.exists() else set()


def finalize(out_dir: Path = OUT_DIR, max_total: int = 8192) -> dict[str, Any]:
    """Write traces.jsonl (verified-correct, within the limit) and coldstart_stats.json."""
    rows = list(io.iter_jsonl(out_dir / "all.jsonl"))
    kept = [r for r in rows if r["correct"] and r["finish_reason"] == "stop"]
    traces = out_dir / "traces.jsonl"
    io.write_jsonl(
        traces,
        (
            {"id": r["id"], "prompt_ids": r["prompt_ids"], "response_ids": r["response_ids"]}
            for r in kept
        ),
    )
    lens = [len(r["prompt_ids"]) + len(r["response_ids"]) for r in kept]
    reasons: dict[str, int] = {}
    for r in rows:
        key = (
            "kept"
            if r["correct"] and r["finish_reason"] == "stop"
            else ("reached_limit" if r["finish_reason"] == "length" else f"dropped:{r['reason']}")
        )
        reasons[key] = reasons.get(key, 0) + 1
    stats = {
        "step": "5.1",
        "n_prompts": len(rows),
        "n_kept": len(kept),
        "kept_rate": len(kept) / len(rows) if rows else float("nan"),
        "outcomes": dict(sorted(reasons.items())),
        "n_correct": sum(bool(r["correct"]) for r in rows),
        "n_finished": sum(r["finish_reason"] == "stop" for r in rows),
        "max_total_tokens": max_total,
        "kept_response_tokens": sum(len(r["response_ids"]) for r in kept),
        "kept_tokens_mean": statistics.fmean(lens) if lens else float("nan"),
        "kept_tokens_median": statistics.median(lens) if lens else float("nan"),
        "kept_tokens_max": max(lens) if lens else 0,
        "file": project_relative(traces),
        "sha256": io.sha256_file(traces),
    }
    io.write_json(out_dir / "coldstart_stats.json", stats)
    return stats


def generate_coldstart(
    teacher_id: str = "T16",
    max_total: int = 8192,
    chunk_size: int = 500,
    kv_cache_gb: float = 40.0,
    seed: int = 0,
    out_dir: Path = OUT_DIR,
    limit: int | None = None,
) -> dict[str, Any]:
    """Generate (or resume) the cold-start traces; returns the final stats."""
    from qopd.config import CONFIG_ROOT, EvalCfg, load  # noqa: PLC0415
    from qopd.data.sources import read_parquet  # noqa: PLC0415
    from qopd.eval.verifiers import verify  # noqa: PLC0415
    from qopd.models.loading import load_tokenizer  # noqa: PLC0415
    from qopd.models.registry import resolve  # noqa: PLC0415
    from qopd.prompts.templates import prompt_ids  # noqa: PLC0415
    from qopd.rollout.engine import RolloutEngine  # noqa: PLC0415
    from qopd.train.opd_loop import request_seed  # noqa: PLC0415

    out_dir.mkdir(parents=True, exist_ok=True)
    all_path = out_dir / "all.jsonl"
    cfg = resolve(teacher_id)
    tok = load_tokenizer(cfg)
    problems = read_parquet(PROJECT_ROOT / "data" / "splits" / "coldstart.parquet")[:limit]
    done = done_ids(all_path)
    todo = [p for p in problems if p.id not in done]
    log.info("cold start: %d prompts, %d done, %d to go", len(problems), len(done), len(todo))
    if todo:
        sampling = load(CONFIG_ROOT / "eval" / "default.yaml", cls=EvalCfg).sampling
        engine = RolloutEngine(
            cfg.path,
            cfg.revision,
            kv_cache_gb=kv_cache_gb,
            max_model_len=max_total + 16,
            seed=seed,
            enable_prefix_caching=True,
        )
        engine.start()
        try:
            for start in range(0, len(todo), chunk_size):
                chunk = todo[start : start + chunk_size]
                prompts = [prompt_ids(tok, p) for p in chunk]
                budgets = [max_total - len(x) for x in prompts]
                rollouts, stats = engine.generate(
                    prompts,
                    sampling.model_copy(update={"max_tokens": max(budgets)}),
                    n=1,
                    problem_ids=[p.id for p in chunk],
                    seeds=[request_seed(seed, -3, p.id) for p in chunk],
                    max_tokens=budgets,
                )
                by_id = {p.id: p for p in chunk}
                for r in rollouts:
                    v = verify(
                        by_id[r.problem_id], tok.decode(r.response_ids, skip_special_tokens=True)
                    )
                    io.append_jsonl(
                        all_path,
                        {
                            "id": r.problem_id,
                            "correct": v.correct,
                            "reason": v.reason,
                            "finish_reason": r.finish_reason.value,
                            "n_tokens": len(r.response_ids),
                            "prompt_ids": r.prompt_ids,
                            "response_ids": r.response_ids,
                        },
                    )
                log.info(
                    "cold start chunk %d: %d seqs in %.0fs (%.0f tok/s)",
                    start // chunk_size + 1,
                    stats.n_sequences,
                    stats.seconds,
                    stats.tokens_per_s,
                )
        finally:
            engine.stop()
    return finalize(out_dir, max_total)
