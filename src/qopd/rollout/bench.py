"""Rollout throughput benchmark (PLAN 1.4): N prompts × n samples at training settings.

Writes ``runs/rollout_bench/<run_id>/`` with summary.json (throughput, lengths, finish
reasons, memory), rollouts.jsonl and monitor.csv.
"""

from __future__ import annotations

import statistics
import time
from pathlib import Path
from typing import Any

from qopd.utils.logging import get_logger

log = get_logger("rollout.bench")


def validate_rollouts(rollouts: list[Any], prompts: list[list[int]], vocab: int) -> dict[str, int]:
    """Counts of problems (all zero = valid): ids out of range, logprob mismatches, bad EOS."""
    bad_ids = sum(1 for r in rollouts for t in r.response_ids if not 0 <= t < vocab)
    bad_lp = sum(1 for r in rollouts if len(r.sampled_logprobs) != len(r.response_ids))
    bad_lp += sum(1 for r in rollouts for x in r.sampled_logprobs if not x <= 0.0)
    prompt_set = {tuple(p) for p in prompts}
    bad_prompt = sum(1 for r in rollouts if tuple(r.prompt_ids) not in prompt_set)
    empty = sum(1 for r in rollouts if not r.response_ids)
    return {
        "ids_out_of_tokenizer_range": bad_ids,
        "logprob_problems": bad_lp,
        "prompt_mismatch": bad_prompt,
        "empty_responses": empty,
    }


def run_bench(
    model_id: str = "S2",
    n_prompts: int = 8,
    n: int = 2,
    max_tokens: int | None = None,
    seed: int = 0,
    kv_cache_gb: float = 6.0,
    root: Path | None = None,
) -> dict[str, Any]:
    from qopd.config import CONFIG_ROOT, OPDCfg, load  # noqa: PLC0415
    from qopd.data.smoke import smoke_problems  # noqa: PLC0415
    from qopd.models.loading import load_tokenizer  # noqa: PLC0415
    from qopd.models.registry import resolve  # noqa: PLC0415
    from qopd.prompts.templates import prompt_ids  # noqa: PLC0415
    from qopd.rollout.engine import RolloutEngine  # noqa: PLC0415
    from qopd.utils import io  # noqa: PLC0415
    from qopd.utils.logging import setup_logging  # noqa: PLC0415
    from qopd.utils.monitor import Monitor  # noqa: PLC0415
    from qopd.utils.run import create_run_dir  # noqa: PLC0415

    cfg = resolve(model_id)
    opd = load(CONFIG_ROOT / "train" / "opd.yaml", cls=OPDCfg)
    sampling = opd.rollout.model_copy(
        update={"max_tokens": max_tokens or opd.rollout.max_tokens, "seed": seed}
    )
    run_cfg = {
        "task": "rollout_bench",
        "model": cfg.model_dump(mode="json"),
        "n_prompts": n_prompts,
        "n": n,
        "sampling": sampling.model_dump(mode="json"),
        "kv_cache_gb": kv_cache_gb,
    }
    run_id = f"{model_id}_p{n_prompts}_n{n}_t{sampling.max_tokens}_s{seed}"
    run = create_run_dir("rollout_bench", run_id, run_cfg, root=root)
    if run.is_done:
        return io.read_json(run.path / "summary.json")
    setup_logging(run.stdout)
    tok = load_tokenizer(cfg)
    problems = smoke_problems(n_prompts)
    prompts = [prompt_ids(tok, p) for p in problems]

    with Monitor(run.monitor):
        engine = RolloutEngine(
            cfg.path,
            cfg.revision,
            kv_cache_gb=kv_cache_gb,
            max_model_len=max(len(p) for p in prompts) + sampling.max_tokens + 16,
            seed=seed,
        )
        engine.start()
        rollouts, stats = engine.generate(
            prompts, sampling, n=n, problem_ids=[p.id for p in problems]
        )
        t0 = time.perf_counter()
        engine.stop()
        stop_s = time.perf_counter() - t0

    lengths = [len(r.response_ids) for r in rollouts]
    eos_ok = all(r.response_ids[-1] in engine.stop_ids for r in rollouts if not r.truncated)
    io.write_jsonl(run.path / "rollouts.jsonl", (r.to_dict() for r in rollouts))
    summary = {
        "model": model_id,
        "n_sequences": stats.n_sequences,
        "n_tokens": stats.n_tokens,
        "generate_s": stats.seconds,
        "tokens_per_s": stats.tokens_per_s,
        "load_s": engine.load_seconds,
        "stop_s": stop_s,
        "response_len_mean": statistics.fmean(lengths),
        "response_len_min": min(lengths),
        "response_len_max": max(lengths),
        "truncated": sum(r.truncated for r in rollouts),
        "finish_reasons": stats.finish_reasons,
        "stopped_on_eos_token": eos_ok,
        "peak_mem_used_gb_system": stats.peak_mem_gb,
        "min_mem_available_gb_system": stats.min_available_gb,
        "kv_cache_gb": kv_cache_gb,
        **validate_rollouts(rollouts, prompts, len(tok)),
    }
    summary["valid"] = (
        summary["ids_out_of_tokenizer_range"] == 0
        and summary["logprob_problems"] == 0
        and summary["prompt_mismatch"] == 0
        and summary["empty_responses"] == 0
        and eos_ok
    )
    run.log_metrics(summary)
    io.write_json(run.path / "summary.json", summary)
    run.mark_done()
    return summary
