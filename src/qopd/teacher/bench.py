"""Teacher load + scoring benchmark (PLAN 4.1 / 4.3): load time, memory, scoring throughput.

Sequences: the first ``n`` S1 MATH-500 sanity-eval responses (step 3.3), re-tokenized and cut
to ``max_response_tokens`` (the OPD rollout length). Re-tokenizing is fine for a *speed*
benchmark; it is never used for rewards. Results: ``runs/teacher_bench/<model>_n<n>/``.
"""

from __future__ import annotations

import statistics
import time
from pathlib import Path
from typing import Any

from qopd.utils.logging import get_logger

log = get_logger("teacher.bench")
SANITY_RUN = "S1__gsm8k+math500__n1__t16384__s0"


def benchmark_sequences(tok: Any, n: int, max_response_tokens: int) -> list[Any]:
    """Rollouts built from saved eval responses (prompt via the chat template)."""
    from qopd.config import PROJECT_ROOT  # noqa: PLC0415
    from qopd.data.sources import read_parquet  # noqa: PLC0415
    from qopd.prompts.templates import prompt_ids  # noqa: PLC0415
    from qopd.types import Rollout  # noqa: PLC0415
    from qopd.utils import io  # noqa: PLC0415

    problems = {p.id: p for p in read_parquet(PROJECT_ROOT / "data" / "eval" / "math500.parquet")}
    rows = [
        r
        for r in io.iter_jsonl(PROJECT_ROOT / "runs" / "eval" / SANITY_RUN / "samples.jsonl")
        if r["suite"] == "math500"
    ][:n]
    out = []
    for r in rows:
        resp = tok.encode(r["text"], add_special_tokens=False)[:max_response_tokens]
        out.append(
            Rollout(
                prompt_ids(tok, problems[r["problem_id"]]), resp, [], problem_id=r["problem_id"]
            )
        )
    return out


def run_teacher_bench(
    model_id: str = "T16",
    n: int = 200,
    max_response_tokens: int = 4096,
    top_k: int = 20,
    kv_cache_gb: float = 16.0,
    gpu_memory_utilization: float = 0.75,
    root: Path | None = None,
) -> dict[str, Any]:
    from qopd.models.loading import load_tokenizer  # noqa: PLC0415
    from qopd.models.registry import resolve  # noqa: PLC0415
    from qopd.teacher.vllm_backend import VLLMTeacher  # noqa: PLC0415
    from qopd.utils import io  # noqa: PLC0415
    from qopd.utils.logging import setup_logging  # noqa: PLC0415
    from qopd.utils.memory import PeakMemory, snapshot  # noqa: PLC0415
    from qopd.utils.monitor import Monitor  # noqa: PLC0415
    from qopd.utils.run import create_run_dir  # noqa: PLC0415

    cfg = resolve(model_id)
    run_cfg = {
        "task": "teacher_bench",
        "model": cfg.model_dump(mode="json"),
        "n": n,
        "max_response_tokens": max_response_tokens,
        "top_k": top_k,
        "kv_cache_gb": kv_cache_gb,
        "gpu_memory_utilization": gpu_memory_utilization,
    }
    run = create_run_dir("teacher_bench", f"{model_id}_n{n}", run_cfg, root=root)
    if run.is_done:
        return io.read_json(run.path / "summary.json")
    setup_logging(run.stdout)
    tok = load_tokenizer(cfg)
    seqs = benchmark_sequences(tok, n, max_response_tokens)
    max_len = max(len(r.input_ids) for r in seqs) + 16
    with Monitor(run.monitor):
        before = snapshot("before load")
        teacher = VLLMTeacher(
            cfg.path,
            cfg.revision,
            kv_cache_gb=kv_cache_gb,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_len,
            max_logprobs=max(20, top_k),
        )
        t0 = time.perf_counter()
        teacher.start()
        load_s = time.perf_counter() - t0
        loaded = snapshot("after load")
        with PeakMemory() as pm:
            t0 = time.perf_counter()
            scores = teacher.score(seqs, top_k)
            score_s = time.perf_counter() - t0
        teacher.stop()
    n_tok = sum(len(s) for s in scores)
    summary = {
        "model": model_id,
        "n_sequences": len(seqs),
        "response_tokens_scored": n_tok,
        "mean_response_tokens": statistics.fmean(len(r.response_ids) for r in seqs),
        "load_s": load_s,
        "score_s": score_s,
        "scored_tokens_per_s": n_tok / score_s if score_s > 0 else float("nan"),
        "mem_used_before_gb": before.mem_used_gb,
        "mem_used_after_load_gb": loaded.mem_used_gb,
        "load_delta_gb": loaded.mem_used_gb - before.mem_used_gb,
        "peak_mem_used_gb": pm.peak_used_gb,
        "min_mem_available_gb": pm.min_available_gb,
        "kv_cache_gb": kv_cache_gb,
    }
    io.write_json(run.path / "summary.json", summary)
    run.log_metrics(summary)
    run.mark_done()
    return summary
