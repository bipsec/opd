"""Backend agreement check (PLAN 1.3 "done when").

A small model (S2) stands in as the teacher: it samples responses to fixed prompts, then
the same sequences are scored by the vLLM and HF backends and compared token by token.
Results go to ``runs/agreement/<run_id>/`` (summary.json + per-sequence metrics.jsonl).
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from qopd.data.smoke import SMOKE_PROBLEMS as AGREEMENT_PROBLEMS
from qopd.types import FinishReason, Problem, Rollout, TeacherScores
from qopd.utils.logging import get_logger

log = get_logger("teacher.agreement")
THRESHOLD_NATS = 0.02  # PLAN 1.3


def agreement_metrics(
    ref: Sequence[TeacherScores], cand: Sequence[TeacherScores]
) -> dict[str, float]:
    """Token-level comparison of two backends' scores on the same rollouts."""
    diffs: list[float] = []
    top1 = overlap = n_topk = 0
    for a, b in zip(ref, cand, strict=True):
        if len(a) != len(b):
            raise ValueError("score lengths differ")
        diffs += [abs(x - y) for x, y in zip(a.token_logprobs, b.token_logprobs, strict=True)]
        for ia, ib in zip(a.topk_ids, b.topk_ids, strict=True):
            n_topk += 1
            top1 += int(ia[0] == ib[0])
            overlap += len(set(ia) & set(ib)) / max(len(ia), 1)
    if not diffs:
        raise ValueError("no tokens to compare")
    srt = sorted(diffs)
    return {
        "n_tokens": float(len(diffs)),
        "mean_abs_diff": statistics.fmean(diffs),
        "median_abs_diff": statistics.median(diffs),
        "p99_abs_diff": srt[min(len(srt) - 1, math.ceil(0.99 * len(srt)) - 1)],
        "max_abs_diff": srt[-1],
        "top1_agreement": top1 / n_topk if n_topk else float("nan"),
        "topk_overlap": overlap / n_topk if n_topk else float("nan"),
    }


def _generate(teacher: Any, prompts: list[list[int]], max_tokens: int, seed: int) -> list[Rollout]:
    """Sample one response per prompt with the vLLM engine (pure sampling, T=1)."""
    from vllm import SamplingParams  # noqa: PLC0415
    from vllm.inputs import TokensPrompt  # noqa: PLC0415

    params = SamplingParams(
        temperature=1.0, top_p=1.0, top_k=-1, max_tokens=max_tokens, logprobs=0, seed=seed
    )
    outs = teacher.llm.generate(
        [TokensPrompt(prompt_token_ids=p) for p in prompts], params, use_tqdm=False
    )
    rollouts = []
    for i, (p, out) in enumerate(zip(prompts, outs, strict=True)):
        c = out.outputs[0]
        ids = list(c.token_ids)
        lps = [float(d[t].logprob) for d, t in zip(c.logprobs, ids, strict=True)]
        reason = FinishReason.LENGTH if c.finish_reason == "length" else FinishReason.STOP
        rollouts.append(Rollout(list(p), ids, lps, reason, f"agree/{i}", 0))
    return rollouts


def run_agreement(
    model_id: str = "S2",
    n: int = 20,
    max_tokens: int = 512,
    top_k: int = 20,
    seed: int = 0,
    kv_cache_gb: float = 4.0,
    root: Path | None = None,
) -> dict[str, Any]:
    """Generate ``n`` rollouts, score with vLLM and HF, write and return the summary."""
    import time  # noqa: PLC0415

    from qopd.models.loading import load_tokenizer  # noqa: PLC0415
    from qopd.models.registry import resolve  # noqa: PLC0415
    from qopd.prompts.templates import prompt_ids  # noqa: PLC0415
    from qopd.teacher.hf_backend import HFTeacher  # noqa: PLC0415
    from qopd.teacher.vllm_backend import VLLMTeacher  # noqa: PLC0415
    from qopd.utils import io  # noqa: PLC0415
    from qopd.utils.logging import setup_logging  # noqa: PLC0415
    from qopd.utils.memory import snapshot  # noqa: PLC0415
    from qopd.utils.monitor import Monitor  # noqa: PLC0415
    from qopd.utils.run import create_run_dir  # noqa: PLC0415
    from qopd.utils.seed import seed_everything  # noqa: PLC0415

    if n > len(AGREEMENT_PROBLEMS):
        raise ValueError(f"n={n} > {len(AGREEMENT_PROBLEMS)} fixed problems")
    cfg = resolve(model_id)
    run_cfg = {
        "task": "backend_agreement",
        "model": cfg.model_dump(mode="json"),
        "n": n,
        "max_tokens": max_tokens,
        "top_k": top_k,
        "seed": seed,
        "kv_cache_gb": kv_cache_gb,
        "threshold_nats": THRESHOLD_NATS,
    }
    run = create_run_dir("agreement", f"{model_id}_vllm_vs_hf_s{seed}", run_cfg, root=root)
    if run.is_done:
        log.info("%s already DONE; returning stored summary", run.path)
        return io.read_json(run.path / "summary.json")
    run.metrics.write_text("")  # a crashed earlier attempt is redone from scratch
    setup_logging(run.stdout)
    seed_everything(seed)
    tok = load_tokenizer(cfg)
    problems = [Problem(f"agree/{i}", q, "", "agreement") for i, q in enumerate(AGREEMENT_PROBLEMS)]
    prompts = [prompt_ids(tok, p) for p in problems[:n]]

    timings: dict[str, float] = {}
    peak_used = 0.0
    with Monitor(run.monitor):
        vt = VLLMTeacher(
            cfg.path,
            cfg.revision,
            kv_cache_gb=kv_cache_gb,
            max_logprobs=max(top_k, 20),
            seed=seed,
            max_model_len=2048,  # prompts < 100 tokens + max 512 response tokens
        )
        t0 = time.perf_counter()
        vt.start()
        timings["vllm_load_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        rollouts = _generate(vt, prompts, max_tokens, seed)
        timings["generate_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        v_scores = vt.score(rollouts, top_k)
        timings["vllm_score_s"] = time.perf_counter() - t0
        peak_used = max(peak_used, snapshot().mem_used_gb)

        ht = HFTeacher(cfg.path, cfg.revision, pad_token_id=int(tok.pad_token_id or 0))
        t0 = time.perf_counter()
        ht.start()
        timings["hf_load_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        h_scores = ht.score(rollouts, top_k)
        timings["hf_score_s"] = time.perf_counter() - t0
        peak_used = max(peak_used, snapshot().mem_used_gb)
        ht.stop()
        vt.stop()

    io.write_jsonl(run.path / "rollouts.jsonl", (r.to_dict() for r in rollouts))
    for i, (r, hs, vs) in enumerate(zip(rollouts, h_scores, v_scores, strict=True)):
        m = agreement_metrics([hs], [vs])
        run.log_metrics({"seq": i, "response_len": len(r.response_ids), **m})
    summary = {
        **agreement_metrics(h_scores, v_scores),
        "n_sequences": len(rollouts),
        "mean_response_len": statistics.fmean(len(r.response_ids) for r in rollouts),
        "truncated": sum(r.truncated for r in rollouts),
        "threshold_nats": THRESHOLD_NATS,
        "reference": "hf",
        "candidate": "vllm",
        "peak_mem_used_gb_system": peak_used,
        **timings,
    }
    summary["pass"] = summary["mean_abs_diff"] < THRESHOLD_NATS
    io.write_json(run.path / "summary.json", summary)
    run.mark_done()
    log.info("agreement %s: mean |Δ| = %.4f nats", model_id, summary["mean_abs_diff"])
    return summary
