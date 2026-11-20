"""Evaluation runner (PLAN 3.2): generate, verify, pass@1 averaged over samples.

- One request per (problem, sample index) with its own seed ``sample_seed(base, id, k)``,
  so every sample is reproducible and resumable independently.
- Requests run in chunks; after each chunk the verified samples are appended to
  ``samples.jsonl`` and ``state.json`` accumulates generation time per suite, so a crashed
  or interrupted run resumes where it stopped (already-scored pairs are skipped).
- ``summary.json`` holds, per suite: pass@1 (mean over problems of the fraction of correct
  samples), its standard error over problems, response length, truncation and no-answer
  rates, generation wall-clock and throughput.
"""

from __future__ import annotations

import hashlib
import math
import statistics
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from qopd.config import EvalCfg, SamplingCfg
from qopd.eval.verifiers import verify
from qopd.prompts.templates import prompt_ids as build_prompt_ids
from qopd.types import Problem, Rollout
from qopd.utils import io
from qopd.utils.logging import get_logger
from qopd.utils.run import RunDir

log = get_logger("eval")
NO_ANSWER_REASONS = ("no_answer", "no_final_answer", "unparsable")


class Engine(Protocol):
    stop_ids: list[int]

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def generate(
        self,
        prompt_ids: list[list[int]],
        sampling: SamplingCfg,
        n: int = 1,
        problem_ids: list[str] | None = None,
        seed: int | None = None,
        seeds: list[int] | None = None,
    ) -> tuple[list[Rollout], Any]: ...


@dataclass(frozen=True)
class Request:
    suite: str
    problem: Problem
    sample_idx: int
    seed: int


@dataclass
class SuiteResult:
    suite: str
    n_problems: int
    n_samples: int
    pass_at_1: float
    seconds: float


def sample_seed(base: int, problem_id: str, k: int) -> int:
    """Deterministic 31-bit seed per (problem, sample index)."""
    return int(hashlib.sha256(f"{base}:{problem_id}:{k}".encode()).hexdigest()[:8], 16) & 0x7FFFFFFF


def pass_at_1(correct_by_problem: Mapping[str, Sequence[bool]]) -> float:
    """Mean over problems of the per-problem fraction of correct samples."""
    if not correct_by_problem:
        raise ValueError("no problems")
    return sum(sum(v) / len(v) for v in correct_by_problem.values()) / len(correct_by_problem)


def summarize_suite(
    rows: list[dict[str, Any]], n_problems: int, n_samples: int, gen_seconds: float
) -> dict[str, Any]:
    by_problem: dict[str, list[bool]] = defaultdict(list)
    for r in rows:
        by_problem[r["problem_id"]].append(bool(r["correct"]))
    per_problem = [sum(v) / len(v) for v in by_problem.values()]
    tokens = [r["n_tokens"] for r in rows]
    n = len(per_problem)
    out: dict[str, Any] = {
        "n_problems": n_problems,
        "n_samples_per_problem": n_samples,
        "n_samples_done": len(rows),
        "complete": len(rows) == n_problems * n_samples,
        "pass_at_1": pass_at_1(by_problem) if by_problem else float("nan"),
        "stderr_over_problems": (statistics.stdev(per_problem) / math.sqrt(n))
        if n > 1
        else float("nan"),
        "mean_response_tokens": statistics.fmean(tokens) if tokens else float("nan"),
        "max_response_tokens": max(tokens) if tokens else 0,
        "truncation_rate": (sum(r["finish_reason"] == "length" for r in rows) / len(rows))
        if rows
        else float("nan"),
        "no_answer_rate": (sum(r["reason"] in NO_ANSWER_REASONS for r in rows) / len(rows))
        if rows
        else float("nan"),
        "gen_seconds": gen_seconds,
        "tokens_per_s": (sum(tokens) / gen_seconds) if gen_seconds > 0 else float("nan"),
    }
    return out


class EvalRunner:
    """Evaluate one model on several suites inside a run directory (resumable)."""

    def __init__(
        self,
        tokenizer: Any,
        cfg: EvalCfg,
        run: RunDir,
        engine_factory: Callable[[int], Engine],
        chunk_size: int = 256,
        seed: int = 0,
    ) -> None:
        self.tok = tokenizer
        self.cfg = cfg
        self.run_dir = run
        self.engine_factory = engine_factory  # max_model_len -> engine
        self.chunk_size = chunk_size
        self.seed = seed
        self.samples_path = run.path / "samples.jsonl"

    # --- bookkeeping -----------------------------------------------------------------
    def _done_rows(self) -> list[dict[str, Any]]:
        return io.read_jsonl(self.samples_path) if self.samples_path.exists() else []

    def _state(self) -> dict[str, Any]:
        return self.run_dir.load_state() if self.run_dir.state_file.exists() else {}

    def _add_seconds(self, suite: str, seconds: float) -> None:
        secs = dict(self._state().get("gen_seconds", {}))
        secs[suite] = secs.get(suite, 0.0) + seconds
        self.run_dir.save_state(gen_seconds=secs)

    # --- main --------------------------------------------------------------------------
    def plan(
        self, problems: Mapping[str, list[Problem]], n_samples: Mapping[str, int]
    ) -> list[Request]:
        done = {(r["suite"], r["problem_id"], r["sample_idx"]) for r in self._done_rows()}
        reqs = []
        for suite, ps in problems.items():
            for p in ps:
                for k in range(n_samples[suite]):
                    if (suite, p.id, k) not in done:
                        reqs.append(Request(suite, p, k, sample_seed(self.seed, p.id, k)))
        return reqs

    def run(
        self, problems: Mapping[str, list[Problem]], n_samples: Mapping[str, int]
    ) -> dict[str, Any]:
        reqs = self.plan(problems, n_samples)
        self.run_dir.save_state(status="running", pending=len(reqs))
        if reqs:
            prompts = {
                p.id: build_prompt_ids(self.tok, p, self.cfg.sampling.enable_thinking)
                for ps in problems.values()
                for p in ps
            }
            max_len = (
                max(len(prompts[r.problem.id]) for r in reqs) + self.cfg.sampling.max_tokens + 16
            )
            engine = self.engine_factory(max_len)
            engine.start()
            try:
                for start in range(0, len(reqs), self.chunk_size):
                    self._run_chunk(engine, reqs[start : start + self.chunk_size], prompts)
                    log.info(
                        "eval: %d/%d requests done",
                        min(start + self.chunk_size, len(reqs)),
                        len(reqs),
                    )
            finally:
                engine.stop()
        summary = self.write_summary(problems, n_samples)
        if all(s["complete"] for s in summary["suites"].values()):
            self.run_dir.mark_done()
        return summary

    def _run_chunk(
        self, engine: Engine, chunk: list[Request], prompts: dict[str, list[int]]
    ) -> None:
        by_suite: dict[str, list[Request]] = defaultdict(list)
        for r in chunk:
            by_suite[r.suite].append(r)
        for suite, rs in by_suite.items():  # per suite, so wall-clock is attributable
            t0 = time.perf_counter()
            rollouts, stats = engine.generate(
                [prompts[r.problem.id] for r in rs],
                self.cfg.sampling,
                n=1,
                problem_ids=[r.problem.id for r in rs],
                seeds=[r.seed for r in rs],
            )
            seconds = time.perf_counter() - t0
            rows = []
            for r, ro in zip(rs, rollouts, strict=True):
                text = self.tok.decode(ro.response_ids, skip_special_tokens=True)
                v = verify(r.problem, text)
                rows.append(
                    {
                        "suite": suite,
                        "problem_id": r.problem.id,
                        "sample_idx": r.sample_idx,
                        "seed": r.seed,
                        "correct": v.correct,
                        "extracted": v.extracted,
                        "reason": v.reason,
                        "method": v.method,
                        "finish_reason": ro.finish_reason.value,
                        "n_tokens": len(ro.response_ids),
                        "text": text,
                    }
                )
            for row in rows:
                io.append_jsonl(self.samples_path, row)
            self._add_seconds(suite, seconds)
            peak = getattr(stats, "peak_mem_gb", None)
            if peak is not None:
                st = self._state()
                self.run_dir.save_state(peak_mem_gb=max(st.get("peak_mem_gb", 0.0), peak))

    def write_summary(
        self, problems: Mapping[str, list[Problem]], n_samples: Mapping[str, int]
    ) -> dict[str, Any]:
        rows = self._done_rows()
        state = self._state()
        suites = {}
        for suite, ps in problems.items():
            ids = {p.id for p in ps}
            srows = [r for r in rows if r["suite"] == suite and r["problem_id"] in ids]
            suites[suite] = summarize_suite(
                srows, len(ps), n_samples[suite], state.get("gen_seconds", {}).get(suite, 0.0)
            )
        summary = {
            "suites": suites,
            "sampling": self.cfg.sampling.model_dump(mode="json"),
            "seed": self.seed,
            "peak_mem_gb_system": state.get("peak_mem_gb"),
        }
        io.write_json(self.run_dir.path / "summary.json", summary)
        return summary


def evaluate(
    model: str,
    suites: Sequence[str],
    cfg: EvalCfg,
    *,
    n_samples: int | None = None,
    limit: int | None = None,
    seed: int = 0,
    kv_cache_gb: float = 8.0,
    run_id: str | None = None,
    root: Any = None,
    chunk_size: int = 256,
) -> dict[str, Any]:
    """CLI entry: registry id or local checkpoint path -> run dir under runs/eval/."""
    from pathlib import Path  # noqa: PLC0415

    from qopd.config import CONFIG_ROOT, PROJECT_ROOT, DataCfg, load  # noqa: PLC0415
    from qopd.data.evalsets import load_suite  # noqa: PLC0415
    from qopd.eval.suites import get_suite  # noqa: PLC0415
    from qopd.models.loading import load_tokenizer  # noqa: PLC0415
    from qopd.models.registry import ModelId, resolve  # noqa: PLC0415
    from qopd.rollout.engine import RolloutEngine  # noqa: PLC0415
    from qopd.utils.io import sha256_file  # noqa: PLC0415
    from qopd.utils.logging import setup_logging  # noqa: PLC0415
    from qopd.utils.monitor import Monitor  # noqa: PLC0415
    from qopd.utils.run import create_run_dir  # noqa: PLC0415

    if model in {m.value for m in ModelId}:
        mcfg = resolve(model)
        path, revision, tag = mcfg.path, mcfg.revision, model
        tokenizer = load_tokenizer(mcfg)
    else:
        from transformers import AutoTokenizer  # noqa: PLC0415

        path, revision, tag = str(Path(model).resolve()), None, Path(model).name
        tokenizer = AutoTokenizer.from_pretrained(path)
    data_cfg = load(CONFIG_ROOT / "data" / "default.yaml", cls=DataCfg)
    problems = {
        s: load_suite(data_cfg, s)[:limit] if limit else load_suite(data_cfg, s) for s in suites
    }
    n_per = {s: n_samples or get_suite(s, cfg).n_samples for s in suites}
    manifest = PROJECT_ROOT / "data" / "manifest.json"
    run_cfg = {
        "task": "eval",
        "model": {"id": tag, "path": path, "revision": revision},
        "suites": {s: {"n_problems": len(problems[s]), "n_samples": n_per[s]} for s in suites},
        "limit": limit,
        "sampling": cfg.sampling.model_dump(mode="json"),
        "seed": seed,
        "kv_cache_gb": kv_cache_gb,
        "data_manifest_sha256": sha256_file(manifest) if manifest.exists() else None,
    }
    rid = run_id or "__".join(
        [tag, "+".join(suites), f"n{n_samples or 'cfg'}", f"t{cfg.sampling.max_tokens}", f"s{seed}"]
        + ([f"lim{limit}"] if limit else [])
    )
    run = create_run_dir("eval", rid, run_cfg, root=root)
    setup_logging(run.stdout)

    def factory(max_len: int) -> RolloutEngine:
        return RolloutEngine(
            path,
            revision,
            kv_cache_gb=kv_cache_gb,
            max_model_len=max_len,
            seed=seed,
            enable_prefix_caching=True,
        )

    runner = EvalRunner(tokenizer, cfg, run, factory, chunk_size=chunk_size, seed=seed)
    with Monitor(run.monitor):
        summary = runner.run(problems, n_per)
    summary["run_dir"] = str(run.path)
    return summary
