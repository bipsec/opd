"""Evaluation suites in the unified schema, one parquet file per suite (PLAN 2.2).

Every suite is pinned to a Hub commit. Evaluation items are never filtered; each item
is annotated with its answer type (``meta.answer_type``) and whether ``\\boxed{gold}``
verifies against the gold (``meta.self_verifies``) so verifier gaps surface in 3.1.

GPQA-Diamond is gated (the user's HF access is needed); until then it is reported as
``blocked`` and nothing is substituted (CLAUDE.md rule 8).
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from qopd.config import PROJECT_ROOT, DataCfg
from qopd.data.filters import classify_answer, normalize_text, normalize_whitespace, self_verifies
from qopd.data.sources import DatasetNotFoundError, load_rows, write_parquet
from qopd.types import Problem, ProblemKind
from qopd.utils import io
from qopd.utils.io import sha256_file
from qopd.utils.logging import get_logger

log = get_logger("data.evalsets")
GPQA_SEED = 0  # fixed choice-shuffling seed (part of the suite definition)
LETTERS = "ABCD"

Row = dict[str, Any]


@dataclass(frozen=True)
class EvalSource:
    suite: str
    hf_repo: str
    revision: str
    filename: str
    convert: Callable[[str, int, Row], Problem]
    license: str | None
    expected_n: int | None = None
    gated: bool = False
    kind: ProblemKind = ProblemKind.MATH


# --- per-suite converters -----------------------------------------------------------------


def _math(suite: str, i: int, problem: str, answer: str, **meta: Any) -> Problem:
    return Problem(
        id=f"{suite}/{i:04d}",
        problem=normalize_text(problem),
        answer=normalize_whitespace(answer),
        source=suite,
        kind=ProblemKind.MATH,
        meta={k: v for k, v in meta.items() if v is not None},
    )


def gsm8k_answer(solution: str) -> str:
    """GSM8K gold: the text after ``####`` with thousands separators removed."""
    if "####" not in solution:
        raise ValueError("GSM8K answer without ####")
    return solution.rsplit("####", 1)[1].strip().replace(",", "")


def _gsm8k(s: str, i: int, r: Row) -> Problem:
    return _math(s, i, r["question"], gsm8k_answer(r["answer"]), row=i)


def _math500(s: str, i: int, r: Row) -> Problem:
    return _math(
        s, i, r["problem"], r["answer"], unique_id=r.get("unique_id"),
        subject=r.get("subject"), level=r.get("level"),
    )


def _amc23(s: str, i: int, r: Row) -> Problem:
    return _math(s, i, r["question"], str(r["answer"]), orig_id=str(r.get("id")), url=r.get("url"))


def _aime_h4(s: str, i: int, r: Row) -> Problem:
    return _math(s, i, r["problem"], str(r["answer"]), orig_id=str(r.get("id")), url=r.get("url"))


def _matharena(s: str, i: int, r: Row) -> Problem:
    return _math(
        s, i, r["problem"], str(r["answer"]), problem_idx=str(r.get("problem_idx")),
        problem_type=r.get("problem_type"),
    )


def gpqa_problem(suite: str, i: int, r: Row, seed: int = GPQA_SEED) -> Problem:
    """Four options shuffled deterministically per question; the answer is the letter."""
    correct = normalize_text(r["Correct Answer"])
    options = [correct] + [normalize_text(r[f"Incorrect Answer {k}"]) for k in (1, 2, 3)]
    rid = str(r.get("Record ID") or i)
    order = list(range(4))
    random.Random(f"{seed}:{rid}").shuffle(order)
    shuffled = [options[j] for j in order]
    letter = LETTERS[shuffled.index(correct)]
    body = normalize_text(r["Question"]) + "\n\n" + "\n".join(
        f"{LETTERS[k]}) {opt}" for k, opt in enumerate(shuffled)
    )
    return Problem(
        id=f"{suite}/{i:04d}",
        problem=body,
        answer=letter,
        source=suite,
        kind=ProblemKind.MULTIPLE_CHOICE,
        meta={"record_id": rid, "domain": r.get("High-level domain"), "shuffle_seed": seed},
    )


EVAL_SOURCES: dict[str, EvalSource] = {
    s.suite: s
    for s in (
        EvalSource("gsm8k", "openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866",
                   "main/test-00000-of-00001.parquet", _gsm8k, "mit", 1319),
        EvalSource("math500", "HuggingFaceH4/MATH-500", "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
                   "test.jsonl", _math500, None, 500),
        EvalSource("amc23", "math-ai/amc23", "80815d37005feb82cd7f8fbc6901d5d3eff43057",
                   "test-00000-of-00001.parquet", _amc23, None, 40),
        EvalSource("aime24", "HuggingFaceH4/aime_2024", "2fe88a2f1091d5048c0f36abc874fb997b3dd99a",
                   "data/train-00000-of-00001.parquet", _aime_h4, None, 30),
        EvalSource("aime25", "MathArena/aime_2025", "c94da77eb22bbd6439e62a323bec18493a421302",
                   "data/train-00000-of-00001.parquet", _matharena, "cc-by-nc-sa-4.0", 30),
        EvalSource("aime26", "MathArena/aime_2026", "d61f486805306e6e6b5913504c07763a99e1b112",
                   "data/train-00000-of-00001.parquet", _matharena, "cc-by-nc-sa-4.0", 30),
        EvalSource("hmmt_feb26", "MathArena/hmmt_feb_2026",
                   "589a53fe152febc0f4f99f3e561007bf8da8e664",
                   "data/train-00000-of-00001.parquet", _matharena, "cc-by-nc-sa-4.0", None),
        EvalSource("gpqa_diamond", "Idavidrein/gpqa", "83022cefff930aea54f654c0b282e74b9eeda5c6",
                   "gpqa_diamond.csv", gpqa_problem, "cc-by-4.0", 198, gated=True,
                   kind=ProblemKind.MULTIPLE_CHOICE),
    )
}


def annotate(p: Problem) -> Problem:
    """Record answer type and boxed self-verification (math only); never drops."""
    if p.kind != ProblemKind.MATH:
        return p
    return replace(
        p, meta={**p.meta, "answer_type": classify_answer(p.answer),
                 "self_verifies": self_verifies(p.answer)},
    )


def eval_dir(cfg: DataCfg) -> Path:
    d = Path(cfg.data_dir)
    return (d if d.is_absolute() else PROJECT_ROOT / d) / "eval"


def build_suite(src: EvalSource, out_dir: Path) -> dict[str, Any]:
    """Download, convert, annotate and write one suite; returns its stats record."""
    from huggingface_hub import hf_hub_download  # noqa: PLC0415
    from huggingface_hub.utils import GatedRepoError, HfHubHTTPError  # noqa: PLC0415

    rec: dict[str, Any] = {
        "suite": src.suite, "hf_repo": src.hf_repo, "revision": src.revision,
        "file": src.filename, "license": src.license, "gated": src.gated,
    }
    try:
        path = Path(hf_hub_download(src.hf_repo, src.filename, repo_type="dataset",
                                    revision=src.revision))
    except GatedRepoError as e:
        return {**rec, "status": "blocked", "reason": f"gated: needs the user's HF access ({e.__class__.__name__})"}
    except HfHubHTTPError as e:
        raise DatasetNotFoundError(f"{src.suite}: {src.hf_repo}@{src.revision}: {e}") from e

    rows = load_rows(path)
    problems = [annotate(src.convert(src.suite, i, r)) for i, r in enumerate(rows)]
    if src.expected_n is not None and len(problems) != src.expected_n:
        log.warning("%s: %d items, expected %d", src.suite, len(problems), src.expected_n)
    sha = write_parquet(problems, out_dir / f"{src.suite}.parquet")
    math_items = [p for p in problems if p.kind == ProblemKind.MATH]
    types: dict[str, int] = {}
    for p in math_items:
        types[p.meta["answer_type"]] = types.get(p.meta["answer_type"], 0) + 1
    return {
        **rec,
        "status": "ok",
        "file_sha256_source": sha256_file(path),
        "n": len(problems),
        "expected_n": src.expected_n,
        "output": str((out_dir / f"{src.suite}.parquet").relative_to(PROJECT_ROOT)),
        "sha256": sha,
        "answer_types": dict(sorted(types.items())),
        "not_self_verifying": sum(1 for p in math_items if not p.meta["self_verifies"]),
    }


def build_eval_suites(cfg: DataCfg, suites: list[str] | None = None) -> dict[str, Any]:
    """Build every (or the named) suite; writes ``data/eval/eval_stats.json``."""
    out = eval_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    stats_path = out / "eval_stats.json"
    stats = io.read_json(stats_path) if stats_path.exists() else {"step": "2.2", "suites": {}}
    t0 = time.perf_counter()
    for name in suites or list(EVAL_SOURCES):
        rec = build_suite(EVAL_SOURCES[name], out)
        stats["suites"][name] = rec
        log.info("%-13s %s n=%s", name, rec["status"], rec.get("n"))
    stats["seconds"] = round(time.perf_counter() - t0, 1)
    io.write_json(stats_path, stats)
    return stats


def load_suite(cfg: DataCfg, suite: str) -> list[Problem]:
    from qopd.data.sources import read_parquet  # noqa: PLC0415

    return read_parquet(eval_dir(cfg) / f"{suite}.parquet")


def report_section(stats: dict[str, Any]) -> list[str]:
    lines = [
        "## Evaluation suites (step 2.2)",
        "",
        "Items are never filtered. `answer types` uses the same math-verify classification as "
        "the pool; `not self-verifying` counts items whose gold does not verify against "
        "`\\boxed{gold}` (to be handled by the 3.1 verifier).",
        "",
        "| Suite | Status | Items | Source @ revision | License | Answer types | Not self-verifying |",
        "| --- | --- | ---: | --- | --- | --- | ---: |",
    ]
    for name, r in stats["suites"].items():
        src = f"`{r['hf_repo']}` @ `{r['revision'][:8]}`"
        if r["status"] != "ok":
            lines.append(f"| {name} | **{r['status']}**: {r['reason']} | TBD | {src} | "
                         f"{r['license'] or 'not stated'} | — | — |")
            continue
        types = ", ".join(f"{k}: {v}" for k, v in r["answer_types"].items()) or "— (multiple choice)"
        lines.append(f"| {name} | ok | {r['n']} | {src} | {r['license'] or 'not stated'} | "
                     f"{types} | {r['not_self_verifying']} |")
    lines.append("")
    return lines
