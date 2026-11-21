"""Fixed rollout pools for E1 and their teacher scores (PLAN 7.1, 7.2).

Pool (7.1), per cold-started student: ``n_samples`` rollouts for each probe prompt (doc:
500 x 4 = 2,000), sampled with the OPD rollout settings and stored as exact token ids. The
student's log-probability and entropy at every response token are then recomputed with the
HF student (the same quantities the OPD trainer uses), so r_t and the entropy buckets of
H3 come from one consistent forward pass.

Scores (7.2), per rung: the log-prob of every realized response token plus the top-k ids and
log-probs, in chunked ``.npz`` files (flat arrays + offsets; ~1 GB per rung at 2,000 x ~3k
tokens x top-20), written chunk by chunk so a crashed scoring job resumes.

Layout::

    <pool_dir>/rollouts.jsonl         one Rollout per line (exact ids, engine log-probs)
    <pool_dir>/student.npz            offsets, student_logprobs, entropy (trainer recompute)
    <pool_dir>/scores/<rung>/chunk_XXXX.npz + DONE
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from qopd.types import Problem, Rollout, TeacherScores
from qopd.utils import io
from qopd.utils.logging import get_logger

log = get_logger("fidelity.pool")


# --- pool ----------------------------------------------------------------------------------


def build_pool(
    engine: Any,
    problems: Sequence[Problem],
    prompt_ids: Callable[[Problem], list[int]],
    sampling: Any,
    n_samples: int,
    out_dir: Path,
    seed: int = 0,
) -> Path:
    """Sample the pool (skips if rollouts.jsonl exists); returns its path."""
    from qopd.train.opd_loop import request_seed  # noqa: PLC0415

    path = out_dir / "rollouts.jsonl"
    if path.exists():
        return path
    out_dir.mkdir(parents=True, exist_ok=True)
    engine.start()
    try:
        rollouts, _ = engine.generate(
            [prompt_ids(p) for p in problems],
            sampling,
            n=n_samples,
            problem_ids=[p.id for p in problems],
            seeds=[request_seed(seed, -1, p.id) for p in problems],
        )
    finally:
        engine.stop()
    io.write_jsonl(path, (r.to_dict() for r in rollouts))
    log.info("pool: %d rollouts -> %s", len(rollouts), path)
    return path


def load_pool(pool_dir: Path) -> list[Rollout]:
    return [Rollout.from_dict(d) for d in io.iter_jsonl(pool_dir / "rollouts.jsonl")]


def offsets_of(rollouts: Sequence[Rollout]) -> np.ndarray:
    return np.concatenate([[0], np.cumsum([len(r.response_ids) for r in rollouts])]).astype(
        np.int64
    )


def student_stats(
    model: Any, rollouts: Sequence[Rollout], pad_id: int, device: str, micro_batch_size: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """(log p_S of each realized token, entropy at each position), flat over the pool."""
    import torch  # noqa: PLC0415

    from qopd.train.opd_loop import _pack, _response_logits  # noqa: PLC0415
    from qopd.train.opd_loss import gather_logprobs, token_entropy  # noqa: PLC0415

    lps, ents = [], []
    model.eval()
    with torch.no_grad():
        for j in range(0, len(rollouts), micro_batch_size):
            rs = rollouts[j : j + micro_batch_size]
            dummy = [TeacherScores([0.0] * len(r.response_ids)) for r in rs]
            pk = _pack(rs, dummy, pad_id, device, 0)
            logits = _response_logits(model, pk)
            lp, ent = gather_logprobs(logits, pk["resp"]), token_entropy(logits)
            for i, r in enumerate(rs):
                n = len(r.response_ids)
                lps.append(lp[i, :n].cpu().numpy())
                ents.append(ent[i, :n].cpu().numpy())
    return np.concatenate(lps).astype(np.float32), np.concatenate(ents).astype(np.float32)


def write_student_stats(
    pool_dir: Path, rollouts: Sequence[Rollout], lp: np.ndarray, ent: np.ndarray
) -> Path:
    off = offsets_of(rollouts)
    if lp.size != off[-1] or ent.size != off[-1]:
        raise ValueError("student stats do not match the pool's response tokens")
    path = pool_dir / "student.npz"
    np.savez(path, offsets=off, student_logprobs=lp, entropy=ent)
    return path


def load_student_stats(pool_dir: Path) -> tuple[list[list[float]], list[list[float]]]:
    """Per-rollout (student log-probs, entropies) from student.npz."""
    z = np.load(pool_dir / "student.npz")
    off = z["offsets"]
    split = lambda a: [a[off[i] : off[i + 1]].tolist() for i in range(len(off) - 1)]  # noqa: E731
    return split(z["student_logprobs"]), split(z["entropy"])


def pool_for_metrics(pool_dir: Path) -> tuple[list[Rollout], list[list[float]]]:
    """Rollouts whose ``sampled_logprobs`` are the trainer-recomputed student log-probs, plus
    per-rollout entropies: exactly what :func:`qopd.fidelity.metrics.compare` needs."""
    rollouts = load_pool(pool_dir)
    lps, ents = load_student_stats(pool_dir)
    fixed = [
        Rollout(r.prompt_ids, r.response_ids, lp, r.finish_reason, r.problem_id, r.sample_idx)
        for r, lp in zip(rollouts, lps, strict=True)
    ]
    return fixed, ents


# --- scores -------------------------------------------------------------------------------------


def _chunk_path(d: Path, i: int) -> Path:
    return d / f"chunk_{i:04d}.npz"


def _save_chunk(path: Path, scores: Sequence[TeacherScores], top_k: int) -> None:
    lens = [len(s.token_logprobs) for s in scores]
    n = sum(lens)
    ids = np.zeros((n, top_k), dtype=np.int32)
    tlp = np.zeros((n, top_k), dtype=np.float32)
    pos = 0
    for s in scores:
        m = len(s.token_logprobs)
        if top_k:
            ids[pos : pos + m] = np.asarray(s.topk_ids, dtype=np.int32).reshape(m, top_k)
            tlp[pos : pos + m] = np.asarray(s.topk_logprobs, dtype=np.float32).reshape(m, top_k)
        pos += m
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(
        tmp,
        lens=np.asarray(lens, dtype=np.int64),
        token_logprobs=np.fromiter(
            (x for s in scores for x in s.token_logprobs), dtype=np.float32, count=n
        ),
        topk_ids=ids,
        topk_logprobs=tlp,
    )
    tmp.replace(path)  # a chunk is either complete or absent


def score_pool(
    scorer: Any,
    rollouts: Sequence[Rollout],
    top_k: int,
    out_dir: Path,
    chunk_size: int = 100,
) -> Path:
    """Score the pool with one rung, chunk by chunk (resumable); returns ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "DONE").exists():
        return out_dir
    n_chunks = (len(rollouts) + chunk_size - 1) // chunk_size
    todo = [i for i in range(n_chunks) if not _chunk_path(out_dir, i).exists()]
    if todo:
        scorer.start()
        try:
            for i in todo:
                chunk = rollouts[i * chunk_size : (i + 1) * chunk_size]
                _save_chunk(_chunk_path(out_dir, i), scorer.score(chunk, top_k), top_k)
                log.info("scored chunk %d/%d -> %s", i + 1, n_chunks, out_dir)
        finally:
            scorer.stop()
    io.write_json(
        out_dir / "meta.json",
        {
            "n_rollouts": len(rollouts),
            "top_k": top_k,
            "chunk_size": chunk_size,
            "n_chunks": n_chunks,
        },
    )
    (out_dir / "DONE").write_text("ok\n")
    return out_dir


def load_scores(out_dir: Path) -> list[TeacherScores]:
    meta = io.read_json(out_dir / "meta.json")
    scores: list[TeacherScores] = []
    for i in range(meta["n_chunks"]):
        z = np.load(_chunk_path(out_dir, i))
        pos = 0
        for m in z["lens"].tolist():
            scores.append(
                TeacherScores(
                    z["token_logprobs"][pos : pos + m].tolist(),
                    z["topk_ids"][pos : pos + m].tolist(),
                    z["topk_logprobs"][pos : pos + m].tolist(),
                )
            )
            pos += m
    if len(scores) != meta["n_rollouts"]:
        raise ValueError(f"{out_dir}: {len(scores)} scores for {meta['n_rollouts']} rollouts")
    return scores


def fidelity_for(pool_dir: Path, rung: str, reference: str = "T16", student: str = "") -> Any:
    """Fidelity report of ``rung`` vs ``reference`` on one pool (7.3 metrics)."""
    from qopd.fidelity.metrics import compare  # noqa: PLC0415

    rollouts, ents = pool_for_metrics(pool_dir)
    ref = load_scores(pool_dir / "scores" / reference)
    cand = load_scores(pool_dir / "scores" / rung)
    return compare(rollouts, ref, cand, entropies=ents, rung=rung, student=student)
