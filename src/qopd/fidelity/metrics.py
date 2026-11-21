"""Teacher-signal fidelity of a quantized rung against BF16 (PLAN 7.3).

All metrics are over the response tokens of a fixed rollout pool scored by both teachers
(same token ids, so differences come only from the teacher). With
``r_t^(b) = log p_T^(b)(y_t) - log p_S(y_t)``:

- ``pearson``        Pearson correlation of r_t^(b) with r_t^(16) over tokens
- ``mae``            mean |r_t^(b) - r_t^(16)| (= mean |log p_T^(b) - log p_T^(16)|)
- ``topk_kl``        mean over positions of KL(p~_16 || p~_b) on K_t, the BF16 top-k ids, both
                     renormalized on K_t. A rung only returns its own top-k, so an id of K_t
                     missing from it gets the rung's k-th log-prob (an upper bound on its true
                     value); ``topk_kl_imputed`` is the share of such entries (step 7.3 decision)
- ``top1_agreement`` share of positions where both teachers have the same argmax token
- ``sign_flip_rate`` share of tokens where r_t^(b) and r_t^(16) have opposite signs
- ``informativeness`` the measure of the student–teacher mismatch paper cited in the doc:
                     not implemented until its exact definition is available (PLAN 7.3 says
                     ask for the PDF rather than guess) -> ``None``

Each metric is also reported per student-entropy bucket (nats; fixed edges so buckets are
comparable across students and rungs).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from itertools import pairwise

import numpy as np

from qopd.types import Rollout, TeacherScores

DEFAULT_ENTROPY_EDGES: tuple[float, ...] = (0.0, 0.1, 0.5, 1.0, 2.0, float("inf"))


@dataclass
class FidelityReport:
    rung: str
    student: str
    n_tokens: int
    pearson: float
    mae: float
    topk_kl: float
    topk_kl_imputed: float
    top1_agreement: float
    sign_flip_rate: float
    informativeness: float | None = None
    by_entropy_bucket: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def rewards(rollouts: Sequence[Rollout], scores: Sequence[TeacherScores]) -> list[list[float]]:
    """r_t = log p_T(y_t) - log p_S(y_t) per rollout (student log-probs from sampling)."""
    out = []
    for r, s in zip(rollouts, scores, strict=True):
        if len(r.sampled_logprobs) != len(s.token_logprobs):
            raise ValueError("student and teacher log-prob lengths differ")
        out.append([t - st for t, st in zip(s.token_logprobs, r.sampled_logprobs, strict=True)])
    return out


def _flat(rows: Sequence[Sequence[float]]) -> np.ndarray:
    return np.fromiter((x for row in rows for x in row), dtype=float)


def _logsumexp(a: np.ndarray) -> float:
    m = float(a.max())
    return m + float(np.log(np.exp(a - m).sum()))


def topk_kl_position(
    ref_ids: Sequence[int],
    ref_lp: Sequence[float],
    cand_ids: Sequence[int],
    cand_lp: Sequence[float],
) -> tuple[float, int]:
    """KL(p~_ref || p~_cand) on the reference top-k ids; returns (kl, number imputed)."""
    lookup = dict(zip(cand_ids, cand_lp, strict=True))
    floor = min(cand_lp) if cand_lp else -np.inf
    c = np.array([lookup.get(i, floor) for i in ref_ids], dtype=float)
    imputed = sum(1 for i in ref_ids if i not in lookup)
    r = np.asarray(ref_lp, dtype=float)
    lr, lc = r - _logsumexp(r), c - _logsumexp(c)
    return float((np.exp(lr) * (lr - lc)).sum()), imputed


def _metrics(
    r_ref: np.ndarray, r_cand: np.ndarray, kl: np.ndarray, top1: np.ndarray
) -> dict[str, float]:
    n = r_ref.size
    if n == 0:
        return {"n_tokens": 0.0}
    pear = (
        float(np.corrcoef(r_ref, r_cand)[0, 1])
        if n > 1 and r_ref.std() > 0 and r_cand.std() > 0
        else float("nan")
    )
    return {
        "n_tokens": float(n),
        "pearson": pear,
        "mae": float(np.abs(r_cand - r_ref).mean()),
        "topk_kl": float(kl.mean()) if kl.size else float("nan"),
        "top1_agreement": float(top1.mean()) if top1.size else float("nan"),
        "sign_flip_rate": float((np.sign(r_ref) * np.sign(r_cand) < 0).mean()),
    }


def bucket_label(lo: float, hi: float) -> str:
    return f"[{lo:g},{hi:g})" if np.isfinite(hi) else f"[{lo:g},inf)"


def compare(
    rollouts: Sequence[Rollout],
    reference: Sequence[TeacherScores],
    candidate: Sequence[TeacherScores],
    entropies: Sequence[Sequence[float]] | None = None,
    entropy_edges: Sequence[float] = DEFAULT_ENTROPY_EDGES,
    rung: str = "",
    student: str = "",
) -> FidelityReport:
    """Fidelity of ``candidate`` scores against ``reference`` (BF16) on the same rollouts.

    ``entropies``: per-rollout student entropy at every response position (nats).
    """
    r_ref = _flat(rewards(rollouts, reference))
    r_cand = _flat(rewards(rollouts, candidate))
    kls, imputed, n_pairs, top1 = [], 0, 0, []
    for ref, cand in zip(reference, candidate, strict=True):
        if not ref.topk_ids or not cand.topk_ids:
            continue
        for ri, rl, ci, cl in zip(
            ref.topk_ids, ref.topk_logprobs, cand.topk_ids, cand.topk_logprobs, strict=True
        ):
            kl, imp = topk_kl_position(ri, rl, ci, cl)
            kls.append(kl)
            imputed += imp
            n_pairs += len(ri)
            top1.append(ri[0] == ci[0])
    kl_arr, top1_arr = np.array(kls), np.array(top1, dtype=float)
    overall = _metrics(r_ref, r_cand, kl_arr, top1_arr)

    buckets: dict[str, dict[str, float]] = {}
    if entropies is not None:
        ent = _flat(entropies)
        if ent.size != r_ref.size:
            raise ValueError("entropies must have one value per response token")
        has_topk = kl_arr.size == r_ref.size
        for lo, hi in pairwise(entropy_edges):
            sel = (ent >= lo) & (ent < hi)
            buckets[bucket_label(lo, hi)] = _metrics(
                r_ref[sel],
                r_cand[sel],
                kl_arr[sel] if has_topk else np.array([]),
                top1_arr[sel] if has_topk else np.array([]),
            )
    return FidelityReport(
        rung=rung,
        student=student,
        n_tokens=int(r_ref.size),
        pearson=overall.get("pearson", float("nan")),
        mae=overall.get("mae", float("nan")),
        topk_kl=overall.get("topk_kl", float("nan")),
        topk_kl_imputed=imputed / n_pairs if n_pairs else float("nan"),
        top1_agreement=overall.get("top1_agreement", float("nan")),
        sign_flip_rate=overall.get("sign_flip_rate", float("nan")),
        informativeness=None,
        by_entropy_bucket=buckets,
    )


def informativeness(*args: object, **kwargs: object) -> float:
    raise NotImplementedError(
        "step 7.3: needs the exact definition from the student-teacher mismatch paper "
        "cited in the experiment plan (ask the user for the PDF; do not guess)"
    )
