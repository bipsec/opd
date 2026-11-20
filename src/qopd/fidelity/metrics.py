"""Fidelity metrics of a quantized rung against BF16 (PLAN 7.3).

Per-token reward r_t under each rung vs under T16: Pearson correlation, mean absolute
error, top-20 KL, top-1 agreement, share of sign flips, informativeness — each also
broken down by student-entropy bucket.

Informativeness follows the definition in the student–teacher mismatch paper cited in
the experiment-plan doc; if that definition is ambiguous, ask the user for the PDF.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from qopd.types import Rollout, TeacherScores


@dataclass
class FidelityReport:
    rung: str
    student: str
    pearson: float
    mae: float
    topk_kl: float
    top1_agreement: float
    sign_flip_rate: float
    informativeness: float | None
    by_entropy_bucket: dict[str, dict[str, float]] = field(default_factory=dict)


def rewards(rollouts: Sequence[Rollout], scores: Sequence[TeacherScores]) -> list[list[float]]:
    """r_t = log p_T(y_t) − log p_S(y_t) per rollout (student log-probs from sampling)."""
    out = []
    for r, s in zip(rollouts, scores, strict=True):
        if len(r.sampled_logprobs) != len(s.token_logprobs):
            raise ValueError("student and teacher log-prob lengths differ")
        out.append([t - st for t, st in zip(s.token_logprobs, r.sampled_logprobs, strict=True)])
    return out


def compare(
    rollouts: Sequence[Rollout],
    reference: Sequence[TeacherScores],
    candidate: Sequence[TeacherScores],
    entropy_buckets: Sequence[float] = (),
) -> FidelityReport:
    raise NotImplementedError("step 7.3")


def informativeness(*args: object, **kwargs: object) -> float:
    raise NotImplementedError("step 7.3 (definition from the cited paper; ask if ambiguous)")
