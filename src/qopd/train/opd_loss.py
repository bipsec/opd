"""Sampled-token reverse-KL policy-gradient loss for on-policy distillation (PLAN 6.1).

Per response token t:  r_t = log p_T(y_t) − log p_S(y_t)   (stop-gradient)
Loss:                  L = −Σ_t r_t · log p_S(y_t)          over unmasked response tokens

Options:
- reward clip τ:        r_t ← clip(r_t, −τ, τ)
- top-k KL variant:     reverse KL against the teacher top-k (k=20), renormalized
- hybrid:               r_t + λ · verifier reward (sequence-level, broadcast to tokens)
All options reduce exactly to the base loss when disabled (tested in step 6.1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class OPDLossCfg:
    clip_tau: float | None = None  # None = no clipping
    top_k: int | None = None  # None = sampled-token estimator only
    hybrid_weight: float = 0.0  # λ for the verifier reward
    normalize: str = "tokens"  # "tokens" (mean over unmasked tokens) or "sequences"


@dataclass
class OPDLossOutput:
    loss: torch.Tensor
    reward_mean: float
    reward_std: float
    n_tokens: int


def per_token_reward(
    teacher_logprobs: torch.Tensor,  # [B, T]
    student_logprobs: torch.Tensor,  # [B, T]
    mask: torch.Tensor,  # [B, T] 1 = response token
    clip_tau: float | None = None,
) -> torch.Tensor:
    """r_t = log p_T − log p_S, detached, optionally clipped, zero where masked."""
    raise NotImplementedError("step 6.1")


def opd_loss(
    student_logprobs: torch.Tensor,  # [B, T] log p_S(y_t), requires grad
    teacher_logprobs: torch.Tensor,  # [B, T]
    mask: torch.Tensor,  # [B, T]
    cfg: OPDLossCfg,
    *,
    verifier_reward: torch.Tensor | None = None,  # [B]
    student_topk_logprobs: torch.Tensor | None = None,  # [B, T, k] at teacher top-k ids
    teacher_topk_logprobs: torch.Tensor | None = None,  # [B, T, k]
) -> OPDLossOutput:
    raise NotImplementedError("step 6.1")
