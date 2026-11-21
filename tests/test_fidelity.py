"""PLAN 7.3: fidelity metrics on synthetic scores with known answers (CPU)."""

import math

import numpy as np
import pytest

from qopd.fidelity.metrics import (
    DEFAULT_ENTROPY_EDGES,
    compare,
    informativeness,
    rewards,
    topk_kl_position,
)
from qopd.types import Rollout, TeacherScores

K, VOCAB = 5, 50


def make_pool(n_roll=40, length=30, seed=0):
    rng = np.random.default_rng(seed)
    rollouts, ref = [], []
    for i in range(n_roll):
        ids = rng.integers(0, VOCAB, length).tolist()
        stud = (-rng.gamma(1.0, 1.0, length)).tolist()
        rollouts.append(Rollout([1, 2], ids, stud, problem_id=f"p{i}"))
        logits = rng.normal(0, 2, (length, VOCAB))
        lp = logits - np.log(np.exp(logits).sum(1, keepdims=True))
        top = np.argsort(-lp, axis=1)[:, :K]
        ref.append((lp, top, ids))
    return rollouts, ref


def scores_from(ref, noise=0.0, seed=1):
    rng = np.random.default_rng(seed)
    out = []
    for lp, _top, ids in ref:
        noisy = lp + noise * rng.normal(0, 1, lp.shape)
        noisy = noisy - np.log(np.exp(noisy).sum(1, keepdims=True))
        top = np.argsort(-noisy, axis=1)[:, :K]
        out.append(
            TeacherScores(
                [float(noisy[t, y]) for t, y in enumerate(ids)],
                top.tolist(),
                [[float(noisy[t, j]) for j in row] for t, row in enumerate(top)],
            )
        )
    return out


def test_identical_teachers_are_perfect():
    rollouts, ref = make_pool()
    s = scores_from(ref, 0.0)
    rep = compare(rollouts, s, s, rung="T16", student="S2")
    assert rep.pearson == pytest.approx(1.0)
    assert rep.mae == 0 and rep.topk_kl == pytest.approx(0, abs=1e-12)
    assert rep.top1_agreement == 1 and rep.sign_flip_rate == 0 and rep.topk_kl_imputed == 0
    assert rep.informativeness is None and rep.n_tokens == 40 * 30


def test_metrics_degrade_monotonically_with_noise():
    rollouts, ref = make_pool()
    base = scores_from(ref, 0.0)
    reps = [compare(rollouts, base, scores_from(ref, n)) for n in (0.05, 0.3, 1.0)]
    assert reps[0].pearson > reps[1].pearson > reps[2].pearson
    assert reps[0].mae < reps[1].mae < reps[2].mae
    assert reps[0].topk_kl < reps[1].topk_kl < reps[2].topk_kl
    assert reps[0].top1_agreement >= reps[1].top1_agreement >= reps[2].top1_agreement
    assert reps[0].sign_flip_rate <= reps[1].sign_flip_rate <= reps[2].sign_flip_rate


def test_mae_is_teacher_logprob_difference():
    rollouts, ref = make_pool(n_roll=3, length=4)
    a, b = scores_from(ref, 0.0), scores_from(ref, 0.5)
    expected = np.mean(
        [
            abs(x - y)
            for sa, sb in zip(a, b, strict=True)
            for x, y in zip(sa.token_logprobs, sb.token_logprobs, strict=True)
        ]
    )
    assert compare(rollouts, a, b).mae == pytest.approx(expected)


def test_topk_kl_position_and_imputation():
    lp = [math.log(0.5), math.log(0.3), math.log(0.2)]
    kl, imp = topk_kl_position([1, 2, 3], lp, [1, 2, 3], lp)
    assert kl == pytest.approx(0) and imp == 0
    kl2, imp2 = topk_kl_position(
        [1, 2, 3], lp, [1, 2, 9], [math.log(0.6), math.log(0.3), math.log(0.1)]
    )
    assert imp2 == 1 and kl2 > 0  # id 3 missing -> gets the rung's lowest top-k log-prob


def test_sign_flips_counted():
    r = Rollout([0], [5, 6], [-1.0, -1.0])
    ref = TeacherScores([-0.5, -2.0])  # rewards +0.5, -1.0
    cand = TeacherScores([-1.5, -0.5])  # rewards -0.5, +0.5 -> both flip
    rep = compare([r], [ref], [cand])
    assert rep.sign_flip_rate == 1.0 and math.isnan(rep.topk_kl)


def test_entropy_buckets_partition_tokens():
    rollouts, ref = make_pool(n_roll=10, length=20)
    rng = np.random.default_rng(5)
    ent = [rng.uniform(0, 3, 20).tolist() for _ in rollouts]
    rep = compare(rollouts, scores_from(ref), scores_from(ref, 0.3), entropies=ent)
    assert len(rep.by_entropy_bucket) == len(DEFAULT_ENTROPY_EDGES) - 1
    assert sum(b["n_tokens"] for b in rep.by_entropy_bucket.values()) == rep.n_tokens
    with pytest.raises(ValueError):
        compare(rollouts, scores_from(ref), scores_from(ref), entropies=[*ent[:-1], [0.0]])


def test_rewards_length_check_and_informativeness_blocked():
    with pytest.raises(ValueError):
        rewards([Rollout([0], [1, 2], [-1.0, -1.0])], [TeacherScores([-1.0])])
    with pytest.raises(NotImplementedError, match="PDF"):
        informativeness()
