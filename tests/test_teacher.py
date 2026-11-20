"""Teacher backends: CPU unit tests for parsing/metrics; GPU test for vLLM-vs-HF agreement."""

from dataclasses import dataclass

import pytest

from qopd.teacher.agreement import THRESHOLD_NATS, agreement_metrics, run_agreement
from qopd.teacher.vllm_backend import scores_from_prompt_logprobs
from qopd.types import TeacherScores


@dataclass
class FakeLogprob:
    logprob: float
    rank: int | None


def test_scores_from_prompt_logprobs_slices_response():
    # input: prompt [10, 11] + response [12, 13]; position 0 has no logprob (None)
    input_ids = [10, 11, 12, 13]
    plp = [
        None,
        {11: FakeLogprob(-0.1, 1), 99: FakeLogprob(-3.0, 2)},
        {12: FakeLogprob(-0.5, 1), 7: FakeLogprob(-1.5, 2)},  # realized token is top-1
        {
            5: FakeLogprob(-0.2, 1),
            6: FakeLogprob(-2.0, 2),
            13: FakeLogprob(-4.0, 9),
        },  # outside top-2
    ]
    s = scores_from_prompt_logprobs(plp, input_ids, n_prompt=2, top_k=2)
    assert s.token_logprobs == [-0.5, -4.0]
    assert s.topk_ids == [[12, 7], [5, 6]]
    assert s.topk_logprobs == [[-0.5, -1.5], [-0.2, -2.0]]


def test_missing_realized_token_raises():
    plp = [None, {5: FakeLogprob(-0.2, 1)}]
    with pytest.raises(RuntimeError, match="realized token"):
        scores_from_prompt_logprobs(plp, [1, 2], n_prompt=1, top_k=1)


def test_agreement_metrics():
    a = [TeacherScores([-1.0, -2.0], [[1, 2], [3, 4]], [[-1.0, -2.0], [-2.0, -3.0]])]
    b = [TeacherScores([-1.01, -2.03], [[1, 2], [4, 3]], [[-1.0, -2.0], [-2.0, -3.0]])]
    m = agreement_metrics(a, b)
    assert m["n_tokens"] == 2
    assert m["mean_abs_diff"] == pytest.approx(0.02)
    assert m["max_abs_diff"] == pytest.approx(0.03)
    assert m["top1_agreement"] == 0.5
    assert m["topk_overlap"] == 1.0


@pytest.mark.gpu
def test_vllm_matches_hf_on_s2(tmp_path):
    summary = run_agreement("S2", n=20, max_tokens=256, top_k=20, seed=0, root=tmp_path)
    assert summary["n_sequences"] == 20
    assert summary["mean_abs_diff"] < THRESHOLD_NATS, summary
