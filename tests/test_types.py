import pytest

from qopd.models.registry import Backend, ModelId, resolve
from qopd.teacher.base import check_scores
from qopd.types import FinishReason, Rollout, TeacherScores


def test_rollout_input_ids_exact():
    r = Rollout(prompt_ids=[1, 2, 3], response_ids=[7, 8], sampled_logprobs=[-0.1, -0.2])
    assert r.input_ids == [1, 2, 3, 7, 8]
    assert r.input_ids[r.response_slice] == [7, 8]
    assert not r.truncated
    assert Rollout([1], [2], [-1.0], FinishReason.LENGTH).truncated


def test_rollout_length_mismatch():
    with pytest.raises(ValueError):
        Rollout(prompt_ids=[1], response_ids=[2, 3], sampled_logprobs=[-0.1])


def test_check_scores():
    r = Rollout([1], [2, 3], [-0.1, -0.2])
    ok = TeacherScores([-0.3, -0.4], [[1, 2], [3, 4]], [[-0.1, -2.0], [-0.2, -3.0]])
    check_scores([r], [ok], top_k=2)
    with pytest.raises(ValueError):
        check_scores([r], [TeacherScores([-0.3])], top_k=0)


def test_registry():
    assert resolve("S2").source == "Qwen/Qwen3-0.6B"
    assert resolve("T4-int").base is ModelId.T16
    assert resolve(ModelId.T2).backend is Backend.GGUF
    with pytest.raises(KeyError):
        resolve("T5")
