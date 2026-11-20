"""PLAN 1.2: teacher-scoring input equals prompt_ids + response_ids byte for byte."""

import array

import pytest

from qopd.models.loading import load_tokenizer
from qopd.models.registry import resolve
from qopd.prompts.templates import (
    ASSISTANT_PREFIX,
    MATH_INSTRUCTION,
    MC_INSTRUCTION,
    TokenDriftError,
    check_teacher_input,
    format_messages,
    prompt_ids,
    prompt_text,
    retokenization_drift,
    teacher_input_ids,
)
from qopd.types import FinishReason, Problem, ProblemKind, Rollout
from qopd.utils import io

PROBLEMS = [
    Problem("t/0", "What is $1+1$?", "2", "test"),
    Problem("t/1", "Find all real $x$ with $x^2 = 4$.", "\\pm 2", "test"),
    Problem(
        "t/2",
        "Which is a noble gas?\nA) N\nB) Ne\nC) Na\nD) Ni",
        "B",
        "test",
        ProblemKind.MULTIPLE_CHOICE,
    ),
]


def as_bytes(ids):
    return array.array("q", ids).tobytes()


# --- tokenizer-free -----------------------------------------------------------


def test_instruction_by_kind():
    assert format_messages(PROBLEMS[0])[0]["content"].endswith(MATH_INSTRUCTION)
    assert format_messages(PROBLEMS[2])[0]["content"].endswith(MC_INSTRUCTION)
    assert format_messages(PROBLEMS[0], "X")[0]["content"].endswith("\n\nX")


def test_teacher_input_is_concatenation():
    r = Rollout([5, 6, 7], [9, 9, 10], [-0.1, -0.2, -0.3], FinishReason.STOP, "t/0")
    assert as_bytes(teacher_input_ids(r)) == as_bytes([5, 6, 7, 9, 9, 10])
    check_teacher_input(r, [5, 6, 7, 9, 9, 10])
    with pytest.raises(TokenDriftError, match="position 4"):
        check_teacher_input(r, [5, 6, 7, 9, 10, 10])
    with pytest.raises(TokenDriftError):
        check_teacher_input(r, [5, 6, 7, 9, 9])


def test_rollout_jsonl_round_trip_exact(tmp_path):
    rs = [
        Rollout(
            [151644, 872], [151667, 42, 151645], [-1.5, -0.25, -1e-6], FinishReason.STOP, "a", 0
        ),
        Rollout([1], [2, 3], [-0.5, -2.0], FinishReason.LENGTH, "b", 3),
    ]
    path = tmp_path / "pool.jsonl"
    io.write_jsonl(path, (r.to_dict() for r in rs))
    back = [Rollout.from_dict(d) for d in io.read_jsonl(path)]
    assert back == rs
    for a, b in zip(rs, back, strict=True):
        assert as_bytes(teacher_input_ids(a)) == as_bytes(teacher_input_ids(b))


# --- real Qwen3 tokenizer (skipped if not cached) -------------------------------


@pytest.fixture(scope="module")
def tok():
    try:
        return load_tokenizer(resolve("S2"), local_files_only=True)
    except OSError:
        pytest.skip("S2 tokenizer not cached (qopd models download S2)")


def test_thinking_prompt_shape(tok):
    for p in PROBLEMS:
        text = prompt_text(tok, p)
        assert text.startswith("<|im_start|>user\n")
        assert text.endswith(ASSISTANT_PREFIX)  # model opens <think> itself
        ids = prompt_ids(tok, p)
        assert tok.decode(ids, skip_special_tokens=False) == text
        assert all(isinstance(i, int) for i in ids)
    off = prompt_text(tok, PROBLEMS[0], enable_thinking=False)
    assert off.endswith("<think>\n\n</think>\n\n")


def test_teacher_input_byte_for_byte(tok):
    """Simulate generated responses (incl. non-canonical splits) and check the exact input."""
    eos = tok.convert_tokens_to_ids("<|im_end|>")
    think_open = tok.convert_tokens_to_ids("<think>")
    think_close = tok.convert_tokens_to_ids("</think>")
    body = tok.encode("\nLet me add: 1 + 1 = 2.\n", add_special_tokens=False)
    # character-level split of a word: valid ids, but not the canonical tokenization
    odd = [tok.convert_tokens_to_ids(c) for c in "answer"]
    answer = tok.encode("\n\nThe answer is \\boxed{2}.", add_special_tokens=False)
    responses = [
        [think_open, *body, think_close, *answer, eos],
        [think_open, *odd, think_close, *answer, eos],
        [think_open, *body],  # truncated (no EOS)
    ]
    for p in PROBLEMS:
        pids = prompt_ids(tok, p)
        for k, resp in enumerate(responses):
            r = Rollout(pids, resp, [-0.5] * len(resp), problem_id=p.id, sample_idx=k)
            scored = teacher_input_ids(r)
            assert as_bytes(scored) == as_bytes(pids) + as_bytes(resp)
            assert scored[r.response_slice] == resp
            check_teacher_input(r, scored)


def test_retokenizing_text_would_drift(tok):
    """Why we never re-tokenize: decode->encode does not preserve sampled ids."""
    odd = [tok.convert_tokens_to_ids(c) for c in "answer"]
    assert retokenization_drift(tok, odd)
    canonical = tok.encode("The answer is 2.", add_special_tokens=False)
    assert not retokenization_drift(tok, canonical)
