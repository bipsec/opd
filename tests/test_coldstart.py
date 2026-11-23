"""PLAN 5.1: cold-start finalize keeps only verified-correct, finished traces (CPU)."""

from qopd.train.coldstart import done_ids, finalize
from qopd.utils import io


def test_finalize_filters_and_counts(tmp_path):
    rows = [
        {
            "id": "a",
            "correct": True,
            "reason": "correct",
            "finish_reason": "stop",
            "n_tokens": 3,
            "prompt_ids": [1, 2],
            "response_ids": [3, 4, 5],
        },
        {
            "id": "b",
            "correct": False,
            "reason": "wrong",
            "finish_reason": "stop",
            "n_tokens": 2,
            "prompt_ids": [1],
            "response_ids": [6, 7],
        },
        {
            "id": "c",
            "correct": False,
            "reason": "no_final_answer",
            "finish_reason": "length",
            "n_tokens": 9,
            "prompt_ids": [1],
            "response_ids": [8] * 9,
        },
    ]
    io.write_jsonl(tmp_path / "all.jsonl", rows)
    st = finalize(tmp_path, max_total=10)
    assert (st["n_prompts"], st["n_kept"]) == (3, 1)
    assert st["outcomes"] == {"dropped:wrong": 1, "kept": 1, "reached_limit": 1}
    kept = io.read_jsonl(tmp_path / "traces.jsonl")
    assert kept == [{"id": "a", "prompt_ids": [1, 2], "response_ids": [3, 4, 5]}]
    assert done_ids(tmp_path / "all.jsonl") == {"a", "b", "c"}
