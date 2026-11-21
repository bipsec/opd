"""PLAN 4.1: calibration records are cut at the token limit and stats are written (CPU)."""

from qopd.quant.calibration import make_records, write_stats
from qopd.types import FinishReason, Problem, Rollout
from qopd.utils import io


class Tok:
    def decode(self, ids, skip_special_tokens=False):
        return " ".join(map(str, ids))


def test_records_cut_at_limit_and_stats(tmp_path):
    probs = [Problem("c/0", "q", "1", "src"), Problem("c/1", "q", "2", "src")]
    rs = [
        Rollout([1, 2, 3], list(range(10, 20)), [], FinishReason.LENGTH, "c/0"),
        Rollout([4, 5], [6, 7], [], FinishReason.STOP, "c/1"),
    ]
    recs = make_records(probs, rs, Tok(), max_total=8)
    assert len(recs[0]["prompt_ids"]) + len(recs[0]["response_ids"]) == 8
    assert recs[0]["text"] == "1 2 3 10 11 12 13 14" and recs[1]["response_ids"] == [6, 7]
    path = tmp_path / "t16_traces.jsonl"
    io.write_jsonl(path, recs)
    st = write_stats(recs, path, 8)
    assert (st["n"], st["reached_limit"], st["total_tokens_max"]) == (2, 1, 8)
    assert (tmp_path / "calibration_stats.json").exists()
