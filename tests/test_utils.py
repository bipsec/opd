import logging
import random

from qopd.utils import io
from qopd.utils.logging import get_logger, setup_logging
from qopd.utils.memory import snapshot, track_memory
from qopd.utils.seed import seed_everything


def test_jsonl_roundtrip(tmp_path):
    p = tmp_path / "x.jsonl"
    assert io.write_jsonl(p, [{"a": 1}, {"b": [1, 2]}]) == 2
    io.append_jsonl(p, {"c": "é"})
    assert io.read_jsonl(p) == [{"a": 1}, {"b": [1, 2]}, {"c": "é"}]


def test_json_atomic_and_hash(tmp_path):
    p = tmp_path / "x.json"
    io.write_json(p, {"k": 1})
    assert io.read_json(p) == {"k": 1}
    assert len(io.sha256_file(p)) == 64
    assert not list(tmp_path.glob(".x.json.*"))  # no temp files left
    assert p.stat().st_mode & 0o044  # readable by group/others (not mkstemp's 0600)


def test_seed_reproducible():
    seed_everything(123)
    a = random.random()
    seed_everything(123)
    assert random.random() == a


def test_logging_to_file(tmp_path):
    log = tmp_path / "stdout.log"
    setup_logging(log)
    get_logger("t").info("hello")
    for h in get_logger().handlers:
        h.flush()
    assert "hello" in log.read_text()
    # second call must not add a duplicate file handler
    setup_logging(log)
    n = sum(isinstance(h, logging.FileHandler) for h in get_logger().handlers)
    assert n == 1


def test_memory_snapshot_and_track(caplog):
    s = snapshot("x")
    assert s.mem_total_gb > 0 and s.proc_rss_gb > 0
    with track_memory("noop"):
        pass
