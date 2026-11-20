"""PLAN 2.3: 13-gram contamination filter (CPU)."""

from qopd.data.contamination import (
    build_index,
    drop_contaminated,
    find_contamination,
    ngrams,
    tokens,
)
from qopd.types import Problem

EVAL = (
    "Let $x$ and $y$ be real numbers such that $x + y = 10$ and $xy = 21$. "
    "Find the value of $x^2 + y^2$ and explain every step of the computation."
)


def P(pid, text, src="train"):
    return Problem(pid, text, "1", src)


def test_tokens_ignore_markup():
    assert tokens("$x^2$ + Y") == tokens("x^2+y") == ["x", "2", "y"]


def test_ngrams_short_text_empty():
    assert ngrams("only five words in here", 13) == set()
    assert len(ngrams(" ".join(f"w{i}" for i in range(15)), 13)) == 3


def test_overlap_detected_despite_formatting():
    ev = [P("math500/0001", EVAL, "math500")]
    reformatted = EVAL.replace("$", "").replace("  ", " ").upper()
    train = [
        P("t/0", "Unrelated: how many primes are below one hundred, counting carefully each one?"),
        P("t/1", "Warm-up. " + reformatted),  # contaminated, with different markup and case
        P("t/2", "Let x and y be real numbers such that x + y = 10."),  # < 13 shared tokens
    ]
    hits = find_contamination(train, ev, n=13)
    assert {h.train_id for h in hits} == {"t/1"}
    assert hits[0].eval_id == "math500/0001" and hits[0].n_shared > 1
    kept, _ = drop_contaminated(train, hits)
    assert [p.id for p in kept] == ["t/0", "t/2"]


def test_index_counts_short_items():
    idx = build_index([P("e/0", "too short", "s"), P("e/1", EVAL, "s")], n=13)
    assert idx.too_short == {"s": 1} and idx.n_items == {"s": 2}


def test_one_hit_per_pair():
    ev = [P("a/0", EVAL, "a"), P("b/0", EVAL, "b")]
    hits = find_contamination([P("t/0", EVAL)], ev)
    assert sorted(h.eval_id for h in hits) == ["a/0", "b/0"]


def test_decontaminate_end_to_end(tmp_path):
    from qopd.config import DataCfg
    from qopd.data.pool import decontaminate
    from qopd.data.sources import read_parquet, write_parquet

    cfg = DataCfg(data_dir=str(tmp_path))
    pool = [
        P("pool/0", "Completely unrelated question about prime gaps below fifty " * 2),
        P("pool/1", EVAL),
        P("pool/2", "Compute 2+2."),
    ]
    write_parquet(pool, tmp_path / "pool" / "pool.parquet")
    write_parquet(
        [P("math500/0000", EVAL, "math500"), P("math500/0001", "Compute 2+2.", "math500")],
        tmp_path / "eval" / "math500.parquet",
    )
    st = decontaminate(cfg, report=tmp_path / "data.md")
    assert st["pool_in"] == 3 and st["pool_out"] == 2
    assert [p.id for p in read_parquet(tmp_path / "pool" / "pool_clean.parquet")] == [
        "pool/0",
        "pool/2",
    ]
    m = st["per_suite"]["math500"]
    assert (m["eval_items_hit"], m["train_dropped"], m["near_copy_train"], m["too_short"]) == (
        1,
        1,
        1,
        1,
    )
    assert st["short_exact_matches"] == [{"train_id": "pool/2", "eval_id": "math500/0001"}]
    assert "gpqa_diamond" in st["suites_not_checked"]
    assert "## Contamination filter (step 2.3)" in (tmp_path / "data.md").read_text()
