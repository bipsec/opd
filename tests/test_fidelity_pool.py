"""PLAN 7.1/7.2: rollout pool, student stats, chunked resumable scoring, fidelity (CPU)."""

import math

import pytest

torch = pytest.importorskip("torch")

from qopd.fidelity.pool import (  # noqa: E402
    build_pool,
    fidelity_for,
    load_pool,
    load_scores,
    pool_for_metrics,
    score_pool,
    student_stats,
    write_student_stats,
)
from tests.test_opd_loop import PROBLEMS, FakeEngine, FakeTeacher, V, cfg, prompt_of, tiny  # noqa: E402


class CountingTeacher(FakeTeacher):
    def __init__(self, seed=99):
        super().__init__(seed)
        self.calls = 0

    def score(self, rollouts, top_k):
        self.calls += 1
        return super().score(rollouts, top_k)


@pytest.fixture
def pool(tmp_path):
    student = tiny(0)
    d = tmp_path / "pool_S2"
    build_pool(FakeEngine(student), PROBLEMS[:6], prompt_of, cfg().rollout, 2, d, seed=0)
    rs = load_pool(d)
    lp, ent = student_stats(student, rs, pad_id=0, device="cpu", micro_batch_size=4)
    write_student_stats(d, rs, lp, ent)
    return d, rs, lp, ent


def test_pool_built_once_with_exact_ids(pool):
    d, rs, _, _ = pool
    assert len(rs) == 12 and {r.sample_idx for r in rs} == {0, 1}
    before = (d / "rollouts.jsonl").stat().st_mtime_ns
    build_pool(None, PROBLEMS[:6], prompt_of, None, 2, d)  # exists: engine never touched
    assert (d / "rollouts.jsonl").stat().st_mtime_ns == before


def test_student_stats_match_engine_and_bound_entropy(pool):
    _, rs, lp, ent = pool
    engine_lp = [x for r in rs for x in r.sampled_logprobs]
    assert lp.tolist() == pytest.approx(engine_lp, abs=1e-4)  # same weights -> same log-probs
    assert (ent >= 0).all() and (ent <= math.log(V) + 1e-5).all()
    fixed, ents = pool_for_metrics(pool[0])
    assert len(ents) == len(fixed) and len(ents[0]) == len(fixed[0].response_ids)


def test_chunked_scoring_resumes(pool):
    d, rs, *_ = pool
    t = CountingTeacher()
    out = score_pool(t, rs, top_k=4, out_dir=d / "scores" / "T16", chunk_size=5)
    assert t.calls == 3 and (out / "DONE").exists()
    direct = FakeTeacher()
    direct.start()
    want = direct.score(rs, 4)
    got = load_scores(out)
    flat_got = [x for s in got for x in s.token_logprobs]
    flat_want = [x for s in want for x in s.token_logprobs]
    assert flat_got == pytest.approx(flat_want, abs=1e-6)
    assert [s.topk_ids for s in got] == [s.topk_ids for s in want]
    (out / "DONE").unlink()
    (out / "chunk_0001.npz").unlink()  # simulate a crash mid-scoring
    t2 = CountingTeacher()
    score_pool(t2, rs, top_k=4, out_dir=out, chunk_size=5)
    assert t2.calls == 1  # only the missing chunk is rescored
    assert len(load_scores(out)) == len(rs)


def test_fidelity_end_to_end(pool):
    d, rs, *_ = pool
    score_pool(FakeTeacher(99), rs, 4, d / "scores" / "T16")
    score_pool(FakeTeacher(99), rs, 4, d / "scores" / "T8")  # identical teacher
    score_pool(FakeTeacher(5), rs, 4, d / "scores" / "T3")  # different teacher
    same = fidelity_for(d, "T8", student="S2")
    other = fidelity_for(d, "T3", student="S2")
    assert same.pearson == pytest.approx(1.0) and same.mae == pytest.approx(0.0, abs=1e-6)
    assert other.mae > 0.1 and other.pearson < 0.99 and other.sign_flip_rate > 0
    assert sum(b["n_tokens"] for b in other.by_entropy_bucket.values()) == other.n_tokens
