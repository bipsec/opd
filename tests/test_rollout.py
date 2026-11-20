"""Rollout engine: CPU tests for helpers; GPU tests for generation and weight sync (S2)."""

import pytest

from qopd.config import CONFIG_ROOT, OPDCfg, SamplingCfg, load
from qopd.data.smoke import SMOKE_PROBLEMS, smoke_problems
from qopd.rollout.bench import run_bench, validate_rollouts
from qopd.rollout.engine import ReloadFromCheckpoint, RolloutEngine, _finish_reason
from qopd.types import FinishReason, Rollout


def test_finish_reason_mapping():
    assert _finish_reason("length") is FinishReason.LENGTH
    assert _finish_reason("stop") is FinishReason.STOP
    assert _finish_reason("abort") is FinishReason.ABORT


def test_smoke_problems():
    ps = smoke_problems(8)
    assert len(ps) == 8 and ps[0].problem == SMOKE_PROBLEMS[0]
    with pytest.raises(ValueError):
        smoke_problems(len(SMOKE_PROBLEMS) + 1)


def test_validate_rollouts():
    good = Rollout([1, 2], [5, 6], [-0.1, -0.2])
    assert set(validate_rollouts([good], [[1, 2]], vocab=10).values()) == {0}
    bad = Rollout([1, 3], [50], [0.5])
    v = validate_rollouts([bad], [[1, 2]], vocab=10)
    assert v["ids_out_of_tokenizer_range"] == 1
    assert v["logprob_problems"] == 1
    assert v["prompt_mismatch"] == 1


# --- GPU ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_s2_generate_8x2(tmp_path):
    summary = run_bench("S2", n_prompts=8, n=2, max_tokens=256, root=tmp_path)
    assert summary["n_sequences"] == 16
    assert summary["valid"], summary
    assert summary["tokens_per_s"] > 0


@pytest.mark.gpu
def test_reload_from_checkpoint_roundtrip(tmp_path):
    import torch

    from qopd.models.loading import load_hf_model, load_tokenizer
    from qopd.models.registry import resolve
    from qopd.prompts.templates import prompt_ids

    cfg = resolve("S2")
    tok = load_tokenizer(cfg)
    prompts = [prompt_ids(tok, p) for p in smoke_problems(4)]
    greedy = SamplingCfg(temperature=0.0, max_tokens=48)
    engine = RolloutEngine(cfg.path, cfg.revision, kv_cache_gb=2.0, max_model_len=512)
    engine.start()
    try:
        before, _ = engine.generate(prompts, greedy)
        model = load_hf_model(cfg)
        sync = ReloadFromCheckpoint(tmp_path / "ckpt", tok)
        sync.sync(engine, model, step=0)  # unchanged weights
        same, _ = engine.generate(prompts, greedy)
        assert [r.response_ids for r in same] == [r.response_ids for r in before]
        with torch.no_grad():
            model.model.norm.weight.mul_(1.5)  # perturb: outputs must change
        sync.sync(engine, model, step=1)
        changed, _ = engine.generate(prompts, greedy)
        assert [r.response_ids for r in changed] != [r.response_ids for r in before] or any(
            abs(a - b) > 1e-3
            for x, y in zip(changed, before, strict=True)
            for a, b in zip(x.sampled_logprobs, y.sampled_logprobs, strict=False)
        )
        assert len(list((tmp_path / "ckpt").glob("sync_*"))) == 1  # keep_last=1
    finally:
        engine.stop()


def test_opd_rollout_settings_are_on_policy():
    s = load(CONFIG_ROOT / "train" / "opd.yaml", cls=OPDCfg).rollout
    assert (s.temperature, s.top_p, s.top_k) == (1.0, 1.0, -1)
