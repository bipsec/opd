"""PLAN 5.2: SFT trainer on a tiny random Qwen3 (CPU, float32)."""

import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from qopd.config import CONFIG_ROOT, SFTCfg, load  # noqa: E402
from qopd.train.sft import (  # noqa: E402
    IGNORE,
    Example,
    collate,
    load_examples,
    lr_lambda,
    response_token_loss,
    train_sft,
)
from qopd.utils import io  # noqa: E402
from qopd.utils.run import create_run_dir  # noqa: E402

V = 64


def tiny_model(seed=0):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(seed)
    cfg = Qwen3Config(
        vocab_size=V,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        max_position_embeddings=128,
        tie_word_embeddings=True,
    )
    return Qwen3ForCausalLM(cfg).float()


def examples(n=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        p = torch.randint(
            1, V, (int(torch.randint(2, 6, (1,), generator=g)),), generator=g
        ).tolist()
        r = [7, 8, 9, 10, 11, 12]  # same response every time -> learnable
        out.append(Example(p, r))
    return out


def cfg(**kw):
    base = load(CONFIG_ROOT / "train" / "sft.yaml", cls=SFTCfg)
    return base.model_copy(
        update={
            "gradient_checkpointing": False,
            "lr": 3e-3,
            "global_batch_size": 4,
            "micro_batch_size": 2,
            "epochs": 3,
            "checkpoint_every_steps": 2,
            **kw,
        }
    )


def test_collate_masks_prompt_and_padding():
    b = collate([Example([1, 2], [3, 4]), Example([5], [6])], pad_id=0)
    assert b["input_ids"].tolist() == [[1, 2, 3, 4], [5, 6, 0, 0]]
    assert b["attention_mask"].tolist() == [[1, 1, 1, 1], [1, 1, 0, 0]]
    assert b["labels"].tolist() == [[IGNORE, IGNORE, 3, 4], [IGNORE, 6, IGNORE, IGNORE]]


def test_load_examples_drops_overlong(tmp_path):
    p = tmp_path / "t.jsonl"
    io.write_jsonl(
        p,
        [
            {"prompt_ids": [1] * 5, "response_ids": [2] * 5},
            {"prompt_ids": [1] * 5, "response_ids": [2] * 6},
            {"prompt_ids": [1], "response_ids": []},
        ],
    )
    ex, dropped = load_examples(p, max_len=10)
    assert len(ex) == 1 and dropped == 1
    with pytest.raises(ValueError):
        load_examples(p, max_len=10, drop_overlong=False)


def test_lr_schedule_warmup_then_cosine_to_floor():
    f = lr_lambda(total_steps=100, warmup_ratio=0.03, min_lr_ratio=0.1)
    assert [round(f(s), 4) for s in range(3)] == [round(1 / 3, 4), round(2 / 3, 4), 1.0]
    assert f(3) == pytest.approx(1.0)
    assert f(99) == pytest.approx(0.1, abs=1e-3) and f(500) == pytest.approx(0.1)
    assert all(f(s) >= f(s + 1) for s in range(3, 99))


def test_response_token_loss_counts_only_response():
    logits = torch.zeros(1, 4, V)
    labels = torch.tensor([[IGNORE, IGNORE, 3, 4]])
    loss, n = response_token_loss(logits, labels)
    assert n == 2 and loss.item() == pytest.approx(2 * math.log(V))


def test_loss_decreases(tmp_path):
    run = create_run_dir("sft", "t", {"x": 1}, root=tmp_path)
    m = tiny_model()
    train_sft(cfg(epochs=10, lr=1e-2), run, m, pad_id=0, examples=examples(), device="cpu")
    losses = [r["loss"] for r in io.read_jsonl(run.metrics)]
    assert len(losses) == 30  # 10 epochs x ceil(12/4)
    assert losses[-1] < 0.5 * losses[0]


def test_microbatch_size_does_not_change_the_update(tmp_path):
    def final_params(mbs, sub):
        run = create_run_dir("sft", sub, {"x": 1}, root=tmp_path)
        m = tiny_model()
        train_sft(cfg(micro_batch_size=mbs, epochs=1), run, m, 0, examples(), device="cpu")
        return torch.cat([p.detach().flatten() for p in m.parameters()])

    assert torch.allclose(final_params(1, "a"), final_params(4, "b"), atol=1e-5)


def test_resume_matches_uninterrupted(tmp_path):
    straight = create_run_dir("sft", "straight", {"x": 1}, root=tmp_path)
    m1 = tiny_model()
    train_sft(cfg(), straight, m1, 0, examples(), device="cpu")

    split = create_run_dir("sft", "split", {"x": 1}, root=tmp_path)
    m2 = tiny_model()
    st = train_sft(cfg(), split, m2, 0, examples(), device="cpu", stop_after_steps=4)
    assert st["step"] == 4
    m3 = tiny_model(seed=123)  # fresh process: different init, must be overwritten by resume
    train_sft(cfg(), split, m3, 0, examples(), device="cpu")
    p1 = torch.cat([p.detach().flatten() for p in m1.parameters()])
    p3 = torch.cat([p.detach().flatten() for p in m3.parameters()])
    assert torch.allclose(p1, p3, atol=1e-5)
    assert [r["step"] for r in io.read_jsonl(split.metrics)] == list(range(1, 10))


def test_token_budget_compute_matching(tmp_path):
    run = create_run_dir("sft", "budget", {"x": 1}, root=tmp_path)
    ex = examples()  # 6 response tokens each, 4 per step -> 24 tokens per step
    st = train_sft(cfg(max_response_tokens=50), run, tiny_model(), 0, ex, device="cpu")
    assert st["stopped_by"] == "token_budget" and st["step"] == 3  # 24, 48, 72 >= 50
    assert st["trained_response_tokens"] == 72


@pytest.mark.gpu
@pytest.mark.slow
def test_s2_sft_smoke_gpu(tmp_path):
    """Real S2, 2 optimizer steps on 8 short traces (needs ~10+ GB free; run with make test-gpu)."""
    from qopd.data.smoke import smoke_problems
    from qopd.models.loading import load_hf_model, load_tokenizer
    from qopd.models.registry import resolve
    from qopd.prompts.templates import prompt_ids

    m = resolve("S2")
    tok = load_tokenizer(m)
    resp = tok.encode("<think>\nquick\n</think>\n\n\\boxed{1}", add_special_tokens=False)
    resp.append(tok.convert_tokens_to_ids("<|im_end|>"))
    ex = [Example(prompt_ids(tok, p), resp) for p in smoke_problems(8)]
    run = create_run_dir("sft", "s2smoke", {"x": 1}, root=tmp_path)
    model = load_hf_model(m, dtype="float32")  # trainers: FP32 master weights + BF16 autocast
    c = cfg(lr=1e-5, global_batch_size=4, micro_batch_size=2, epochs=1, gradient_checkpointing=True)
    st = train_sft(c, run, model, int(tok.pad_token_id), ex, device="cuda")
    rows = io.read_jsonl(run.metrics)
    assert st["step"] == 2 and all(r["torch_allocated_gb"] is not None for r in rows)
    assert all(math.isfinite(r["loss"]) for r in rows)


def test_save_hf_bf16_keeps_live_model_fp32(tmp_path):
    """Trainers keep FP32 master weights; checkpoints for vLLM are written in BF16."""
    import json

    from safetensors.torch import load_file

    from qopd.models.loading import save_hf_bf16, train_autocast

    class Tok:
        def save_pretrained(self, out):
            (Path(out) / "tok.json").write_text("{}")

    model = tiny_model()
    ref = {k: v.clone() for k, v in model.state_dict().items()}
    save_hf_bf16(model, Tok(), tmp_path / "ck")
    w = load_file(tmp_path / "ck" / "model.safetensors")
    assert w and all(v.dtype == torch.bfloat16 for v in w.values())
    cfg = json.loads((tmp_path / "ck" / "config.json").read_text())
    assert cfg.get("dtype", cfg.get("torch_dtype")) == "bfloat16"
    assert all(v.dtype == torch.float32 for v in model.state_dict().values())
    assert all(torch.equal(ref[k], v) for k, v in model.state_dict().items())
    with train_autocast(model):  # CPU model: no autocast, output stays FP32
        out = model(input_ids=torch.tensor([[1, 2, 3]])).logits
    assert out.dtype == torch.float32
