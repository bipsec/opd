"""Cold-start and off-policy SFT (PLAN 5.2, baseline in 8.2).

Settings (configs/train/sft.yaml, from the experiment plan): full fine-tune, AdamW (betas
0.9/0.999, eps 1e-8, weight decay 0), BF16, gradient checkpointing, lr 1e-5, 1 epoch, 32
sequences per optimizer step, linear warmup over 3% of steps then cosine decay to 10% of peak,
gradient clipping at norm 1.0, loss = mean over all response tokens of the optimizer batch,
prompt + response at most 8,192 tokens (longer traces are dropped, not truncated), no packing.

Input: JSONL with exact ``prompt_ids`` and ``response_ids`` per line (written by step 5.1), so
nothing is re-tokenized. Resumable: model, optimizer, scheduler and the step counter are saved
every ``checkpoint_every_steps``; the data order is a fixed permutation (``seed``) so a resumed
run sees exactly the batches an uninterrupted run would. Memory is logged at every step.
"""

from __future__ import annotations

import math
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from qopd.utils import io
from qopd.utils.logging import get_logger
from qopd.utils.memory import snapshot

if TYPE_CHECKING:
    import torch

    from qopd.config import SFTCfg
    from qopd.utils.run import RunDir

log = get_logger("train.sft")
IGNORE = -100


@dataclass(frozen=True)
class Example:
    prompt_ids: list[int]
    response_ids: list[int]

    @property
    def length(self) -> int:
        return len(self.prompt_ids) + len(self.response_ids)


def load_examples(
    path: Path, max_len: int, drop_overlong: bool = True
) -> tuple[list[Example], int]:
    """Read traces; returns (examples, number dropped as too long)."""
    out, dropped = [], 0
    for row in io.iter_jsonl(path):
        ex = Example([int(i) for i in row["prompt_ids"]], [int(i) for i in row["response_ids"]])
        if not ex.response_ids:
            continue
        if ex.length > max_len:
            if drop_overlong:
                dropped += 1
                continue
            raise ValueError(f"trace of {ex.length} tokens exceeds max_seq_len {max_len}")
        out.append(ex)
    return out, dropped


def collate(batch: Sequence[Example], pad_id: int) -> dict[str, torch.Tensor]:
    """Right-padded input_ids / attention_mask / labels (prompt and padding = -100)."""
    import torch  # noqa: PLC0415

    width = max(e.length for e in batch)
    ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    att = torch.zeros((len(batch), width), dtype=torch.long)
    lab = torch.full((len(batch), width), IGNORE, dtype=torch.long)
    for i, e in enumerate(batch):
        seq = e.prompt_ids + e.response_ids
        ids[i, : len(seq)] = torch.tensor(seq)
        att[i, : len(seq)] = 1
        lab[i, len(e.prompt_ids) : len(seq)] = torch.tensor(e.response_ids)
    return {"input_ids": ids, "attention_mask": att, "labels": lab}


def lr_lambda(total_steps: int, warmup_ratio: float, min_lr_ratio: float) -> Callable[[int], float]:
    """Linear warmup over ``warmup_ratio`` of the steps, then cosine to ``min_lr_ratio``."""
    warm = max(1, math.ceil(warmup_ratio * total_steps)) if warmup_ratio > 0 else 0

    def f(step: int) -> float:  # step = number of optimizer steps already taken
        if warm and step < warm:
            return (step + 1) / warm
        progress = (step - warm) / max(1, total_steps - warm)
        progress = min(max(progress, 0.0), 1.0)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return f


def response_token_loss(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Summed next-token cross-entropy over response tokens (float32) and their count."""
    import torch  # noqa: PLC0415

    shift_logits = logits[:, :-1].float()
    shift_labels = labels[:, 1:]
    loss = torch.nn.functional.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        ignore_index=IGNORE,
        reduction="sum",
    )
    return loss, int((shift_labels != IGNORE).sum())


def _order(n: int, seed: int) -> list[int]:
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    return idx


def train_sft(
    cfg: SFTCfg,
    run: RunDir,
    model: Any,
    pad_id: int,
    examples: list[Example],
    save_final: Callable[[Any, Path], None] | None = None,
    device: str = "cuda",
    stop_after_steps: int | None = None,  # testing / time budgets: stop early, stay resumable
) -> dict[str, Any]:
    """Train ``model`` in place; returns the final state. Resumes from ``run`` if possible."""
    import torch  # noqa: PLC0415

    from qopd.models.loading import train_autocast  # noqa: PLC0415

    if not examples:
        raise ValueError("no training examples")
    gbs = cfg.global_batch_size
    mbs = cfg.micro_batch_size or 1
    order = _order(len(examples), cfg.seed)
    steps_per_epoch = math.ceil(len(order) / gbs)
    total_steps = math.ceil(cfg.epochs * steps_per_epoch)

    if cfg.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False
    model.train()
    o = cfg.optimizer
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, betas=o.betas, eps=o.eps, weight_decay=o.weight_decay
    )
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda(total_steps, cfg.warmup_ratio, cfg.min_lr_ratio)
    )

    ckpt = run.checkpoints / "last.pt"
    step = 0
    if ckpt.exists():
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        step = int(state["step"])
        log.info("resumed from %s at step %d", ckpt, step)

    def save_state() -> None:
        tmp = ckpt.with_suffix(".tmp")
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict(),
                "step": step,
            },
            tmp,
        )
        tmp.replace(ckpt)
        run.save_state(step=step, total_steps=total_steps, status="running")

    run.save_state(status="running", total_steps=total_steps, n_examples=len(examples))
    taken = 0
    trained_tokens = int(run.load_state().get("trained_response_tokens", 0))
    budget = cfg.max_response_tokens
    while step < total_steps and (budget is None or trained_tokens < budget):
        if stop_after_steps is not None and taken >= stop_after_steps:
            save_state()
            return run.load_state()
        t0 = time.perf_counter()
        start = (step % steps_per_epoch) * gbs
        batch = [examples[i] for i in order[start : start + gbs]]
        n_tok = sum(len(e.response_ids) for e in batch)  # global normalizer
        opt.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for j in range(0, len(batch), mbs):
            b = {k: v.to(device) for k, v in collate(batch[j : j + mbs], pad_id).items()}
            with train_autocast(model):
                logits = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits
            loss, _ = response_token_loss(logits, b["labels"])
            (loss / n_tok).backward()
            loss_sum += float(loss.detach())
        clip = o.grad_clip_norm if o.grad_clip_norm is not None else float("inf")
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), clip))
        lr = sched.get_last_lr()[0]
        opt.step()
        sched.step()
        step += 1
        taken += 1
        trained_tokens += n_tok
        run.save_state(trained_response_tokens=trained_tokens)
        mem = snapshot()
        run.log_metrics(
            {
                "step": step,
                "loss": loss_sum / n_tok,
                "lr": lr,
                "grad_norm": grad_norm,
                "response_tokens": n_tok,
                "sequences": len(batch),
                "seconds": time.perf_counter() - t0,
                "mem_used_gb": mem.mem_used_gb,
                "torch_allocated_gb": mem.torch_allocated_gb,
            }
        )
        log.info(
            "step %d/%d loss %.4f lr %.2e grad_norm %.3f tokens %d %.1fs mem %.1f GB",
            step,
            total_steps,
            loss_sum / n_tok,
            lr,
            grad_norm,
            n_tok,
            time.perf_counter() - t0,
            mem.mem_used_gb,
        )
        if (
            step % cfg.checkpoint_every_steps == 0
            or step == total_steps
            or (budget is not None and trained_tokens >= budget)
        ):
            save_state()
    if save_final is not None:
        save_final(model, run.checkpoints / "final")
    run.save_state(
        status="trained",
        step=step,
        total_steps=total_steps,
        stopped_by="token_budget" if budget is not None and trained_tokens >= budget else "epochs",
    )
    return run.load_state()


def run_sft(spec_path: Path) -> Path:
    """CLI entry (``qopd train sft --spec``): load model + traces, train, save, mark DONE."""
    from qopd.config import PROJECT_ROOT as ROOT  # noqa: PLC0415
    from qopd.config import ExperimentCfg, load  # noqa: PLC0415
    from qopd.models.loading import load_hf_model, load_tokenizer, save_hf_bf16  # noqa: PLC0415
    from qopd.models.registry import resolve  # noqa: PLC0415
    from qopd.utils.logging import setup_logging  # noqa: PLC0415
    from qopd.utils.monitor import Monitor  # noqa: PLC0415
    from qopd.utils.run import create_run_dir  # noqa: PLC0415

    exp = load(spec_path, cls=ExperimentCfg)
    if exp.sft is None:
        raise ValueError(f"{spec_path} has no sft section")
    run = create_run_dir(exp.name, spec_path.stem, exp)
    if run.is_done:
        return run.checkpoints / "final"
    setup_logging(run.stdout)
    mcfg = resolve(exp.student)
    tok = load_tokenizer(mcfg)
    traces = Path(exp.sft.traces)
    examples, dropped = load_examples(
        traces if traces.is_absolute() else ROOT / traces,
        exp.sft.max_seq_len,
        exp.sft.drop_overlong,
    )
    run.save_state(
        dropped_overlong=dropped,
        traces_sha256=io.sha256_file(traces if traces.is_absolute() else ROOT / traces),
    )
    model = load_hf_model(mcfg, dtype="float32")

    def save_final(m: Any, out: Path) -> None:
        save_hf_bf16(m, tok, out)

    with Monitor(run.monitor):
        train_sft(exp.sft, run, model, int(tok.pad_token_id), examples, save_final)
    run.mark_done()
    return run.checkpoints / "final"
