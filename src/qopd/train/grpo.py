"""RLVR baseline with GRPO, verifier reward only, wall-clock-matched (PLAN 8.2).

Each round: the next ``prompts_per_round`` prompts (same order as OPD), ``group_size`` rollouts
each from the rollout engine, a 0/1 verifier reward per rollout, group-normalized advantages
``A = (R - mean_g) / (std_g + eps)``, and ONE optimizer step on

    L = - sum_{b,t} m_bt * A_b * log p_S(y_bt) / N      (N = response tokens of the round)

Samples are fresh (on-policy) and there is one step per round, so the PPO ratio is 1 and
clipping never binds; there is no KL-to-reference term ("verifier reward only"). Groups whose
rewards are all equal carry no signal (A = 0). Sampling, optimizer and schedule are the OPD
settings. The run stops at ``wall_clock_hours`` of accumulated round time (resumable) or at
``max_rounds``, whichever comes first.
"""

from __future__ import annotations

import statistics
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from qopd.train.opd_loop import (
    LoopComponents,
    _pack,
    _response_logits,
    repetition_rate,
    request_seed,
    round_prompts,
    warmup_constant,
)
from qopd.train.opd_loss import gather_logprobs, token_entropy
from qopd.types import Rollout, TeacherScores
from qopd.utils.logging import get_logger
from qopd.utils.memory import PeakMemory

if TYPE_CHECKING:
    from qopd.config import GRPOCfg, OPDCfg
    from qopd.utils.run import RunDir

log = get_logger("train.grpo")


def group_advantages(
    rewards: Sequence[float],
    groups: Sequence[str],
    normalize_std: bool = True,
    eps: float = 1e-6,
) -> list[float]:
    """Per-rollout advantage, normalized within each group (rollouts of the same prompt)."""
    by: dict[str, list[int]] = defaultdict(list)
    for i, g in enumerate(groups):
        by[g].append(i)
    adv = [0.0] * len(rewards)
    for idx in by.values():
        rs = [rewards[i] for i in idx]
        mu = statistics.fmean(rs)
        sd = statistics.pstdev(rs) if len(rs) > 1 else 0.0
        for i in idx:
            a = rewards[i] - mu
            adv[i] = a / (sd + eps) if normalize_std else a
    return adv


def grpo_step(
    model: Any,
    optimizer: Any,
    rollouts: Sequence[Rollout],
    advantages: Sequence[float],
    pad_id: int,
    device: str,
    micro_batch_size: int,
    grad_clip: float | None,
) -> dict[str, float]:
    import torch  # noqa: PLC0415

    n_total = sum(len(r.response_ids) for r in rollouts)
    optimizer.zero_grad(set_to_none=True)
    loss_sum, ents = 0.0, []
    for j in range(0, len(rollouts), micro_batch_size):
        rs = rollouts[j : j + micro_batch_size]
        dummy = [TeacherScores([0.0] * len(r.response_ids)) for r in rs]
        pk = _pack(rs, dummy, pad_id, device, 0)
        logits = _response_logits(model, pk)
        lp = gather_logprobs(logits, pk["resp"])
        a = torch.tensor(advantages[j : j + micro_batch_size], dtype=torch.float32, device=device)
        loss = -(a[:, None] * lp * pk["mask"]).sum() / n_total
        loss.backward()
        loss_sum += float(loss.detach())
        with torch.no_grad():
            ents += token_entropy(logits)[pk["mask"].bool()].tolist()
    gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip or float("inf")))
    optimizer.step()
    return {
        "loss": loss_sum,
        "entropy_mean": statistics.fmean(ents),
        "grad_norm": gn,
        "response_tokens": n_total,
    }


def run_grpo(
    opd: OPDCfg,
    gcfg: GRPOCfg,
    run: RunDir,
    comps: LoopComponents,
    seed: int = 0,
    stop_after_rounds: int | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Run (or resume) GRPO; ``comps.teacher`` is unused, ``comps.verifier`` is required."""
    import torch  # noqa: PLC0415

    if comps.verifier is None:
        raise ValueError("GRPO needs a verifier")
    if opd.lr is None:
        raise ValueError("opd.lr is null")
    if gcfg.kl_coef:
        raise NotImplementedError("kl_coef > 0 is not part of the planned baseline")
    model = comps.model
    model.train()
    o = opd.optimizer
    opt = torch.optim.AdamW(
        model.parameters(), lr=opd.lr, betas=o.betas, eps=o.eps, weight_decay=o.weight_decay
    )
    sched = torch.optim.lr_scheduler.LambdaLR(opt, warmup_constant(opd.warmup_rounds))
    max_rounds = gcfg.max_rounds or opd.rounds
    budget_s = gcfg.wall_clock_hours * 3600 if gcfg.wall_clock_hours else float("inf")
    state_file = run.checkpoints / "trainer_state.pt"
    rnd, elapsed = 0, 0.0
    if state_file.exists():
        st = torch.load(state_file, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["optimizer"])
        sched.load_state_dict(st["scheduler"])
        rnd, elapsed = int(st["round"]), float(st["elapsed_s"])
        if st.get("engine_path"):
            comps.engine.restart(st["engine_path"])

    def checkpoint() -> None:
        if comps.save_model is not None:
            comps.save_model(model, run.checkpoints / f"round_{rnd:04d}")
        tmp = state_file.with_suffix(".tmp")
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict(),
                "round": rnd,
                "elapsed_s": elapsed,
                "engine_path": getattr(comps.engine, "model_path", None),
            },
            tmp,
        )
        tmp.replace(state_file)
        run.save_state(round=rnd, elapsed_s=elapsed, status="running")

    run.save_state(status="running", max_rounds=max_rounds, budget_s=budget_s)
    comps.engine.start()
    done = 0
    try:
        while rnd < max_rounds and elapsed < budget_s:
            if stop_after_rounds is not None and done >= stop_after_rounds:
                break
            t_round = clock()
            with PeakMemory() as pm:
                probs = round_prompts(comps.problems, rnd, opd.prompts_per_round, seed)
                rollouts, _ = comps.engine.generate(
                    [comps.prompt_ids(p) for p in probs],
                    opd.rollout,
                    n=gcfg.group_size,
                    problem_ids=[p.id for p in probs],
                    seeds=[request_seed(seed, rnd, p.id) for p in probs],
                )
                by_id = {p.id: p for p in probs}
                rewards = [
                    float(comps.verifier(by_id[r.problem_id], r.response_ids)) for r in rollouts
                ]
                adv = group_advantages(
                    rewards, [r.problem_id for r in rollouts], gcfg.normalize_std, gcfg.adv_eps
                )
                m = grpo_step(
                    model,
                    opt,
                    rollouts,
                    adv,
                    comps.pad_id,
                    comps.device,
                    opd.micro_batch_size or 1,
                    o.grad_clip_norm,
                )
                lr = sched.get_last_lr()[0]
                sched.step()
                comps.weight_sync.sync(comps.engine, model, rnd + 1)
            secs = clock() - t_round
            elapsed += secs
            rnd += 1
            done += 1
            zero_groups = sum(
                1
                for pid in by_id
                if len({rw for rw, r in zip(rewards, rollouts, strict=True) if r.problem_id == pid})
                == 1
            )
            run.log_metrics(
                {
                    "round": rnd,
                    "lr": lr,
                    **m,
                    "reward_mean": statistics.fmean(rewards),
                    "zero_signal_groups": zero_groups / len(by_id),
                    "response_len_mean": statistics.fmean(len(r.response_ids) for r in rollouts),
                    "truncation_rate": sum(r.truncated for r in rollouts) / len(rollouts),
                    "repetition_rate": repetition_rate([r.response_ids for r in rollouts]),
                    "round_seconds": secs,
                    "elapsed_s": elapsed,
                    "peak_mem_used_gb": pm.peak_used_gb,
                }
            )
            if rnd % opd.checkpoint_every == 0:
                checkpoint()
    finally:
        comps.engine.stop()
    checkpoint()
    finished = rnd >= max_rounds or elapsed >= budget_s
    run.save_state(
        status="trained" if finished else "running",
        stopped_by="wall_clock" if elapsed >= budget_s else "rounds",
    )
    return run.load_state()


def run_grpo_spec(spec_path: Path, stop_after_rounds: int | None = None) -> dict[str, Any]:
    """CLI entry (``qopd train grpo --spec``)."""
    from qopd.config import PROJECT_ROOT, DataCfg, ExperimentCfg, GRPOCfg, load  # noqa: PLC0415
    from qopd.data.sources import read_parquet  # noqa: PLC0415
    from qopd.eval.verifiers import verify  # noqa: PLC0415
    from qopd.models.loading import load_hf_model, load_tokenizer  # noqa: PLC0415
    from qopd.models.registry import resolve  # noqa: PLC0415
    from qopd.prompts.templates import prompt_ids as build_prompt_ids  # noqa: PLC0415
    from qopd.rollout.engine import ReloadFromCheckpoint, RolloutEngine  # noqa: PLC0415
    from qopd.utils.logging import setup_logging  # noqa: PLC0415
    from qopd.utils.monitor import Monitor  # noqa: PLC0415
    from qopd.utils.run import create_run_dir  # noqa: PLC0415

    exp = load(spec_path, cls=ExperimentCfg)
    if exp.opd is None:
        raise ValueError(f"{spec_path}: GRPO uses the opd section for sampling/optimizer")
    gcfg = exp.grpo or GRPOCfg()
    run = create_run_dir(exp.name, spec_path.stem, exp)
    if run.is_done:
        return run.load_state()
    setup_logging(run.stdout)
    student = resolve(exp.student)
    tok = load_tokenizer(student)
    data = exp.data or DataCfg()
    train = read_parquet(PROJECT_ROOT / data.data_dir / "splits" / "train.parquet")
    engine = RolloutEngine(
        student.path,
        student.revision,
        max_model_len=1024 + exp.opd.rollout.max_tokens,
        seed=exp.seeds[0],
    )
    model = load_hf_model(student)

    def save_model(m: Any, out: Path) -> None:
        m.save_pretrained(out, safe_serialization=True)
        tok.save_pretrained(out)

    comps = LoopComponents(
        model=model,
        engine=engine,
        teacher=None,
        weight_sync=ReloadFromCheckpoint(run.checkpoints / "sync", tok),
        problems=train,
        prompt_ids=lambda p: build_prompt_ids(tok, p, exp.opd.rollout.enable_thinking),
        pad_id=int(tok.pad_token_id),
        save_model=save_model,
        verifier=lambda p, ids: verify(p, tok.decode(ids, skip_special_tokens=True)).correct,
    )
    with Monitor(run.monitor):
        state = run_grpo(exp.opd, gcfg, run, comps, exp.seeds[0], stop_after_rounds)
    if state.get("status") == "trained":
        run.mark_done()
    return state
