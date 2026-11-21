"""OPD round loop (PLAN 6.2): sample -> score -> one optimizer step -> sync weights.

Round r:
1. prompts: positions ``r*P .. r*P+P-1`` of a fixed seeded permutation of the training split
   (P = prompts_per_round; no prompt repeats within the first pass);
2. sample ``rollouts_per_prompt`` responses per prompt with the rollout engine (exact ids);
3. score every response token with the teacher (log-prob of the realized token + top-k);
4. recompute the student's log-probs with the trainer (same weights the rollouts came from),
   form r_t, and take ONE optimizer step on the OPD loss averaged over all response tokens
   of the round (micro-batched with a global normalizer);
5. sync the new weights to the rollout engine.

Modes: ``co_resident`` keeps engine and teacher loaded; ``phase_alternating`` starts each of
them only for its stage (load -> use -> unload) so the teacher and the engine never share
memory. The trainer model stays loaded in both modes.

Per round, ``metrics.jsonl`` gets reward mean/std/percentiles, student entropy, response
length, truncation rate, repetition rate, rollout verifier accuracy (if a verifier is given),
student log-prob mismatch (engine vs trainer), seconds per stage and peak memory; the 1 Hz
``monitor.csv`` has power. Every ``checkpoint_every`` rounds (and at the end) the model is saved
in HF format (``checkpoints/round_XXXX``) and the full trainer state (model, optimizer,
scheduler, round) to ``checkpoints/trainer_state.pt``; a restarted run resumes from it.
"""

from __future__ import annotations

import hashlib
import math
import random
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from qopd.config import LoopMode
from qopd.train.opd_loss import OPDLossCfg, gather_logprobs, opd_loss, token_entropy
from qopd.types import Problem, Rollout, TeacherScores
from qopd.utils.logging import get_logger
from qopd.utils.memory import PeakMemory

if TYPE_CHECKING:
    import torch

    from qopd.config import OPDCfg
    from qopd.utils.run import RunDir

log = get_logger("train.opd")


def repetition_rate(responses: Sequence[Sequence[int]], n: int = 8) -> float:
    """Share of responses "ending in a repeated n-gram": the last n tokens also occur as a
    contiguous n-gram earlier in the same response (step 6.2 definition; catches looping)."""
    if not responses:
        return float("nan")

    def ends_repeated(r: Sequence[int]) -> bool:
        if len(r) < 2 * n:
            return False
        tail = tuple(r[-n:])
        return any(tuple(r[i : i + n]) == tail for i in range(len(r) - 2 * n + 1))

    return sum(ends_repeated(r) for r in responses) / len(responses)


def percentiles(
    values: Sequence[float], qs: Sequence[int] = (5, 25, 50, 75, 95)
) -> dict[str, float]:
    if not values:
        return {f"p{q}": float("nan") for q in qs}
    s = sorted(values)
    return {f"p{q}": s[min(len(s) - 1, max(0, math.ceil(q / 100 * len(s)) - 1))] for q in qs}


def request_seed(base: int, rnd: int, pid: str) -> int:
    return int(hashlib.sha256(f"{base}:{rnd}:{pid}".encode()).hexdigest()[:8], 16) & 0x7FFFFFFF


def round_prompts(
    problems: Sequence[Problem], rnd: int, per_round: int, seed: int
) -> list[Problem]:
    order = sorted(problems, key=lambda p: p.id)
    random.Random(seed).shuffle(order)
    n = len(order)
    return [order[(rnd * per_round + i) % n] for i in range(per_round)]


def warmup_constant(warmup_rounds: int) -> Callable[[int], float]:
    def f(step: int) -> float:
        return (step + 1) / warmup_rounds if warmup_rounds and step < warmup_rounds else 1.0

    return f


@dataclass
class LoopComponents:
    """Everything the loop needs; real implementations or test fakes."""

    model: Any  # HF causal LM being trained (stays loaded)
    engine: Any  # RolloutEngine-like: start/stop/running/generate/restart/model_path
    teacher: Any  # TeacherScorer-like: start/stop/score
    weight_sync: Any  # sync(engine, model, step) -> seconds
    problems: list[Problem]
    prompt_ids: Callable[[Problem], list[int]]
    pad_id: int
    save_model: Callable[[Any, Path], None] | None = None
    verifier: Callable[[Problem, list[int]], bool] | None = None  # needed for hybrid
    device: str = "cuda"


def _pack(
    rollouts: Sequence[Rollout],
    scores: Sequence[TeacherScores],
    pad_id: int,
    device: str,
    top_k: int,
) -> dict[str, torch.Tensor]:
    import torch  # noqa: PLC0415

    width = max(len(r.input_ids) for r in rollouts)
    t_max = max(len(r.response_ids) for r in rollouts)
    b = len(rollouts)
    ids = torch.full((b, width), pad_id, dtype=torch.long)
    att = torch.zeros((b, width), dtype=torch.long)
    resp = torch.full((b, t_max), pad_id, dtype=torch.long)
    mask = torch.zeros((b, t_max), dtype=torch.float32)
    t_lp = torch.zeros((b, t_max), dtype=torch.float32)
    k_ids = torch.zeros((b, t_max, max(top_k, 1)), dtype=torch.long)
    k_lp = torch.full((b, t_max, max(top_k, 1)), -1e4, dtype=torch.float32)
    starts = []
    for i, (r, s) in enumerate(zip(rollouts, scores, strict=True)):
        seq, n = r.input_ids, len(r.response_ids)
        ids[i, : len(seq)] = torch.tensor(seq)
        att[i, : len(seq)] = 1
        resp[i, :n] = torch.tensor(r.response_ids)
        mask[i, :n] = 1
        t_lp[i, :n] = torch.tensor(s.token_logprobs)
        if top_k and s.topk_ids:
            k_ids[i, :n] = torch.tensor(s.topk_ids)
            k_lp[i, :n] = torch.tensor(s.topk_logprobs)
        starts.append(len(r.prompt_ids))
    out = {
        "ids": ids,
        "att": att,
        "resp": resp,
        "mask": mask,
        "t_lp": t_lp,
        "k_ids": k_ids,
        "k_lp": k_lp,
    }
    out = {k: v.to(device) for k, v in out.items()}
    out["starts"] = torch.tensor(starts)
    return out


def _response_logits(model: Any, pk: dict[str, torch.Tensor]) -> torch.Tensor:
    """Student logits that predict each response token: [B, T_max, V] (float32)."""
    import torch  # noqa: PLC0415

    logits = model(input_ids=pk["ids"], attention_mask=pk["att"]).logits
    t_max = pk["resp"].shape[1]
    rows = []
    for i, st in enumerate(pk["starts"].tolist()):
        sl = logits[i, st - 1 : st - 1 + t_max]
        if sl.shape[0] < t_max:  # pad positions beyond this rollout's response
            sl = torch.cat([sl, sl.new_zeros(t_max - sl.shape[0], sl.shape[1])], 0)
        rows.append(sl)
    return torch.stack(rows).float()


def train_step(
    model: Any,
    optimizer: Any,
    rollouts: Sequence[Rollout],
    scores: Sequence[TeacherScores],
    loss_cfg: OPDLossCfg,
    pad_id: int,
    device: str,
    micro_batch_size: int,
    top_k: int,
    grad_clip: float | None,
    verifier_rewards: Sequence[float] | None = None,
) -> dict[str, float]:
    """One optimizer step over all rollouts (micro-batched, global token normalizer)."""
    import torch  # noqa: PLC0415

    n_total = sum(len(r.response_ids) for r in rollouts)
    optimizer.zero_grad(set_to_none=True)
    rewards, ents, mism, clip_n, loss_sum = [], [], [], 0, 0.0
    for j in range(0, len(rollouts), micro_batch_size):
        rs, ss = rollouts[j : j + micro_batch_size], scores[j : j + micro_batch_size]
        pk = _pack(rs, ss, pad_id, device, top_k if loss_cfg.top_k_kl else 0)
        logits = _response_logits(model, pk)
        s_lp = gather_logprobs(logits, pk["resp"])
        kwargs: dict[str, Any] = {"n_tokens_total": n_total}
        if loss_cfg.top_k_kl:
            kwargs["student_topk_logprobs"] = torch.log_softmax(logits, -1).gather(-1, pk["k_ids"])
            kwargs["teacher_topk_logprobs"] = pk["k_lp"]
        if verifier_rewards is not None:
            kwargs["verifier_reward"] = torch.tensor(
                verifier_rewards[j : j + micro_batch_size], dtype=torch.float32, device=device
            )
        out = opd_loss(s_lp, pk["t_lp"], pk["mask"], loss_cfg, **kwargs)
        out.loss.backward()
        loss_sum += float(out.loss.detach())
        m = pk["mask"].bool()
        with torch.no_grad():
            r = (
                (pk["t_lp"] - s_lp.detach()).clamp(-loss_cfg.clip_tau, loss_cfg.clip_tau)
                if loss_cfg.clip_tau
                else pk["t_lp"] - s_lp
            )
            rewards += r[m].tolist()
            ents += token_entropy(logits.detach())[m].tolist()
            sampled = torch.zeros_like(s_lp)
            for i, ro in enumerate(rs):
                if ro.sampled_logprobs:
                    sampled[i, : len(ro.sampled_logprobs)] = torch.tensor(ro.sampled_logprobs)
            mism += (s_lp.detach() - sampled)[m].abs().tolist()
        clip_n += round(out.clip_frac * out.n_tokens)
    gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip or float("inf")))
    optimizer.step()
    return {
        "loss": loss_sum,
        "reward_mean": statistics.fmean(rewards),
        "reward_std": statistics.pstdev(rewards),
        **{f"reward_{k}": v for k, v in percentiles(rewards).items()},
        "reverse_kl_est": -statistics.fmean(rewards),  # E[log p_S - log p_T] on sampled tokens
        "entropy_mean": statistics.fmean(ents),
        "student_logprob_mismatch": statistics.fmean(mism),
        "clip_frac": clip_n / max(n_total, 1),
        "grad_norm": gn,
        "response_tokens": n_total,
    }


def run_opd(  # noqa: PLR0912 - one linear round loop reads best in one place
    cfg: OPDCfg,
    run: RunDir,
    comps: LoopComponents,
    mode: LoopMode | str,
    seed: int = 0,
    stop_after_rounds: int | None = None,
) -> dict[str, Any]:
    """Run (or resume) the OPD loop; returns the final state."""
    import torch  # noqa: PLC0415

    from qopd.train.opd_loss import loss_cfg_from  # noqa: PLC0415

    mode = LoopMode(mode)
    if cfg.lr is None:
        raise ValueError("opd.lr is null: set it (step 6.5 freezes it)")
    loss_cfg = loss_cfg_from(cfg.loss)
    if loss_cfg.hybrid_weight and comps.verifier is None:
        raise ValueError("hybrid loss needs a verifier")
    model = comps.model
    model.train()
    o = cfg.optimizer
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, betas=o.betas, eps=o.eps, weight_decay=o.weight_decay
    )
    sched = torch.optim.lr_scheduler.LambdaLR(opt, warmup_constant(cfg.warmup_rounds))
    state_file = run.checkpoints / "trainer_state.pt"
    start_round = 0
    if state_file.exists():
        st = torch.load(state_file, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["optimizer"])
        sched.load_state_dict(st["scheduler"])
        start_round = int(st["round"])
        if st.get("engine_path"):
            comps.engine.restart(st["engine_path"])
        log.info("resumed at round %d", start_round)

    def checkpoint(rnd: int) -> None:
        if comps.save_model is not None:
            comps.save_model(model, run.checkpoints / f"round_{rnd:04d}")
        tmp = state_file.with_suffix(".tmp")
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict(),
                "round": rnd,
                "engine_path": getattr(comps.engine, "model_path", None),
            },
            tmp,
        )
        tmp.replace(state_file)
        run.save_state(round=rnd, status="running")

    run.save_state(status="running", mode=mode.value, rounds=cfg.rounds)
    if mode == LoopMode.CO_RESIDENT:
        comps.engine.start()
        comps.teacher.start()
    done = 0
    try:
        for rnd in range(start_round, cfg.rounds):
            if stop_after_rounds is not None and done >= stop_after_rounds:
                break
            secs: dict[str, float] = {}
            with PeakMemory() as pm:
                probs = round_prompts(comps.problems, rnd, cfg.prompts_per_round, seed)
                prompts = [comps.prompt_ids(p) for p in probs]
                t0 = time.perf_counter()
                if mode == LoopMode.PHASE_ALTERNATING:
                    comps.engine.start()
                rollouts, _ = comps.engine.generate(
                    prompts,
                    cfg.rollout,
                    n=cfg.rollouts_per_prompt,
                    problem_ids=[p.id for p in probs],
                    seeds=[request_seed(seed, rnd, p.id) for p in probs],
                )
                if mode == LoopMode.PHASE_ALTERNATING:
                    comps.engine.stop()
                secs["sample"] = time.perf_counter() - t0

                t0 = time.perf_counter()
                if mode == LoopMode.PHASE_ALTERNATING:
                    comps.teacher.start()
                scores = comps.teacher.score(rollouts, cfg.teacher_top_k)
                if mode == LoopMode.PHASE_ALTERNATING:
                    comps.teacher.stop()
                secs["score"] = time.perf_counter() - t0

                vrew = None
                by_id = {p.id: p for p in probs}
                if comps.verifier is not None:
                    vrew = [
                        float(comps.verifier(by_id[r.problem_id], r.response_ids)) for r in rollouts
                    ]
                t0 = time.perf_counter()
                m = train_step(
                    model,
                    opt,
                    rollouts,
                    scores,
                    loss_cfg,
                    comps.pad_id,
                    comps.device,
                    cfg.micro_batch_size or 1,
                    cfg.teacher_top_k,
                    o.grad_clip_norm,
                    vrew if loss_cfg.hybrid_weight else None,
                )
                lr = sched.get_last_lr()[0]
                sched.step()
                secs["train"] = time.perf_counter() - t0

                t0 = time.perf_counter()
                comps.weight_sync.sync(comps.engine, model, rnd + 1)
                secs["sync"] = time.perf_counter() - t0
            lengths = [len(r.response_ids) for r in rollouts]
            row = {
                "round": rnd + 1,
                "lr": lr,
                **m,
                "response_len_mean": statistics.fmean(lengths),
                "truncation_rate": sum(r.truncated for r in rollouts) / len(rollouts),
                "repetition_rate": repetition_rate([r.response_ids for r in rollouts]),
                "verifier_accuracy": statistics.fmean(vrew) if vrew else None,
                "seconds": secs,
                "round_seconds": sum(secs.values()),
                "peak_mem_used_gb": pm.peak_used_gb,
                "min_mem_available_gb": pm.min_available_gb,
            }
            run.log_metrics(row)
            done += 1
            if (rnd + 1) % cfg.checkpoint_every == 0 or rnd + 1 == cfg.rounds:
                checkpoint(rnd + 1)
    finally:
        comps.engine.stop()
        comps.teacher.stop()
    if stop_after_rounds is not None and start_round + done < cfg.rounds:
        checkpoint(start_round + done)  # resumable mid-way
        return run.load_state()
    run.save_state(status="trained", round=cfg.rounds)
    return run.load_state()


def run_opd_spec(spec_path: Path, stop_after_rounds: int | None = None) -> dict[str, Any]:
    """CLI entry (``qopd train opd --spec``): real engine, teacher, student and data."""
    from qopd.config import PROJECT_ROOT, DataCfg, ExperimentCfg, load  # noqa: PLC0415
    from qopd.data.sources import read_parquet  # noqa: PLC0415
    from qopd.eval.verifiers import verify  # noqa: PLC0415
    from qopd.models.loading import load_hf_model, load_tokenizer  # noqa: PLC0415
    from qopd.models.registry import resolve  # noqa: PLC0415
    from qopd.prompts.templates import prompt_ids as build_prompt_ids  # noqa: PLC0415
    from qopd.rollout.engine import ReloadFromCheckpoint, RolloutEngine  # noqa: PLC0415
    from qopd.teacher.vllm_backend import VLLMTeacher  # noqa: PLC0415
    from qopd.utils.logging import setup_logging  # noqa: PLC0415
    from qopd.utils.monitor import Monitor  # noqa: PLC0415
    from qopd.utils.run import create_run_dir  # noqa: PLC0415

    exp = load(spec_path, cls=ExperimentCfg)
    if exp.opd is None or exp.teacher is None:
        raise ValueError(f"{spec_path}: needs an opd section and a teacher")
    if exp.opd.mode is None:
        raise ValueError("opd.mode is null: choose co_resident or phase_alternating (6.4 gate)")
    run = create_run_dir(exp.name, spec_path.stem, exp)
    if run.is_done:
        return run.load_state()
    setup_logging(run.stdout)
    student, teacher = resolve(exp.student), resolve(exp.teacher)
    tok = load_tokenizer(student)
    data = exp.data or DataCfg()
    train = read_parquet(PROJECT_ROOT / data.data_dir / "splits" / "train.parquet")
    max_len = 1024 + exp.opd.rollout.max_tokens  # prompt budget + response
    engine = RolloutEngine(student.path, student.revision, max_model_len=max_len, seed=exp.seeds[0])
    scorer = VLLMTeacher(
        teacher.path,
        teacher.revision,
        max_model_len=max_len,
        max_logprobs=max(20, exp.opd.teacher_top_k),
        seed=exp.seeds[0],
    )
    model = load_hf_model(student)
    if exp.sft is not None and exp.sft.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    def save_model(m: Any, out: Path) -> None:
        m.save_pretrained(out, safe_serialization=True)
        tok.save_pretrained(out)

    def verifier(p: Problem, ids: list[int]) -> bool:
        return verify(p, tok.decode(ids, skip_special_tokens=True)).correct

    comps = LoopComponents(
        model=model,
        engine=engine,
        teacher=scorer,
        weight_sync=ReloadFromCheckpoint(run.checkpoints / "sync", tok),
        problems=train,
        prompt_ids=lambda p: build_prompt_ids(tok, p, exp.opd.rollout.enable_thinking),
        pad_id=int(tok.pad_token_id),
        save_model=save_model,
        verifier=verifier,
    )
    with Monitor(run.monitor):
        state = run_opd(
            exp.opd,
            run,
            comps,
            exp.opd.mode,
            seed=exp.seeds[0],
            stop_after_rounds=stop_after_rounds,
        )
    if state.get("status") == "trained":
        run.mark_done()
    return state
