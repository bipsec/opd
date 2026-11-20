# Quantized-Teacher On-Policy Distillation: Experiment Plan

Copy of the experiment-plan doc as of Nov 20, 2026. The live doc is authoritative for scientific choices; if this copy and the doc disagree, ask the user. Two diagrams in the doc (the OPD loop and the timeline) are described in words here.

## Summary

The study asks how many bits a teacher needs for on-policy distillation (OPD) to still work, measured on one DGX Spark with public math and code data. Nearly all OPD papers use a full-precision teacher; desktop users actually run FP8, NVFP4 or 4-bit teachers.

**Research question.** As the teacher drops from BF16 to FP8, 4-bit and 3-bit, how much does the distilled student lose, why, and can the loss be recovered cheaply?

**Hypotheses** (to be tested, not assumed):

1. **H1, robustness.** FP8 and 4-bit teachers produce students within 1 point of a BF16-teacher student on MATH-500.
2. **H2, a breaking point.** Below 4 bits, student gains fall sharply. Teacher-signal fidelity on student rollouts predicts the fall better than the teacher's own benchmark score.
3. **H3, where it breaks.** The lost signal concentrates on high-entropy tokens, where the student is choosing between reasoning paths. Clipping or top-k smoothing of the per-token reward recovers part of the gap.
4. **H4, equal memory.** At a fixed memory budget, a larger quantized teacher teaches better than a smaller full-precision one.

**What the paper would claim.** A practical rule for teacher precision in OPD, a fidelity metric that predicts it before any training, and a cost-quality frontier measured on a single desktop machine. A null result ("4-bit is enough, here is the evidence") is still a useful, citable finding.

## Background and positioning

The closest prior work quantizes the student and keeps the teacher in full precision; this study does the reverse. In OPD, the student samples its own outputs and a teacher scores each token. Thinking Machines Lab popularized the method in October 2025, and it is now a standard post-training step.

| Work | What it shows | Why it matters here |
| --- | --- | --- |
| [Train Where the Quantized Model Goes](https://www.alphaxiv.org/abs/2609.26708) (Sep 2026) | OPD recovers reasoning in a low-bit *student*, guided by a frozen full-precision teacher | Closest work; the teacher side is untouched |
| [Understanding OPD in LLM Training](https://www.alphaxiv.org/overview/2607.13399) (Jul 2026) | A wide student–teacher gap makes the signal uninformative; proposes an informativeness metric | Supplies a fidelity metric to reuse for H2 |
| [Rethinking OPD](https://www.emergentmind.com/papers/2604.13016) (Apr 2026) | Thinking-pattern alignment matters more than teacher strength; off-policy cold start fixes failures | Basis for the cold-start step and H2 |
| [Lightning OPD](https://arxiv.org/pdf/2604.13010) (NVIDIA, Apr 2026) | Precomputed teacher log-probs remove the live teacher server | Cheaper training mode if compute runs short |
| [Reopold](https://arxiv.org/abs/2603.11137v1) (Mar 2026) | Treats the teacher–student log-ratio as a token reward; filters harmful signals | Reward clipping used in H3 recovery |
| [Trust Region OPD](https://arxiv.org/html/2606.01249) (Jun 2026) | Stabilizes OPD for small reasoning models | Alternative stabilizer baseline |
| [Survey of OPD](https://arxiv.org/abs/2604.00626) (Apr 2026) | Taxonomy of feedback signal, teacher access and loss granularity | Related-work framing |
| [EasyOPD](https://www.alphaxiv.org/overview/2607.11012) (Jul 2026) | Modular framework for comparing OPD methods | Possible codebase to build on |

**Novelty check before week 1.** Search arXiv, alphaXiv and Semantic Scholar for "quantized teacher", "low-bit teacher distillation" and "teacher precision" with "on-policy". This review was quick, not exhaustive. Repeat the search weekly until submission.

## Hardware and software setup

Everything runs on one DGX Spark (GB10, 128 GB unified memory). The binding limit is compute and memory bandwidth, not capacity, so the design keeps students small and runs short, well-measured jobs. Stop the resident vLLM process (about 60 GB) before experiments.

**Memory budget, worst case (BF16 teacher co-resident).** Figures are approximate and get measured in the week-1 pilot.

| Component | Approx. memory |
| --- | --- |
| Teacher Qwen3-30B-A3B, BF16 weights | 61 GB |
| Student 1.7B, full fine-tune (weights, grads, AdamW states) | 25–30 GB |
| Student rollout engine (vLLM weights + KV cache) | 10–15 GB |
| Total | 96–106 GB of 128 GB |

Two run modes, chosen in the pilot:

- **Co-resident.** Teacher and student both stay loaded. Fastest if it fits at BF16.
- **Phase-alternating.** Each round generates rollouts, unloads the student engine, loads the teacher to score, then trains. Model loads from NVMe cost about a minute each, small against a 10-minute round.

**Software stack.**

- NVIDIA NGC PyTorch and vLLM containers built for GB10 (aarch64, sm_121).
- vLLM serves student rollouts and teacher scoring (`prompt_logprobs` with top-k = 20 returns per-token teacher log-probs on the student's sequence).
- Training loop: a small custom PyTorch loop (about 300 lines) or EasyOPD; TRL's GKD trainer as a cross-check.
- Quantization: llm-compressor or NVIDIA ModelOpt for FP8 and NVFP4; llm-compressor (AWQ) for INT4 and GPTQModel for 3-bit (AutoAWQ is archived).
- Fallback ladder: llama.cpp GGUF (Q8_0, Q4_K_M, Q3_K_M, Q2_K) if a format fails on sm_121. It returns log-probs and gives a clean bit ladder.
- Evaluation: math-verify for answer checking, lm-evaluation-harness or lighteval for standard sets; Weights & Biases for logs.
- Power: log `nvidia-smi` power draw once a second to compute energy per run.

## Models

All models come from one family so teacher and student share a tokenizer, which token-level OPD requires. The plan uses Qwen3 (Apache 2.0); if a newer same-tokenizer Qwen release is current when you start, swap it in throughout.

| Role | Model | Why |
| --- | --- | --- |
| Main teacher | Qwen3-30B-A3B, thinking mode | Strong reasoner; 30.5B total, 3.3B active, so scoring is fast |
| Dense teacher | Qwen3-8B, thinking mode | Separates MoE from dense quantization effects; the equal-memory rival for H4 |
| Student S1 | Qwen3-1.7B | Main student; large enough to gain from distillation |
| Student S2 | Qwen3-0.6B | Cheap student for seeds and replication |

**Teacher quantization ladder.** Each rung is one condition; memory is approximate for the 30B-A3B teacher.

| Condition | Format | Tool | Approx. weights |
| --- | --- | --- | --- |
| T16 | BF16 (reference) | none | 61 GB |
| T8 | FP8 W8A8 | llm-compressor | 31 GB |
| T4-fp | NVFP4 | ModelOpt or llm-compressor | 17 GB |
| T4-int | INT4 AWQ, W4A16, group 128 | llm-compressor (AWQ) | 17 GB |
| T3 | 3-bit GPTQ, group 128 | GPTQModel | 12 GB |
| T2 (stress, optional) | 2-bit (GGUF Q2_K) | llama.cpp | 8–10 GB |

Use the same calibration set for every quantized rung: 512 samples drawn from the training prompts, disjoint from all evaluation sets. Record each rung's own benchmark score so H2 can compare it with the fidelity metric.

**Fixed across all conditions:** student initialization, cold-start data, prompts, seeds, sampling settings, and the evaluation harness. Only teacher precision changes in the main grid.

## Data

Training uses public math prompts with checkable answers; evaluation adds held-out contests released after the models, to limit contamination. Code is a secondary track, run only if the math grid finishes on time.

**Training prompts**

- Math pool: about 16k problems from a public set such as DeepScaleR-Preview or NuminaMath, filtered to integer or simple closed-form answers so math-verify can check them.
- Cold-start SFT: 5k of those prompts, answered once by the BF16 teacher, keeping only correct traces. This set is generated once and shared by every condition.
- Calibration: 512 prompts reserved for quantization, never trained on.
- Code track (optional): about 4k problems with unit tests from a public set such as TACO.

**Evaluation**

| Benchmark | Size | Samples per problem | Role |
| --- | --- | --- | --- |
| GSM8K test | 1,319 | 1 | Easy sanity check |
| MATH-500 | 500 | 4 | Primary metric |
| AMC 2023 | 40 | 8 | Medium difficulty |
| AIME 2024 + 2025 | 60 | 16 | Hard; high variance |
| AIME 2026 + HMMT Feb 2026 | about 60 | 16 | Post-release, contamination-clean |
| GPQA-Diamond | 198 | 4 | Out-of-domain check |
| LiveCodeBench, problems after Apr 2025 | about 200 | 4 | Code track only |

**Decoding.** Thinking mode, temperature 0.6, top-p 0.95, max 16k tokens for evaluation. Report pass@1 averaged over the samples.

**Contamination controls.** Remove training prompts with 13-gram overlap against any evaluation set. Report the post-release contests separately; if conclusions differ between them and older sets, say so.

## Method

Each run is a short cold-start SFT followed by about 100 OPD rounds; the only thing that differs between conditions is the teacher's precision.

One OPD round has five steps: (1) the student generates 64 prompts × 4 rollouts; (2) the teacher at bit-width b scores every token via vLLM; (3) the per-token reward is r = log πT − log πS; (4) one policy-gradient step updates the student; (5) weights sync to the rollout engine, and the next round begins.

**Objective.** Sampled-token reverse KL, written as a policy gradient with a stop-gradient (sg) on the reward. Here π_T^(b) is the teacher quantized to b bits and π_θ is the student.

```latex
\mathcal{L}(\theta) = -\,\mathbb{E}_{y \sim \pi_\theta}\Big[\sum_t \operatorname{sg}\big(\log \pi_T^{(b)}(y_t \mid y_{<t}) - \log \pi_\theta(y_t \mid y_{<t})\big)\,\log \pi_\theta(y_t \mid y_{<t})\Big]
```

**Recipe.**

1. Cold start: 1 epoch of SFT on the shared 5k BF16-teacher traces (learning rate 1e-5). Saved once and reused by every condition.
2. OPD rounds: sample 64 prompts, 4 rollouts each, score with the teacher, take one optimizer step, sync weights to vLLM.
3. Checkpoint every 20 rounds; evaluate MATH-500 at each checkpoint and the full suite at the end.

**Hyperparameters**

| Setting | Value |
| --- | --- |
| Student update | Full fine-tune, AdamW, BF16 |
| Learning rate | Sweep 5e-7, 1e-6, 2e-6 with the BF16 teacher only, then freeze |
| Prompts × rollouts per round | 64 × 4 |
| Max rollout length (training) | 4,096 tokens |
| Rollout temperature | 1.0 |
| Rounds per run | 100 |
| Teacher top-k logged | 20 |
| Seeds | 2 per condition; a third for S1 on T16, T4-int and T3 if time allows |

Tune nothing per condition: a learning rate re-tuned for each teacher would hide the effect being measured.

**Training details settled in build step 0.3** (Nov 20, 2026; same for every condition):

| Setting | OPD | Cold-start SFT |
| --- | --- | --- |
| Optimizer | AdamW, betas 0.9 / 0.999, eps 1e-8 | Same |
| Weight decay | 0.0 | 0.0 |
| Gradient clipping | Global norm 1.0 | Global norm 1.0 |
| Batch | All 256 rollouts per round, one optimizer step via gradient accumulation | 32 sequences globally |
| Loss averaging | Mean over all response tokens in the batch | Mean over all response tokens in the batch |
| Schedule | Linear warmup over 5 rounds, then constant | Linear warmup over 3% of steps, cosine decay to 10% of peak |
| Max sequence length | 4,096 response tokens (as above) | 8,192 tokens, prompt plus response; longer traces dropped, not truncated; no packing |

Quantization calibration uses the 512 calibration prompts, each paired with a BF16-teacher reasoning trace in chat format and cut at 2,048 tokens. Near-duplicate removal uses MinHash on word 5-grams (128 permutations) and treats Jaccard similarity of 0.8 or higher as a duplicate.

Watch one interaction: SFT traces up to 8,192 tokens can teach the student to reason past the 4,096-token OPD rollout cap. The pilot reports the truncation rate; if more than 30% of round-0 rollouts are truncated, revisit either limit before the grid.

## Experiments

Five experiments, ordered so the cheapest one produces a publishable figure first. E1 needs no training and finishes in a day or two; E2 is the main result.

**E0 — Pilot (week 1).** One short OPD run (20 rounds) of S2 with the BF16 teacher. Confirms the pipeline, measures real throughput and memory, and picks co-resident or phase-alternating mode. Every compute estimate in this plan is replaced with these measurements.

**E1 — Teacher-signal fidelity, no training.** Generate a fixed pool of 2,000 rollouts from each cold-started student. Score the same pool with every teacher rung and compare against BF16 (metrics in the next section). Also score each rung on MATH-500 directly, so H2 can test whether fidelity beats teacher accuracy as a predictor.

**E2 — Main grid.** Full OPD runs for every student × teacher rung, plus baselines.

| Condition | Teacher | Students | Seeds | Runs |
| --- | --- | --- | --- | --- |
| OPD | T16, T8, T4-fp, T4-int, T3 | S1, S2 | 2 | 20 |
| OPD stress (optional) | T2 | S1 | 2 | 2 |
| Baseline: cold start only | none | S1, S2 | 1 | 2 (eval only) |
| Baseline: off-policy SFT, compute-matched | T16 traces | S1, S2 | 2 | 4 |
| Baseline: RLVR (GRPO, verifier reward only), wall-clock-matched | none | S1 | 2 | 2 |

The RLVR baseline answers the practical question: is a cheap teacher still better than no teacher at all?

**E3 — Mechanism and recovery (S1 only, worst rung that still trains plus T4-int).**

- Fidelity by token entropy: bucket tokens by student entropy and show where quantized-teacher rewards diverge (tests H3).
- Reward clipping: cap |r_t| at 2 or 5.
- Top-k smoothing: replace the sampled-token reward with KL over the teacher's top-20 tokens, renormalized.
- Hybrid: add the verifier's correctness reward to the OPD reward.

**E4 — Equal-memory teachers (tests H4).** Qwen3-30B-A3B at 4-bit (about 17 GB) against Qwen3-8B at BF16 (about 16 GB), both teaching S1, 2 seeds each. Adds the dense 8B at FP8 and 4-bit to check whether MoE and dense teachers degrade differently.

**E5 — Cost frontier.** No new runs: plot student accuracy against teacher memory, wall-clock and energy for every E2 and E4 run.

## Metrics and diagnostics

The primary outcome is student MATH-500 pass@1; everything else either explains it or prices it.

| Group | Metric | Used for |
| --- | --- | --- |
| Student quality | Pass@1 on each evaluation set (samples per problem as in Data) | H1, H2, H4 |
| Teacher fidelity (vs BF16, on the E1 pool) | Pearson correlation of per-token rewards r_t | H2 |
| | Mean absolute reward error, mean of abs(r_t − r_t BF16) | H2 |
| | KL between quantized and BF16 teacher over the top-20 tokens | H2 |
| | Top-1 token agreement | H2 |
| | Share of tokens whose reward changes sign | H2, H3 |
| | Informativeness, as defined in the student–teacher mismatch paper | H2 |
| Training dynamics | Student entropy, response length, KL to the cold-start model | Failure diagnosis |
| | Repetition rate: share of outputs ending in a repeated 8-gram | Catches looping, a known low-bit failure |
| Cost | Wall-clock hours, peak memory, energy (kWh), tokens processed | E5 frontier |

**Statistics.**

- Report mean and 95% confidence intervals over seeds, plus bootstrap intervals over problems.
- H1 is an equivalence claim, so test it with two one-sided tests (TOST) and a margin of ±1.0 point on MATH-500. Fix the margin before running E2.
- For H2, rank-correlate each rung's fidelity score and its teacher accuracy with the student's final accuracy, and compare the two correlations.
- Fix the comparisons in advance: each quantized rung vs T16, and E4's two equal-memory teachers. Correct for multiple comparisons with Holm.

## Compute budget and timeline

The plan needs roughly 500 machine-hours, about 21 days of continuous Spark time, spread over 8 weeks. All hour figures are estimates; replace them with the week-1 pilot measurements and shrink the grid if they come in high.

Timeline (weeks from start): W1 setup and pilot (gate: S1 round under 15 min); W2 signal fidelity E1 (gate: E1 figure ready, early note possible); W3–W5 main grid E2; W6 mechanism and equal memory E3, E4; W7 evaluation and analysis E5; W8 write and release, then post to arXiv.

| Item | Runs | Est. hours per run | Est. total hours |
| --- | --- | --- | --- |
| E0 pilot | 1 | 4 | 4 |
| E1 fidelity scoring, all rungs, both pools | none | none | 15 |
| E2 OPD, S1 (incl. T2 stress) | 12 | 17 | 204 |
| E2 OPD, S2 | 10 | 6 | 60 |
| E2 baselines: SFT (4) and RLVR (2) | 6 | 5 SFT, 17 RLVR | 54 |
| E3 ablations, 60 rounds each | 6 | 10 | 60 |
| E4 equal-memory 8B teacher | 4 | 17 | 68 |
| Evaluation of all checkpoints | none | none | 40 |
| **Total** | | | **about 505** |

Per-round estimate behind these numbers, for S1: about 5 minutes of rollouts (64 × 4 rollouts, up to 4k tokens), 2 minutes of teacher scoring, 3 minutes of training. Run jobs overnight and on weekends; evaluation can overlap with the next training run only if memory allows.

## Risks and release plan

Every likely outcome still yields a paper; the main risks are tooling on GB10 and running out of compute.

| Risk | Sign it is happening | Fallback |
| --- | --- | --- |
| A quant format will not run in vLLM on GB10 (sm_121) | Load errors or wrong outputs in week 1 | Score with llama.cpp GGUF or plain transformers; slower but format-agnostic |
| Compute overrun | Pilot round time above 15 min for S1 | Cut S1 to 2 seeds, rollout length to 3k tokens, or switch to Lightning-OPD-style precomputed teacher log-probs |
| OPD fails even with the BF16 teacher | No MATH-500 gain after 40 rounds | Longer cold start and teacher-aligned prompt selection (Rethinking OPD) |
| Null result: every rung matches BF16 | E1 fidelity barely moves down to 4 bits | Push to T3 and T2 to find the break point; publish as "4 bits is enough" with the fidelity evidence |
| Getting scooped | A matching preprint appears | Weekly search; post an E1-only note early if one looks close |
| High variance on AIME | Wide intervals | Rely on MATH-500 for claims; AIME as supporting evidence only |

**Release (what makes the preprint credible).**

- [ ] GitHub repo: training loop, configs, quantization recipes, seeds, exact container versions.
- [ ] Hugging Face dataset: the E1 rollout pool with every rung's teacher log-probs, so others can test fidelity metrics without a GPU.
- [ ] Hugging Face models: final student checkpoints for each condition.
- [ ] Public Weights & Biases report for all runs, failed ones included.
- [ ] One script that reproduces E1 on a single DGX Spark in under a day.

## Paper outline

A short empirical paper of about 8 pages plus appendix, posted to arXiv under cs.LG with cs.CL cross-listed. Working title: *How Many Bits Does a Teacher Need? On-Policy Distillation with Quantized Teachers.*

1. **Introduction.** OPD is now standard, but people run quantized teachers; nobody has measured the cost. State the four findings in one line each.
2. **Background.** OPD as policy optimization with a token reward; quantization formats; closest work.
3. **Setup.** Models, quantization ladder, data, single-desktop hardware.
4. **Teacher-signal fidelity (E1).** Figure 1: fidelity metrics against teacher bits, for both students.
5. **Distillation results (E2).** Table 1: every rung × student on all benchmarks. Figure 2: student MATH-500 against teacher bits, with BF16, SFT and RLVR reference lines.
6. **Why it breaks and how to fix it (E3).** Figure 3: reward error by token-entropy bucket. Table 2: recovery methods at the worst rung.
7. **Equal memory and cost (E4, E5).** Figure 4: accuracy against teacher memory, with the 30B-A3B and 8B ladders as two curves.
8. **Limitations.** One model family, math-heavy evaluation, small students, single seeds where noted.
9. **Appendix.** Hyperparameters, quantization recipes, per-benchmark tables, compute log.

**Before posting.** arXiv may ask a first-time submitter for an endorsement in cs categories; arrange one by week 6. Write a short thread or blog post with Figures 2 and 4, since that is how most readers will find the paper.