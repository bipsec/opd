# Data report

Generated from `data/**/*_stats.json` by `qopd data report`; do not edit by hand.

## Training pool (step 2.1)

Source: [`agentica-org/DeepScaleR-Preview-Dataset`](https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset) @ `b6ae8c60f5c1f2b594e2140b91c49c9ad0949e29`, file `deepscaler.json` (sha256 `4e9e7b18248d982b`), license `mit`. AIME 1984-2023, AMC before 2023, Omni-MATH, STILL (per dataset README).

Raw rows: 40315.

| Stage | In | Kept | Dropped (by reason) |
| --- | ---: | ---: | --- |
| normalize | 40315 | 40309 | empty_answer: 6 |
| answer_type | 40309 | 38562 | other:Add: 249, other:And: 23, other:Equality: 222, other:FiniteSet: 331, other:GreaterThan: 11, other:Integers: 1, other:Interval: 160, other:LessThan: 5, other:Mul: 274, other:MutableDenseMatrix: 98, other:Pow: 35, other:StrictGreaterThan: 10, other:StrictLessThan: 6, other:Symbol: 100, other:Tuple: 143, other:Unequality: 4, other:Union: 9, other:binomial: 2, other:ceiling: 1, other:cot: 1, other:f: 1, other:factorial: 5, other:floor: 2, other:g: 1, string: 53 |
| self_check | 38562 | 38560 | boxed_gold_does_not_verify: 2 |
| dedup_exact | 38560 | 37199 | duplicate_with_conflicting_answers: 520, exact_duplicate: 841 |
| dedup_near | 37199 | 37080 | near_duplicate: 119 |

Pool: **37080** problems in `data/pool/pool.parquet` (sha256 `ca874f0ab5e934e9`).

| Answer type | Count |
| --- | ---: |
| closed_form | 6251 |
| integer | 22845 |
| number | 7984 |

| Source | Count |
| --- | ---: |
| deepscaler_preview | 37080 |

Kept answer types: integer, number, closed_form (math-verify parses the answer to one real number with no free variables). Near-duplicates: MinHash on word 5-grams, 128 permutations, Jaccard >= 0.8 (doc setting; it can also merge distinct variants that differ in one symbol). Identical problems whose gold answers disagree are all dropped: True. Every dropped id and reason is in `data/pool/dropped.jsonl`.

## Evaluation suites (step 2.2)

Items are never filtered. `answer types` uses the same math-verify classification as the pool; `not self-verifying` counts items whose gold does not verify against `\boxed{gold}` (to be handled by the 3.1 verifier).

| Suite | Status | Items | Source @ revision | License | Answer types | Not self-verifying |
| --- | --- | ---: | --- | --- | --- | ---: |
| gsm8k | ok | 1319 | `openai/gsm8k` @ `740312ad` | mit | integer: 1319 | 0 |
| math500 | ok | 500 | `HuggingFaceH4/MATH-500` @ `6e4ed1a2` | not stated | closed_form: 28, integer: 338, number: 68, other:Add: 15, other:Equality: 5, other:FiniteSet: 6, other:Interval: 10, other:Mul: 5, other:MutableDenseMatrix: 7, other:Symbol: 8, other:Tuple: 6, other:Union: 3, other:cot: 1 | 0 |
| amc23 | ok | 40 | `math-ai/amc23` @ `80815d37` | not stated | number: 40 | 0 |
| aime24 | ok | 30 | `HuggingFaceH4/aime_2024` @ `2fe88a2f` | not stated | integer: 30 | 0 |
| aime25 | ok | 30 | `MathArena/aime_2025` @ `c94da77e` | cc-by-nc-sa-4.0 | integer: 30 | 0 |
| aime26 | ok | 30 | `MathArena/aime_2026` @ `d61f4868` | cc-by-nc-sa-4.0 | integer: 30 | 0 |
| hmmt_feb26 | ok | 33 | `MathArena/hmmt_feb_2026` @ `589a53fe` | cc-by-nc-sa-4.0 | closed_form: 11, integer: 17, number: 5 | 0 |
| gpqa_diamond | **blocked**: gated: needs the user's HF access (GatedRepoError) | TBD | `Idavidrein/gpqa` @ `83022cef` | cc-by-4.0 | — | — |

## Contamination filter (step 2.3)

Training prompts sharing any 13-gram (lower-cased [a-z0-9]+ word tokens) with an evaluation item are dropped. Pool: 37080 → **36665** (dropped 415; 2259 train–eval pairs) in `data/pool/pool_clean.parquet` (sha256 `8264b101c87262f6`).

| Suite | Eval items | Too short to match | Eval items hit | Training prompts dropped | Near copies: eval items / training prompts |
| --- | ---: | ---: | ---: | ---: | ---: |
| gsm8k | 1319 | 0 | 0 | 0 | 0 / 0 |
| math500 | 500 | 51 | 54 | 241 | 16 / 21 |
| amc23 | 40 | 1 | 11 | 42 | 0 / 0 |
| aime24 | 30 | 0 | 9 | 228 | 0 / 0 |
| aime25 | 30 | 0 | 11 | 202 | 0 / 0 |
| aime26 | 30 | 0 | 10 | 209 | 0 / 0 |
| hmmt_feb26 | 33 | 0 | 1 | 1 | 0 / 0 |

Diagnostic, not dropped: 0 training prompts exactly equal (same tokens) an evaluation item too short for a 13-gram (none).

**Not checked (suite not built):** gpqa_diamond. Re-run `qopd data decontaminate` after building them.

Near copy: the pair shares >= 50% of the evaluation item's n-grams (likely the same problem). Other hits are mostly shared contest boilerplate (e.g. "... where m and n are relatively prime positive integers. Find m+n"); per the doc they are dropped too. A prompt matching several suites counts once per suite. Every drop, with its evaluation ids and an example n-gram, is in `data/contamination/dropped.jsonl`.

## Splits and manifest (step 2.4)

From `data/pool/pool_clean.parquet` (sha256 `8264b101c87262f6`), seed 0: ordered by sha256(seed:id), then cut in order train → calibration → probe; cold-start is the first part of train. Calibration and probe are disjoint from train and from each other. Unused (reserve): 19653.

| Split | Problems | File | sha256 |
| --- | ---: | --- | --- |
| train | 16000 | `data/splits/train.parquet` | `3c22f34faa722daf` |
| coldstart | 5000 | `data/splits/coldstart.parquet` | `296b59855cb9cec7` |
| calibration | 512 | `data/splits/calibration.parquet` | `7dffc645b088bbfe` |
| probe | 500 | `data/splits/probe.parquet` | `2ed0a24b7ec12961` |

Evaluation files recorded in the manifest:

| Suite | Items | sha256 |
| --- | ---: | --- |
| aime24 | 30 | `46c27c49b77be210` |
| aime25 | 30 | `ba905c1632b70193` |
| aime26 | 30 | `1f7c942485b323c9` |
| amc23 | 40 | `a282089d6d215187` |
| gsm8k | 1319 | `a03651a072783521` |
| hmmt_feb26 | 33 | `4b0aab777b65e991` |
| math500 | 500 | `e4f518cf77b980f9` |

**Not built (TBD):** gpqa_diamond — the pool was decontaminated against gsm8k, math500, amc23, aime24, aime25, aime26, hmmt_feb26 only. After building a missing suite, re-run `qopd data decontaminate` and `qopd data splits`.

`data/manifest.json` (tracked in git) holds counts, file sha256 and a sha256 over sorted ids for every file; `qopd data verify` checks them.
