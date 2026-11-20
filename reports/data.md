# Data report

Generated from `data/**/..._stats.json` by `qopd data report`; do not edit by hand.

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

TBD (step 2.2).

## Contamination filter (step 2.3)

TBD (step 2.3).

## Splits and manifest (step 2.4)

TBD (step 2.4).
