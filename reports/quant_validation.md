# Quantization tooling validation on S2 (step 4.2)

llm-compressor 0.15.0 installed with `--no-deps` (its torch<=2.14.1 pin bypassed; torch 2.15.0a0 from NGC) in `qopd-quant`; checkpoints loaded and scored by the unchanged vLLM 0.31 in `qopd-dev`. Same 20 exact-id S2 rollouts (`runs/agreement/S2_vllm_vs_hf_s0`), top-20. Generated from `models/quant/_val/*/validation.json` and `recipe.json`.

| Checkpoint | tool | build s | calib. samples | tokens | mean abs Δ (nats) | median | p99 | max | top-1 agr. | top-20 overlap |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FP8 W8A8 (static) | llmcompressor 0.15.0 | 213 | 512 | 10240 | 0.076 | 0.0063 | 0.773 | 2.73 | 0.961 | 0.932 |
| AWQ W4A16 g128 | llmcompressor 0.15.0 | 955 | 512 | 10240 | 0.272 | 0.0275 | 2.734 | 9.59 | 0.879 | 0.802 |

Calibration: `data/calibration/t16_traces.jsonl` (sha256 0afa22a805c6432e).
