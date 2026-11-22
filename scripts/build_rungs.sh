#!/usr/bin/env bash
# Build + smoke-test + benchmark teacher rungs in order (PLAN 4.2/4.3). Host-side launcher.
#   scripts/build_rungs.sh T4-fp T3 T4-int        (run inside tmux; logs to runs/quant/)
# Stops the resident vLLM container `vllm` first and ALWAYS restarts it on exit (user-approved
# protocol, PROGRESS.md). A failed rung stops the chain; later rungs are skipped.
set -u
cd "$(dirname "$0")/.."
LOG=runs/quant/build_rungs.log
mkdir -p runs/quant
FLAGS=(--rm --gpus all --ipc=host -u "$(id -u):$(id -g)" -e HOME=/tmp/home -e HF_HUB_OFFLINE=1
       -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro -v "$PWD:/workspace/opd"
       -v "$HOME/.cache/huggingface:/hf" -w /workspace/opd)
restart() { echo "restarting resident vLLM $(date)" | tee -a "$LOG"; docker start vllm >>"$LOG" 2>&1; }
trap restart EXIT
echo "stopping resident vLLM $(date)" | tee -a "$LOG"
docker stop vllm >>"$LOG" 2>&1
for rung in "$@"; do
  echo "=== $rung build start $(date)" | tee -a "$LOG"
  if ! docker run "${FLAGS[@]}" qopd-quant:26.10 qopd quant build "$rung" >>"$LOG" 2>&1; then
    echo "=== $rung BUILD FAILED $(date); stopping the chain" | tee -a "$LOG"; exit 1
  fi
  echo "=== $rung build ok $(date); smoke test + bench" | tee -a "$LOG"
  if ! docker run "${FLAGS[@]}" qopd-dev:26.10 qopd teacher bench --model "$rung" --n 200 >>"$LOG" 2>&1; then
    echo "=== $rung SMOKE TEST FAILED $(date); stopping the chain" | tee -a "$LOG"; exit 1
  fi
  echo "=== $rung done $(date)" | tee -a "$LOG"
done
echo "=== all done $(date)" | tee -a "$LOG"
