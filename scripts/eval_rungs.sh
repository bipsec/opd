#!/usr/bin/env bash
# Evaluate teacher rungs on MATH-500 (n=1, eval settings) in order (PLAN 4.4). Host-side launcher.
#   scripts/eval_rungs.sh T16 T8 T4-fp T4-int T3     (run inside tmux; logs to runs/quant/)
# Stops the resident vLLM container `vllm` first and ALWAYS restarts it on exit. Each eval is
# resumable (qopd eval skips finished samples); a failed rung stops the chain.
set -u
cd "$(dirname "$0")/.."
LOG=runs/quant/eval_rungs.log
KV_GB=${KV_GB:-32}
mkdir -p runs/quant
FLAGS=(--rm --gpus all --ipc=host -u "$(id -u):$(id -g)" -e HOME=/tmp/home -e HF_HUB_OFFLINE=1
       -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro -v "$PWD:/workspace/opd"
       -v "$HOME/.cache/huggingface:/hf" -w /workspace/opd)
restart() { echo "restarting resident vLLM $(date)" | tee -a "$LOG"; docker start vllm >>"$LOG" 2>&1; }
trap restart EXIT
sample_mem() {
  echo "t_unix,mem_available_kb,swap_free_kb" > "$1"
  while true; do
    echo "$(date +%s),$(awk '/MemAvailable/{print $2}' /proc/meminfo),$(awk '/SwapFree/{print $2}' /proc/meminfo)" >> "$1"
    sleep 30
  done
}
echo "stopping resident vLLM $(date)" | tee -a "$LOG"
docker stop vllm >>"$LOG" 2>&1
for rung in "$@"; do
  echo "=== $rung eval start $(date)" | tee -a "$LOG"
  sample_mem "runs/quant/mem_eval_${rung}.csv" & SAMPLER=$!
  if ! docker run "${FLAGS[@]}" qopd-dev:26.10 qopd eval --model "$rung" --suites math500 \
       --samples 1 --kv-cache-gb "$KV_GB" >>"$LOG" 2>&1; then
    kill "$SAMPLER" 2>/dev/null
    echo "=== $rung EVAL FAILED $(date); stopping the chain" | tee -a "$LOG"; exit 1
  fi
  kill "$SAMPLER" 2>/dev/null
  echo "=== $rung eval done $(date)" | tee -a "$LOG"
done
echo "=== all done $(date)" | tee -a "$LOG"
