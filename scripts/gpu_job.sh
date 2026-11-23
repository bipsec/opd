#!/usr/bin/env bash
# Run one long qopd GPU command in the dev image (host-side launcher; run inside tmux).
#   scripts/gpu_job.sh <name> <qopd args...>     e.g. scripts/gpu_job.sh coldstart_T16 data coldstart
# Stops the resident vLLM container `vllm` first and ALWAYS restarts it on exit (user-approved
# protocol, PROGRESS.md). Log: runs/jobs/<name>.log; memory: runs/jobs/<name>.mem.csv.
set -u
cd "$(dirname "$0")/.."
NAME=$1; shift
mkdir -p runs/jobs
LOG=runs/jobs/$NAME.log
FLAGS=(--rm --gpus all --ipc=host -u "$(id -u):$(id -g)" -e HOME=/tmp/home -e HF_HUB_OFFLINE=1
       -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro -v "$PWD:/workspace/opd"
       -v "$HOME/.cache/huggingface:/hf" -w /workspace/opd)
( echo "t_unix,mem_available_kb,swap_free_kb"
  while true; do
    echo "$(date +%s),$(awk '/MemAvailable/{print $2}' /proc/meminfo),$(awk '/SwapFree/{print $2}' /proc/meminfo)"
    sleep 30
  done ) > "runs/jobs/$NAME.mem.csv" & SAMPLER=$!
cleanup() { kill "$SAMPLER" 2>/dev/null; echo "restarting resident vLLM $(date)" | tee -a "$LOG"
            docker start vllm >>"$LOG" 2>&1; }
trap cleanup EXIT
echo "stopping resident vLLM $(date)" | tee -a "$LOG"
docker stop vllm >>"$LOG" 2>&1
echo "=== $NAME start $(date): qopd $*" | tee -a "$LOG"
docker run "${FLAGS[@]}" qopd-dev:26.10 qopd "$@" >>"$LOG" 2>&1
echo "=== $NAME exit $? at $(date)" | tee -a "$LOG"
