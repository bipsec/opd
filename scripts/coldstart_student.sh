#!/usr/bin/env bash
# PLAN 5.3: train one cold-start student and evaluate it before/after (host-side; run in tmux).
#   tmux new -s coldstart_S2 'scripts/coldstart_student.sh S2'
# Steps (each resumable; a finished step is skipped on re-run):
#   1. GPU smoke test of the SFT trainer on S2 (2 steps, short traces)
#   2. qopd train sft --spec configs/experiments/coldstart_<S>.yaml
#   3. link checkpoints/coldstart/<S> -> runs/coldstart/coldstart_<S>/checkpoints/final
#   4. after:  MATH-500 (4 samples) and GSM8K (1 sample) on the cold-start student
#   5. before: MATH-500 (4 samples) on the official student, only with BEFORE_N4=1. Default: skip
#      and use the n=1 sanity runs (runs/eval/<S>__gsm8k+math500__n1__t16384__s0; user 2026-11-25)
# Stops the resident vLLM container `vllm` first and ALWAYS restarts it on exit (user-approved
# protocol, PROGRESS.md). Log: runs/jobs/coldstart_<S>.log; memory: runs/jobs/coldstart_<S>.mem.csv.
set -u
cd "$(dirname "$0")/.."
S=${1:?student id (S1 or S2)}
NAME=coldstart_$S
KV=${KV_CACHE_GB:-40}
mkdir -p runs/jobs checkpoints/coldstart
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

step() {  # step <label> <command...>  (inside the dev image); stops the chain on failure
  local label=$1; shift
  echo "=== $NAME [$label] start $(date): $*" | tee -a "$LOG"
  docker run "${FLAGS[@]}" qopd-dev:26.10 "$@" >>"$LOG" 2>&1
  local rc=$?
  echo "=== $NAME [$label] exit $rc at $(date)" | tee -a "$LOG"
  [ "$rc" -eq 0 ] || exit "$rc"
}

echo "stopping resident vLLM $(date)" | tee -a "$LOG"
docker stop vllm >>"$LOG" 2>&1

step smoke python -m pytest -q -p no:cacheprovider -m gpu tests/test_sft.py::test_s2_sft_smoke_gpu
step sft qopd train sft --spec "configs/experiments/$NAME.yaml"
ln -sfn "../../runs/coldstart/$NAME/checkpoints/final" "checkpoints/coldstart/$S"
echo "linked checkpoints/coldstart/$S" | tee -a "$LOG"
step after_math500 qopd eval --model "checkpoints/coldstart/$S" --suites math500 --samples 4 \
  --kv-cache-gb "$KV" --run-id "$S-cs__math500__n4__t16384__s0"
step after_gsm8k qopd eval --model "checkpoints/coldstart/$S" --suites gsm8k --samples 1 \
  --kv-cache-gb "$KV" --run-id "$S-cs__gsm8k__n1__t16384__s0"
if [ "${BEFORE_N4:-0}" = 1 ]; then
  step before_math500 qopd eval --model "$S" --suites math500 --samples 4 \
    --kv-cache-gb "$KV" --run-id "${S}__math500__n4__t16384__s0"
fi
echo "=== $NAME all steps done $(date)" | tee -a "$LOG"
