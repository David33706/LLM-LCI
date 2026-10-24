#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --job-name=llmlci
#SBATCH --output=/work/%u/logs/gpu_%j.log

# run_gpu.sbatch — GENERIC unattended GPU job: allocate -> (serve vLLM) -> wait
# ready -> run YOUR COMMAND -> cleanup. Submit from the LOGIN node; everything
# after the script name is the command to run.
#
#   sbatch run_gpu.sbatch python step3_orchestrate.py --all --workers 8
#   sbatch run_gpu.sbatch python step3_orchestrate.py --all --workers 8 --trim
#   NO_SERVER=1 sbatch --time=00:30:00 run_gpu.sbatch python eval/evaluate.py ...
#
# MODEL SWAP (override everything that differs — server env AND client flags):
#   MODEL=/work/$USER/models/Qwen/Qwen2___5-14B-Instruct-AWQ SERVED_NAME=qwen2.5-14b \
#   MAX_LEN=32768 GPU_UTIL=0.9 DTYPE=float16 EXTRA_ARGS="" \
#   sbatch run_gpu.sbatch \
#     python step3_orchestrate.py --all --model qwen2.5-14b --ctx 32768
#
# DEFAULTS = ADOPTED READER (2026-07-15, flipped 2026-07-23): Qwen3.5-35B-A3B-FP8,
# L20 48 GB, 131K ctx, trim+tables on. Bare submission reproduces the adopted config:
#   sbatch run_gpu.sbatch python step3_orchestrate.py --all --workers 8 --dump-answers
#
# BASELINE 28900 (Qwen2.5-32B-AWQ, A40) — reproduce with overrides:
#   MODEL=/work/$USER/models/Qwen/Qwen2___5-32B-Instruct-AWQ SERVED_NAME=qwen2.5-32b-awq \
#   MAX_LEN=32768 GPU_UTIL=0.9 DTYPE=float16 EXTRA_ARGS="" \
#   sbatch run_gpu.sbatch \
#     python step3_orchestrate.py --all --workers 8 --model qwen2.5-32b-awq --ctx 32768 --trim --tables --dump-answers
# JUDGE runs also use the 2.5-32B overrides above — the referee is pinned.
#
# SLURM CLI flags override the #SBATCH headers (no file edits needed):
#   sbatch --time=06:00:00 run_gpu.sbatch ...        # longer walltime
#   sbatch --account=YOUR_ACCOUNT --partition=YOUR_GPU_PARTITION run_gpu.sbatch ...   # if your cluster needs them
#   (runs used an L20 48 GB; the Qwen2.5 baseline and judge an A40 48 GB; BF16 needs ~96 GB)
#   READY_TIMEOUT_MIN=30 sbatch run_gpu.sbatch ...   # slow model load (default 10 min)
#
# Monitor:  squeue -u $USER        tail -f /work/$USER/logs/gpu_<jobid>.log
#
# WORK DIR: models, venvs, logs and hf_cache live under /work/$USER. Set
# LLMLCI_WORK to use another folder; #SBATCH lines can't read env vars, so then
# also pass --output=$LLMLCI_WORK/logs/gpu_%j.log to sbatch.
# Survives VPN drops / SSH disconnects: the job runs on the cluster, not your shell.



set -u
WORK_ROOT="${LLMLCI_WORK:-/work/$USER}"
MODEL=${MODEL:-$WORK_ROOT/models/Qwen3.5-35B-A3B-FP8}
DEFAULT_EXTRA_ARGS="--reasoning-parser qwen3 --language-model-only --kv-cache-dtype fp8 --enforce-eager --max-num-batched-tokens 2048"
EXTRA_ARGS=${EXTRA_ARGS-$DEFAULT_EXTRA_ARGS}
SERVED_NAME=${SERVED_NAME:-qwen3.5-35b-a3b-fp8}
MAX_LEN=${MAX_LEN:-131072}
GPU_UTIL=${GPU_UTIL:-0.95}
DTYPE=${DTYPE:-bfloat16}
READY_TIMEOUT_MIN=${READY_TIMEOUT_MIN:-10}
# run the code you submitted from (SLURM_SUBMIT_DIR); override with REPO=...
REPO=${REPO:-${SLURM_SUBMIT_DIR:-$WORK_ROOT/LLM-LCI}}
VLLM_LOG=$WORK_ROOT/logs/vllm_${SLURM_JOB_ID}.log
mkdir -p "$WORK_ROOT/logs"
cd "$REPO" || { echo "ERROR: cannot cd to REPO=$REPO"; exit 1; }
echo "=== repo: $REPO ==="
case "$READY_TIMEOUT_MIN" in
    ''|*[!0-9]*|0*) echo "ERROR: READY_TIMEOUT_MIN must be a positive integer (minutes, no leading zero), got '$READY_TIMEOUT_MIN'"; exit 2 ;;
esac

if [ $# -eq 0 ]; then
    echo "ERROR: no command given. Usage: sbatch run_gpu.sbatch <command ...>"
    exit 2
fi

module load anaconda
export HF_HOME=$WORK_ROOT/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

SERVER_PID=""
if [ "${NO_SERVER:-0}" = "1" ]; then
    echo "=== [1/3] NO_SERVER=1: skipping vLLM startup ==="
else
    echo "=== [1/3] starting vLLM: $SERVED_NAME"
    echo "           model:   $MODEL"
    echo "           max_len: $MAX_LEN | gpu_util: $GPU_UTIL | log: $VLLM_LOG ==="
    if [ ! -d "$MODEL" ]; then
        echo "ERROR: model directory not found: $MODEL"
        exit 1
    fi
    if curl -s -o /dev/null localhost:3000/v1/models; then
        echo "ERROR: something already answers on localhost:3000 — refusing to start a second server"
        echo "       (an older server would be mistaken for this one). Stop it first."
        exit 1
    fi
    source "$WORK_ROOT/venv-vllm/bin/activate"
    vllm serve "$MODEL" \
      --served-model-name "$SERVED_NAME" \
      --max-model-len "$MAX_LEN" \
      --gpu-memory-utilization "$GPU_UTIL" \
      --dtype "$DTYPE" \
      --port 3000 $EXTRA_ARGS > "$VLLM_LOG" 2>&1 &
    SERVER_PID=$!

    echo "=== [2/3] waiting for server readiness ==="
    READY=0
    for i in $(seq 1 $((READY_TIMEOUT_MIN * 12))); do
        if curl -sf localhost:3000/v1/models 2>/dev/null | grep -qF "\"$SERVED_NAME\""; then
            READY=1
            break
        fi
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "ERROR: vLLM process died during startup — see $VLLM_LOG"
            exit 1
        fi
        sleep 5
    done
    if [ "$READY" -ne 1 ]; then
        echo "ERROR: vLLM did not become ready in $READY_TIMEOUT_MIN min (serving $SERVED_NAME) — see $VLLM_LOG"
        echo "       slow load? resubmit with READY_TIMEOUT_MIN=30"
        kill "$SERVER_PID" 2>/dev/null
        exit 1
    fi
    echo "SERVER READY ($SERVED_NAME)"
fi

echo "=== [3/3] running: $* ==="
source "$WORK_ROOT/venv-llmlci/bin/activate"
# cwd is already $REPO (checked cd at the top)
"$@"
STATUS=$?

if [ -n "$SERVER_PID" ]; then
    kill "$SERVER_PID" 2>/dev/null
fi
echo "=== done (exit $STATUS) ==="
exit $STATUS
