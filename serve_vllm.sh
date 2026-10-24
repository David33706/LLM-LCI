#!/bin/bash
# serve_vllm.sh — vLLM reader on the CURRENT GPU node (interactive sessions).
# Defaults = adopted reader. Interactive L20 first:
#   srun --account=YOUR_ACCOUNT --partition=YOUR_GPU_PARTITION --gres=gpu:1 --cpus-per-task=8 --mem=48G --time=03:00:00 --pty bash
# Baseline referee instead:
#   MODEL=/work/$USER/models/Qwen/Qwen2___5-32B-Instruct-AWQ SERVED_NAME=qwen2.5-32b-awq \
#   MAX_LEN=32768 GPU_UTIL=0.9 DTYPE=float16 EXTRA_ARGS= bash serve_vllm.sh
# Work dir: /work/$USER by default; set LLMLCI_WORK to use another folder.
set -u
WORK_ROOT="${LLMLCI_WORK:-/work/$USER}"
MODEL=${MODEL:-$WORK_ROOT/models/Qwen3.5-35B-A3B-FP8}
SERVED_NAME=${SERVED_NAME:-qwen3.5-35b-a3b-fp8}
MAX_LEN=${MAX_LEN:-131072}
GPU_UTIL=${GPU_UTIL:-0.95}
DTYPE=${DTYPE:-bfloat16}
DEFAULT_EXTRA_ARGS="--reasoning-parser qwen3 --language-model-only --kv-cache-dtype fp8 --enforce-eager --max-num-batched-tokens 2048"
EXTRA_ARGS=${EXTRA_ARGS-$DEFAULT_EXTRA_ARGS}
LOG=$WORK_ROOT/vllm.log

if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: no GPU on this node ($(hostname)) — this looks like the login node."
    echo "Get a GPU node first, then re-run:"
    echo "  srun --account=YOUR_ACCOUNT --partition=YOUR_GPU_PARTITION --gres=gpu:1 --cpus-per-task=8 --mem=48G --time=03:00:00 --pty bash"
    exit 1
fi

module load anaconda
source "$WORK_ROOT/venv-vllm/bin/activate"
export HF_HOME=$WORK_ROOT/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

if curl -s -o /dev/null localhost:3000/v1/models; then
    echo "ERROR: something already answers on localhost:3000 — refusing to start a second server"
    echo "       (an older server would be mistaken for this one). Stop it first (e.g. pkill -f 'vllm serve')."
    exit 1
fi

echo "starting vLLM: $SERVED_NAME ($MODEL) | max_len $MAX_LEN | log: $LOG"
vllm serve "$MODEL" \
  --served-model-name "$SERVED_NAME" \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --dtype "$DTYPE" \
  --port 3000 $EXTRA_ARGS > "$LOG" 2>&1 &
SERVER_PID=$!

for i in $(seq 1 120); do
    if curl -sf localhost:3000/v1/models 2>/dev/null | grep -qF "\"$SERVED_NAME\""; then
        echo "SERVER READY (pid $SERVER_PID, port 3000)"
        echo "next:  source $WORK_ROOT/venv-llmlci/bin/activate  && run your script"
        exit 0
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "ERROR: vLLM process died during startup — tail $LOG"
        exit 1
    fi
    sleep 5
done
echo "ERROR: vLLM did not become ready in 10 min — tail $LOG"
exit 1
