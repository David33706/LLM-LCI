#!/bin/bash
# ask.sh — one-shot question, ZERO setup: submits a Slurm job, waits, prints
# the answer. Each call boots its own server (queue + model load ≈ 3–6 min).
# For "which study" questions, skip the GPU entirely:
#   module load anaconda && source /work/$USER/venv-llmlci/bin/activate && python tools/ask.py "..."
# Usage:   ./ask.sh "your question" [extra tools/ask.py flags]
# Work dir: /work/$USER by default; set LLMLCI_WORK to use another folder.
set -u
WORK_ROOT="${LLMLCI_WORK:-/work/$USER}"
cd "$(dirname "$0")"
if [ $# -eq 0 ]; then
    echo 'usage: ./ask.sh "your question" [flags]'
    exit 2
fi
echo "submitting one-shot job — waits through queue + model load (typically 3-6 min)..." >&2
mkdir -p "$WORK_ROOT/logs"
JOBID=$(sbatch --wait --parsable --time=00:30:00 --output="$WORK_ROOT/logs/gpu_%j.log" run_gpu.sbatch python tools/ask.py "$@")
RC=$?
JOBID=${JOBID%%;*}
if [ -z "$JOBID" ]; then
    echo "ERROR: submission failed" >&2
    exit 1
fi
LOG=$WORK_ROOT/logs/gpu_${JOBID}.log
if [ ! -f "$LOG" ]; then
    echo "ERROR: job $JOBID finished (exit $RC) but its log is missing: expected $LOG" >&2
    [ "$RC" -ne 0 ] && exit $RC
    exit 1
fi
awk '/=== \[3\/3\]/{found=1; next} found' "$LOG"
if [ "$RC" -ne 0 ]; then
    echo "(job exited $RC — full log: $LOG)" >&2
fi
exit $RC
