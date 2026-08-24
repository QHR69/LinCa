#!/bin/bash
# Grid search over predictor capacity, one job per GPU, pulling from a shared
# queue so a GPU picks up the next configuration as soon as it frees up.
# Each job is a full run_train.sh invocation parameterised by environment.
#
# Usage:
#   bash run_sweep.sh --gpus 0,1,2,3
#   bash run_sweep.sh --gpus 0,1 --blocks 1,2 --hiddens 64,128,256
#   bash run_sweep.sh --gpus 4,5,6,7 --intervals 6

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Default to the GPUs the scheduler gave us, otherwise all eight.
GPU_IDS="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
BLOCKS_STR="1,2,3,4"
HIDDENS_STR="256"
INTERVALS_STR="6"

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)      GPU_IDS="$2";       shift 2 ;;
        --blocks)    BLOCKS_STR="$2";    shift 2 ;;
        --hiddens)   HIDDENS_STR="$2";   shift 2 ;;
        --intervals) INTERVALS_STR="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: bash run_sweep.sh --gpus 0,1,2,3 [--blocks 1,2] [--hiddens 64,128,256] [--intervals 6]"
            echo ""
            echo "  --gpus       comma-separated GPU ids"
            echo "  --blocks     NUM_BLOCKS candidates (default: 1,2,3,4)"
            echo "  --hiddens    HIDDEN_DIM candidates (default: 256)"
            echo "  --intervals  INTERVALS passed through to training (default: 6)"
            exit 0 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

IFS=',' read -ra BLOCKS_LIST <<< "$BLOCKS_STR"
IFS=',' read -ra HIDDENS_LIST <<< "$HIDDENS_STR"
IFS=',' read -ra GPU_LIST <<< "$GPU_IDS"

JOBS=()
for b in "${BLOCKS_LIST[@]}"; do
    for h in "${HIDDENS_LIST[@]}"; do
        JOBS+=("${b}:${h}")
    done
done

TOTAL_JOBS=${#JOBS[@]}
NUM_GPUS=${#GPU_LIST[@]}

echo "========================================"
echo "Wan2.1 LinCa hyper-parameter sweep"
echo "========================================"
echo "GPUs:      ${GPU_LIST[*]} (${NUM_GPUS})"
echo "Intervals: ${INTERVALS_STR}"
echo "Jobs:      ${TOTAL_JOBS}"
for job in "${JOBS[@]}"; do
    echo "  NUM_BLOCKS=${job%%:*}  HIDDEN_DIM=${job##*:}"
done
echo "========================================"
echo ""

JOB_INDEX_FILE=$(mktemp /tmp/sweep_idx_XXXXXX)
echo "0" > "$JOB_INDEX_FILE"
LOCK_FILE="${JOB_INDEX_FILE}.lock"

LOG_DIR="${SCRIPT_DIR}/sweep_logs"
mkdir -p "$LOG_DIR"
RESULT_FILE="${LOG_DIR}/results.txt"
echo "# Sweep started at $(date)" > "$RESULT_FILE"

cleanup() {
    echo ""
    echo "Interrupted, stopping all jobs..."
    kill -- -$$ 2>/dev/null
    rm -f "$JOB_INDEX_FILE" "$LOCK_FILE"
    exit 1
}
trap cleanup SIGINT SIGTERM

worker() {
    local gpu_id=$1

    while true; do
        # Claim the next job index atomically.
        local idx
        {
            flock -x 200
            idx=$(cat "$JOB_INDEX_FILE")
            if [ "$idx" -lt "$TOTAL_JOBS" ]; then
                echo $((idx + 1)) > "$JOB_INDEX_FILE"
            fi
        } 200>"$LOCK_FILE"

        if [ "$idx" -ge "$TOTAL_JOBS" ]; then
            break
        fi

        local job="${JOBS[$idx]}"
        local num_blocks="${job%%:*}"
        local hidden_dim="${job##*:}"
        local log_file="${LOG_DIR}/b${num_blocks}_h${hidden_dim}_interval${INTERVALS_STR}.log"
        local start_time=$(date +%s)

        echo "[GPU $gpu_id] start ($((idx+1))/$TOTAL_JOBS): blocks=$num_blocks hidden=$hidden_dim  log: $log_file"

        # `if` rather than `$?` so that set -e does not kill the worker on a
        # failed job; the queue should carry on to the next configuration.
        if CUDA_VISIBLE_DEVICES=$gpu_id \
           NUM_BLOCKS=$num_blocks \
           HIDDEN_DIM=$hidden_dim \
           INTERVALS=$INTERVALS_STR \
               bash "${SCRIPT_DIR}/run_train.sh" > "$log_file" 2>&1; then
            local elapsed=$(( ($(date +%s) - start_time) / 60 ))
            echo "[GPU $gpu_id] done  ($((idx+1))/$TOTAL_JOBS): blocks=$num_blocks hidden=$hidden_dim  (${elapsed} min)"
            {
                flock -x 201
                echo "OK   blocks=$num_blocks hidden=$hidden_dim  gpu=$gpu_id  ${elapsed}min" >> "$RESULT_FILE"
            } 201>"${RESULT_FILE}.lock"
        else
            echo "[GPU $gpu_id] FAIL  ($((idx+1))/$TOTAL_JOBS): blocks=$num_blocks hidden=$hidden_dim  see: $log_file"
            {
                flock -x 201
                echo "FAIL blocks=$num_blocks hidden=$hidden_dim  gpu=$gpu_id" >> "$RESULT_FILE"
            } 201>"${RESULT_FILE}.lock"
        fi
    done

    echo "[GPU $gpu_id] worker exiting, queue empty"
}

for gpu in "${GPU_LIST[@]}"; do
    worker "$gpu" &
done
wait

echo ""
echo "========================================"
echo "All $TOTAL_JOBS jobs finished"
echo "========================================"
cat "$RESULT_FILE"
echo ""
echo "Logs:    $LOG_DIR"
echo "Outputs: ${SCRIPT_DIR}/outputs/"

rm -f "$JOB_INDEX_FILE" "$LOCK_FILE" "${RESULT_FILE}.lock"
