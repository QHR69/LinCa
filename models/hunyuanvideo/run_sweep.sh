#!/bin/bash
# Hyperparameter sweep for the HunyuanVideo LinCa predictor.
# Runs different (NUM_BLOCKS, HIDDEN_DIM) combinations in parallel across GPUs,
# with each GPU claiming the next job as soon as it finishes one.
#
# Usage:
#   bash run_sweep.sh --gpus 2,3,4,5,6,7
#   bash run_sweep.sh --gpus 0,1 --blocks 1,2 --hiddens 64,128,256
#   bash run_sweep.sh --gpus 4,5,6,7 --blocks 1 --hiddens 64,128

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# ========== Defaults ==========
# Prefer the GPUs SLURM assigned via CUDA_VISIBLE_DEVICES, otherwise assume all eight
GPU_IDS="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
BLOCKS_STR="1,2,3,4"
HIDDENS_STR="256"
INTERVALS_STR="6"

# GPU_IDS="0,1"
# BLOCKS_STR="1,2"
# HIDDENS_STR="256"

# ========== Parse arguments ==========
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)    GPU_IDS="$2";     shift 2 ;;
        --blocks)  BLOCKS_STR="$2";  shift 2 ;;
        --hiddens) HIDDENS_STR="$2"; shift 2 ;;
        --intervals) INTERVALS_STR="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: bash run_sweep.sh --gpus 0,1,2,3 [--blocks 1,2] [--hiddens 64,128,256]"
            echo ""
            echo "Options:"
            echo "  --gpus       comma-separated GPU ids (required)"
            echo "  --blocks     NUM_BLOCKS values to try (default: 1,2)"
            echo "  --hiddens    HIDDEN_DIM values to try (default: 64,128,256)"
            echo "  --intervals  interval values (default: 3,4,5,6,7,8,9,10)"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ========== Build the job list ==========
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
echo "HunyuanVideo LinCa hyperparameter sweep"
echo "========================================"
echo "GPUs:     ${GPU_LIST[*]} (${NUM_GPUS} total)"
echo "INTERVALS: ${INTERVALS_STR}"
echo "Jobs:     ${TOTAL_JOBS}"
echo "Grid:"
for job in "${JOBS[@]}"; do
    b=$(echo "$job" | cut -d: -f1)
    h=$(echo "$job" | cut -d: -f2)
    echo "  NUM_BLOCKS=$b  HIDDEN_DIM=$h"
done
echo "========================================"
echo ""

# ========== Job queue, claimed atomically ==========
JOB_INDEX_FILE=$(mktemp /tmp/sweep_idx_XXXXXX)
echo "0" > "$JOB_INDEX_FILE"
LOCK_FILE="${JOB_INDEX_FILE}.lock"

# Log directory
LOG_DIR="${SCRIPT_DIR}/sweep_logs_8gpu_interval3-10"
mkdir -p "$LOG_DIR"

# Results file
RESULT_FILE="${LOG_DIR}/results.txt"
echo "# Sweep started at $(date)" > "$RESULT_FILE"

# ========== Cleanup ==========
cleanup() {
    echo ""
    echo "Interrupted; stopping all jobs..."
    kill -- -$$ 2>/dev/null
    rm -f "$JOB_INDEX_FILE" "$LOCK_FILE"
    exit 1
}
trap cleanup SIGINT SIGTERM

# ========== Worker ==========
worker() {
    local gpu_id=$1

    while true; do
        # Claim the next job index atomically
        local idx
        {
            flock -x 200
            idx=$(cat "$JOB_INDEX_FILE")
            if [ "$idx" -ge "$TOTAL_JOBS" ]; then
                # No jobs left
                true
            else
                echo $((idx + 1)) > "$JOB_INDEX_FILE"
            fi
        } 200>"$LOCK_FILE"

        if [ "$idx" -ge "$TOTAL_JOBS" ]; then
            break
        fi

        local job="${JOBS[$idx]}"
        local num_blocks=$(echo "$job" | cut -d: -f1)
        local hidden_dim=$(echo "$job" | cut -d: -f2)
        local log_file="${LOG_DIR}/b${num_blocks}_h${hidden_dim}_interval${INTERVALS_STR}.log"

        echo "[GPU $gpu_id] start ($((idx+1))/$TOTAL_JOBS): blocks=$num_blocks, hidden=$hidden_dim  log: $log_file"

        local start_time=$(date +%s)

        # Pass the settings through the environment to run_train.sh
        CUDA_VISIBLE_DEVICES=$gpu_id \
        NUM_BLOCKS=$num_blocks \
        HIDDEN_DIM=$hidden_dim \
        INTERVALS=$INTERVALS_STR \
            bash "${SCRIPT_DIR}/run_train.sh" \
            > "$log_file" 2>&1
        local exit_code=$?

        local end_time=$(date +%s)
        local elapsed=$(( (end_time - start_time) / 60 ))

        if [ $exit_code -eq 0 ]; then
            echo "[GPU $gpu_id] done ($((idx+1))/$TOTAL_JOBS): blocks=$num_blocks, hidden=$hidden_dim  (${elapsed} min)"
            {
                flock -x 201
                echo "OK   blocks=$num_blocks hidden=$hidden_dim  gpu=$gpu_id  ${elapsed}min" >> "$RESULT_FILE"
            } 201>"${RESULT_FILE}.lock"
        else
            echo "[GPU $gpu_id] FAILED ($((idx+1))/$TOTAL_JOBS): blocks=$num_blocks, hidden=$hidden_dim  exit=$exit_code  see: $log_file"
            {
                flock -x 201
                echo "FAIL blocks=$num_blocks hidden=$hidden_dim  gpu=$gpu_id  exit=$exit_code" >> "$RESULT_FILE"
            } 201>"${RESULT_FILE}.lock"
        fi
    done

    echo "[GPU $gpu_id] worker exiting, no jobs left"
}

# ========== Start the workers ==========
for gpu in "${GPU_LIST[@]}"; do
    worker "$gpu" &
done

# Wait for every worker
wait

# ========== Summary ==========
echo ""
echo "========================================"
echo "All $TOTAL_JOBS sweep jobs finished"
echo "========================================"
echo ""
cat "$RESULT_FILE"
echo ""
echo "Logs: $LOG_DIR"
echo "Output directory: ${SCRIPT_DIR}/outputs/"

# Remove the queue file
rm -f "$JOB_INDEX_FILE" "$LOCK_FILE" "${RESULT_FILE}.lock"
