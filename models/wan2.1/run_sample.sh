#!/bin/bash
# Stage 3 of the Wan2.1 pipeline: cache-accelerated sampling with a trained LinCa
# predictor, over the VBench prompt set.
#
# Usage:
#   bash run_sample.sh <wan_ckpt_dir> <predictor_dir> [interval ...]
#
# <predictor_dir> is a training output directory holding best_predictor_stage1.pt
# and best_predictor_stage2.pt. Extra arguments are the cache intervals to sweep;
# it defaults to a single run at interval 6.

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH="${SCRIPT_DIR}/src:${PYTHONPATH}"

CKPT_DIR="${1:-${WAN_CKPT_DIR:-./checkpoints/Wan2.1-T2V-1.3B}}"
PREDICTOR_DIR="${2:-}"
shift 2 2>/dev/null || true
INTERVALS=("$@")
if [ ${#INTERVALS[@]} -eq 0 ]; then
    INTERVALS=(6)
fi

if [ -z "$PREDICTOR_DIR" ]; then
    echo "Usage: bash run_sample.sh <wan_ckpt_dir> <predictor_dir> [interval ...]"
    echo "Example: bash run_sample.sh ./checkpoints/Wan2.1-T2V-1.3B outputs/wan_linca_b2_h64_... 6 8"
    exit 1
fi

STAGE1="${PREDICTOR_DIR}/best_predictor_stage1.pt"
STAGE2="${PREDICTOR_DIR}/best_predictor_stage2.pt"
for f in "$STAGE1" "$STAGE2"; do
    if [ ! -e "$f" ]; then
        echo "Predictor checkpoint not found: $f"
        exit 1
    fi
done
if [ ! -d "$CKPT_DIR" ]; then
    echo "Wan checkpoint directory not found: $CKPT_DIR"
    exit 1
fi

TASK="${TASK:-t2v-1.3B}"
SIZE="${SIZE:-832*480}"
FRAME_NUM="${FRAME_NUM:-81}"
SAMPLE_STEPS="${SAMPLE_STEPS:-50}"
BASE_SEED="${BASE_SEED:-42}"
VBENCH_JSON="${VBENCH_JSON:-./data/VBench_full_info.json}"
INDEX_START="${INDEX_START:-0}"
INDEX_END="${INDEX_END:-946}"

# Cache settings, matched to training.
MAX_ORDER="${MAX_ORDER:-2}"
MIN_ORDER="${MIN_ORDER:-0}"
FIRST_ENHANCE="${FIRST_ENHANCE:-3}"
FORECAST_METHOD="${FORECAST_METHOD:-hermite}"

EXP_NAME=$(basename "$PREDICTOR_DIR")
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-samples/${EXP_NAME}_infer}"
mkdir -p "$BASE_OUTPUT_DIR"

echo "========================================"
echo "Wan2.1 LinCa sampling"
echo "========================================"
echo "  Wan checkpoint: $CKPT_DIR"
echo "  Predictor:      $PREDICTOR_DIR"
echo "  Task:           $TASK  $SIZE  $FRAME_NUM frames  $SAMPLE_STEPS steps"
echo "  Intervals:      ${INTERVALS[*]}"
echo "  Output root:    $BASE_OUTPUT_DIR"
echo ""

for INTERVAL in "${INTERVALS[@]}"; do
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/interval_${INTERVAL}"
    mkdir -p "$OUTPUT_DIR"

    echo "---------- interval $INTERVAL ----------"
    python sample_wan.py \
        --task "$TASK" \
        --ckpt_dir "$CKPT_DIR" \
        --size "$SIZE" \
        --frame_num "$FRAME_NUM" \
        --sample_steps "$SAMPLE_STEPS" \
        --base_seed "$BASE_SEED" \
        --vbench-json-path "$VBENCH_JSON" \
        --index-start "$INDEX_START" \
        --index-end "$INDEX_END" \
        --save_folder "$OUTPUT_DIR" \
        --interval "$INTERVAL" \
        --forecast-steps "$INTERVAL" \
        --max-order "$MAX_ORDER" \
        --min-order "$MIN_ORDER" \
        --first-enhance "$FIRST_ENHANCE" \
        --forecast-method "$FORECAST_METHOD" \
        --decompose-method learned \
        --learned-checkpoint-stage1 "$STAGE1" \
        --learned-checkpoint-stage2 "$STAGE2"

    echo ">>> interval $INTERVAL done: $OUTPUT_DIR"
    echo ""
done

echo "========================================"
echo "All intervals finished. Videos under: $BASE_OUTPUT_DIR"
echo "Score them with VBench (https://github.com/Vchitect/VBench)."
echo "========================================"
