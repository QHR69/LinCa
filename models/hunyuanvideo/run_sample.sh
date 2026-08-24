#!/bin/bash
# HunyuanVideo cache-accelerated sampling, for VBench scoring.
# Uses a trained LinCa predictor to skip most DiT evaluations,
# sweeping several intervals and writing videos for VBench to score.

# === Environment ===
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH="${SCRIPT_DIR}/src:${PYTHONPATH}"
export MODEL_BASE=./checkpoints/hunyuanvideo

# === Check the checkpoint ===
CHECKPOINT="${1:-outputs/hunyuan_linca_b_6_h_1024_splits_2304x384x384_intervals_3x4x5x6x7x8x9x10_epochs_100_dropout_0.0_seed_42_20260222_214417/best_predictor.pt}"

if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: checkpoint not found: $CHECKPOINT"
    echo "Usage: bash run_sample.sh [checkpoint_path]"
    echo "Example: bash run_sample.sh outputs/hunyuan_linca_.../best_predictor.pt"
    exit 1
fi

# === Derive the experiment name from the checkpoint path ===
EXP_NAME=$(basename $(dirname "$CHECKPOINT"))
echo "Experiment: $EXP_NAME"
echo "Checkpoint: $CHECKPOINT"

# === Backbone weights ===
MODEL_CKPTS=./checkpoints/hunyuanvideo
DIT_WEIGHT=${MODEL_CKPTS}/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt
VBENCH_JSON=./data/VBench_full_info.json

# === Sampling ===
SEED=42

# Cache settings, which must match training
MAX_ORDER=2
MIN_ORDER=0
FIRST_ENHANCE=3
FORECAST_METHOD="hermite"

# === Output directory ===
BASE_OUTPUT_DIR="samples/${EXP_NAME}_infer"
mkdir -p "$BASE_OUTPUT_DIR"

echo "Output root: $BASE_OUTPUT_DIR"
echo ""

# ========== Generate ==========
for INTERVAL in 6; do
    FORECAST_STEPS=$INTERVAL
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/interval_${INTERVAL}"

    echo "========================================"
    echo "HunyuanVideo sampling (interval=$INTERVAL)"
    echo "========================================"
    echo ""
    echo "Settings:"
    echo "  Checkpoint: $CHECKPOINT"
    echo "  Output directory: $OUTPUT_DIR"
    echo "  Interval: $INTERVAL"
    echo "  Forecast Steps: $FORECAST_STEPS"
    echo "  Seed: $SEED"
    echo ""

    python sample_video.py \
        --vbench-json-path ${VBENCH_JSON} \
        --index-start 0 \
        --index-end 945 \
        --model-base ${MODEL_CKPTS} \
        --dit-weight ${DIT_WEIGHT} \
        --video-size 480 640 \
        --video-length 65 \
        --infer-steps 50 \
        --flow-reverse \
        --seed $SEED \
        --save-path "$OUTPUT_DIR" \
        --use-cpu-offload \
        --interval $INTERVAL \
        --max-order $MAX_ORDER \
        --min-order $MIN_ORDER \
        --first-enhance $FIRST_ENHANCE \
        --forecast-method "$FORECAST_METHOD" \
        --decompose-method learned \
        --forecast-steps $FORECAST_STEPS \
        --learned-checkpoint "$CHECKPOINT"

    echo ""
    echo ">>> interval $INTERVAL done. Output: $OUTPUT_DIR"
    echo ""
done

echo "========================================"
echo "All intervals finished."
echo "Videos: $BASE_OUTPUT_DIR"
echo ""
echo "Score them with VBench:"
echo "  python eval/vbench/calc_vbench.py --video_path $BASE_OUTPUT_DIR/interval_X"
echo "========================================"
