#!/bin/bash
# Generate FLUX.1-dev cache features for LinCa predictor training.
#
# src/flux/ is a FLUX-derived toolkit that both backbones in this folder share:
# train_encoder_decoder.py reads cached features off disk and is backbone-agnostic,
# so the same trainer produces FLUX and HunyuanVideo predictors. This script is the
# FLUX side of feature extraction; generate_features.sh is the HunyuanVideo side.
#
# Usage:
#   bash generate_features_flux.sh [start_idx] [end_idx]
#
# Needs FLUX.1-dev weights. src/flux/util.py reads FLUX_MODEL and FLUX_AE:
#   export FLUX_MODEL=/path/to/FLUX.1-dev/flux1-dev.safetensors
#   export FLUX_AE=/path/to/FLUX.1-dev/ae.safetensors

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH="${SCRIPT_DIR}/src:${PYTHONPATH}"

FLUX_CHECKPOINT_DIR="${FLUX_CHECKPOINT_DIR:-./checkpoints/FLUX.1-dev}"
if [ -d "$FLUX_CHECKPOINT_DIR" ]; then
    export FLUX_MODEL="${FLUX_MODEL:-${FLUX_CHECKPOINT_DIR}/flux1-dev.safetensors}"
    export FLUX_AE="${FLUX_AE:-${FLUX_CHECKPOINT_DIR}/ae.safetensors}"
fi

PROMPT_FILE="${PROMPT_FILE:-prompts/DrawBench200.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-./cache_data/flux}"

MODEL_NAME="${MODEL_NAME:-flux-dev}"
NUM_STEPS="${NUM_STEPS:-50}"
WIDTH="${WIDTH:-1024}"
HEIGHT="${HEIGHT:-1024}"
SEED="${SEED:-0}"

# Prompt range, so extraction can be sharded across runs.
START_IDX="${1:-0}"
END_IDX="${2:-200}"

if [ ! -f "$PROMPT_FILE" ]; then
    echo "Prompt file not found: $PROMPT_FILE"
    exit 1
fi

echo "========================================"
echo "FLUX.1-dev cache feature extraction"
echo "========================================"
echo "  Prompts:      $PROMPT_FILE"
echo "  Output:       $OUTPUT_DIR"
echo "  Model:        $MODEL_NAME"
echo "  Steps:        $NUM_STEPS"
echo "  Resolution:   ${WIDTH}x${HEIGHT}"
echo "  Prompt range: $START_IDX - $END_IDX"
echo "========================================"

python src/flux/generate_cache_features.py \
    --prompt_file "$PROMPT_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --model_name "$MODEL_NAME" \
    --num_steps "$NUM_STEPS" \
    --width "$WIDTH" \
    --height "$HEIGHT" \
    --seed "$SEED" \
    --start_idx "$START_IDX" \
    --end_idx "$END_IDX"

echo ""
echo "========================================"
echo "Feature extraction complete: $OUTPUT_DIR"
echo "========================================"
echo ""
echo "Next: train a predictor on these features, e.g."
echo "  bash run_train.sh    # after pointing DATA_DIR at $OUTPUT_DIR"
