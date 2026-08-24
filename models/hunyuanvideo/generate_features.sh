#!/bin/bash
# Generate HunyuanVideo cache features for LinCa predictor training.
#
# Runs full 50-step HunyuanVideo inference and dumps the intermediate DiT
# features that train_encoder_decoder.py later learns to predict. Feature
# extraction must use the same video size / length / step count as the sampling
# you intend to accelerate, so the defaults here mirror run_sample.sh.
#
# Usage:
#   bash generate_features.sh [index_start] [index_end]
#
#   bash generate_features.sh            # the whole prompt set
#   bash generate_features.sh 0 200      # first 200 prompts, to shard the job
#
# Prompts come from VBench when ./data/VBench_full_info.json is present (this is
# what run_train.sh trains on, --train_prompts 0-946); otherwise a plain text
# file with one prompt per line is used.

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH="${SCRIPT_DIR}/src:${PYTHONPATH}"

MODEL_BASE="${MODEL_BASE:-./checkpoints/hunyuanvideo}"
DIT_WEIGHT="${DIT_WEIGHT:-${MODEL_BASE}/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt}"

# Must match run_train.sh's --data_dir.
OUTPUT_DIR="${OUTPUT_DIR:-./cache_data/hunyuan}"

VBENCH_JSON="${VBENCH_JSON:-./data/VBench_full_info.json}"
PROMPT_FILE="${PROMPT_FILE:-prompts/DrawBench200.txt}"

# Sampling configuration -- keep in step with run_sample.sh.
VIDEO_SIZE="${VIDEO_SIZE:-480 640}"
VIDEO_LENGTH="${VIDEO_LENGTH:-65}"
INFER_STEPS="${INFER_STEPS:-50}"
SEED="${SEED:-42}"

INDEX_START="${1:-0}"
INDEX_END="${2:-0}"   # 0 lets the script use every prompt

if [ ! -d "$MODEL_BASE" ]; then
    echo "HunyuanVideo weights not found: $MODEL_BASE"
    echo "Download them from Tencent-Hunyuan/HunyuanVideo and set MODEL_BASE."
    exit 1
fi

echo "========================================"
echo "HunyuanVideo cache feature extraction"
echo "========================================"
echo "  Model base:   $MODEL_BASE"
echo "  Output:       $OUTPUT_DIR"
echo "  Video size:   $VIDEO_SIZE   length: $VIDEO_LENGTH   steps: $INFER_STEPS"

if [ -f "$VBENCH_JSON" ]; then
    echo "  Prompts:      $VBENCH_JSON (VBench)"
    echo "  Index range:  $INDEX_START - $INDEX_END"
    echo "========================================"
    python generate_cache_features.py \
        --output_dir "$OUTPUT_DIR" \
        --vbench-json-path "$VBENCH_JSON" \
        --index-start "$INDEX_START" \
        --index-end "$INDEX_END" \
        --model-base "$MODEL_BASE" \
        --dit-weight "$DIT_WEIGHT" \
        --video-size $VIDEO_SIZE \
        --video-length "$VIDEO_LENGTH" \
        --infer-steps "$INFER_STEPS" \
        --flow-reverse \
        --seed "$SEED" \
        --use-cpu-offload
else
    if [ ! -f "$PROMPT_FILE" ]; then
        echo "Neither $VBENCH_JSON nor $PROMPT_FILE exists."
        echo "Provide VBench's VBench_full_info.json, or set PROMPT_FILE to a"
        echo "text file with one prompt per line."
        exit 1
    fi
    echo "  Prompts:      $PROMPT_FILE (text file; VBench JSON not found)"
    END_ARG=""
    if [ "$INDEX_END" -gt 0 ]; then
        END_ARG="--end_idx $INDEX_END"
    fi
    echo "========================================"
    python generate_cache_features.py \
        --output_dir "$OUTPUT_DIR" \
        --prompt_file "$PROMPT_FILE" \
        --start_idx "$INDEX_START" \
        $END_ARG \
        --model-base "$MODEL_BASE" \
        --dit-weight "$DIT_WEIGHT" \
        --video-size $VIDEO_SIZE \
        --video-length "$VIDEO_LENGTH" \
        --infer-steps "$INFER_STEPS" \
        --flow-reverse \
        --seed "$SEED" \
        --use-cpu-offload
fi

echo ""
echo "========================================"
echo "Feature extraction complete: $OUTPUT_DIR"
echo "========================================"
echo ""
echo "Next: bash run_train.sh"
