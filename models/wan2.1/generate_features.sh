#!/bin/bash
# Stage 1 of the Wan2.1 pipeline: dump per-step cumulative residual features from
# a full 50-step run, which the LinCa predictor is then trained on.
#
# Usage:
#   bash generate_features.sh <wan_ckpt_dir> [start_idx] [end_idx]
#
# Override any of the environment variables below to change the capture setting.

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH="${SCRIPT_DIR}/src:${PYTHONPATH}"

CKPT_DIR="${1:-${WAN_CKPT_DIR:-./checkpoints/Wan2.1-T2V-1.3B}}"
START_IDX="${2:-0}"
END_IDX="${3:-946}"

OUTPUT_DIR="${OUTPUT_DIR:-./cache_data/wan}"
VBENCH_JSON="${VBENCH_JSON:-./data/VBench_full_info.json}"
TASK="${TASK:-t2v-1.3B}"
SIZE="${SIZE:-832*480}"
FRAME_NUM="${FRAME_NUM:-81}"
SAMPLE_STEPS="${SAMPLE_STEPS:-50}"
BASE_SEED="${BASE_SEED:-42}"

if [ ! -d "$CKPT_DIR" ]; then
    echo "Wan checkpoint directory not found: $CKPT_DIR"
    echo "Usage: bash generate_features.sh <wan_ckpt_dir> [start_idx] [end_idx]"
    echo "Download with:"
    echo "  huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir $CKPT_DIR"
    exit 1
fi

if [ ! -f "$VBENCH_JSON" ]; then
    echo "VBench prompt file not found: $VBENCH_JSON"
    echo "Set VBENCH_JSON to VBench_full_info.json, or pass --prompt_file to the"
    echo "Python script directly to use a plain text prompt list instead."
    exit 1
fi

echo "========================================"
echo "Wan2.1 cache feature generation"
echo "========================================"
echo "  Checkpoint:   $CKPT_DIR"
echo "  Task:         $TASK"
echo "  Resolution:   $SIZE, $FRAME_NUM frames, $SAMPLE_STEPS steps"
echo "  Prompts:      $VBENCH_JSON [$START_IDX, $END_IDX)"
echo "  Output:       $OUTPUT_DIR"
echo ""

python generate_cache_features.py \
    --ckpt_dir "$CKPT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --task "$TASK" \
    --size "$SIZE" \
    --frame_num "$FRAME_NUM" \
    --sample_steps "$SAMPLE_STEPS" \
    --base_seed "$BASE_SEED" \
    --vbench-json-path "$VBENCH_JSON" \
    --index-start "$START_IDX" \
    --index-end "$END_IDX"

echo ""
echo "Features written to: $OUTPUT_DIR"
echo "Next step: bash run_train.sh"
