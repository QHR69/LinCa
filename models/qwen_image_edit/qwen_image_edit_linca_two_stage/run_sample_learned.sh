#!/bin/bash
# 单卡推理 1212 张
# Usage: bash run_sample_learned.sh [checkpoint或checkpoint_dir]
CKPT="${1:-outputs/qwen_edit_linca_two_stage/qwen_edit_linca_two_stage/qwen_edit_two_stage_b3_h256}"
DATASET_PATH="./data/gedit_bench"
OUTPUT_DIR="samples/qwen_edit_1212_full"

cd "$(dirname "$0")"

if [ -d "$CKPT" ]; then
    python sample_learned.py --checkpoint_dir "$CKPT" --dataset_path $DATASET_PATH --output_dir $OUTPUT_DIR --seed 0 --interval 7
else
    python sample_learned.py --checkpoint "$CKPT" --dataset_path $DATASET_PATH --output_dir $OUTPUT_DIR --seed 0 --interval 7
fi
