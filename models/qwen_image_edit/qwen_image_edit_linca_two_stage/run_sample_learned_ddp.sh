#!/bin/bash
# 多卡推理 1212 张
# Usage: bash run_sample_learned_ddp.sh <checkpoint_dir> [nproc]
CHECKPOINT_DIR=${1:-"outputs/qwen_edit_linca_two_stage/qwen_edit_linca_two_stage/qwen_edit_two_stage_b3_h256"}
NPROC=${2:-4}
DATASET_PATH="./data/gedit_bench"
OUTPUT_DIR="samples/qwen_edit_1212_full"

cd "$(dirname "$0")"

torchrun --nproc_per_node=$NPROC sample_learned_ddp.py \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --dataset_path $DATASET_PATH \
    --output_dir $OUTPUT_DIR \
    --seed 0 \
    --interval 7
