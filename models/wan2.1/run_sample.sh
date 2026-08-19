#!/bin/bash
# HunyuanVideo 推理+VBench评估脚本
# 使用训练好的 learned predictor 进行 cache 加速推理
# 循环跑多个 INTERVAL，生成视频用于 VBench 评估

# === 设置环境变量 ===
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH="${SCRIPT_DIR}/src:${PYTHONPATH}"
export MODEL_BASE=./checkpoints/hunyuanvideo

# === 检查 checkpoint ===
CHECKPOINT="${1:-outputs/hunyuan_v4_learned_b_6_h_1024_splits_2304x384x384_intervals_3x4x5x6x7x8x9x10_epochs_100_dropout_0.0_seed_42_20260222_214417/best_predictor.pt}"

if [ ! -f "$CHECKPOINT" ]; then
    echo "错误: checkpoint文件不存在: $CHECKPOINT"
    echo "Usage: bash run_sample.sh [checkpoint_path]"
    echo "示例: bash run_sample.sh outputs/hunyuan_v4_learned_.../best_predictor.pt"
    exit 1
fi

# === 从 checkpoint 路径提取实验名称 ===
EXP_NAME=$(basename $(dirname "$CHECKPOINT"))
echo "实验名称: $EXP_NAME"
echo "Checkpoint: $CHECKPOINT"

# === 模型路径 ===
MODEL_CKPTS=./checkpoints/hunyuanvideo
DIT_WEIGHT=${MODEL_CKPTS}/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt
VBENCH_JSON=./data/VBench_full_info.json

# === 推理参数 ===
SEED=42

# Cache参数 (与训练一致)
MAX_ORDER=2
MIN_ORDER=0
FIRST_ENHANCE=3
FORECAST_METHOD="hermite"

# === Output directory ===
BASE_OUTPUT_DIR="samples/${EXP_NAME}_infer"
mkdir -p "$BASE_OUTPUT_DIR"

echo "总Output directory: $BASE_OUTPUT_DIR"
echo ""

# ========== 推理生成 ==========
for INTERVAL in 6; do
    FORECAST_STEPS=$INTERVAL
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/interval_${INTERVAL}"

    echo "========================================"
    echo "HunyuanVideo 推理 (INTERVAL=$INTERVAL)"
    echo "========================================"
    echo ""
    echo "配置:"
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
    echo ">>> INTERVAL=$INTERVAL 生成完成! 输出: $OUTPUT_DIR"
    echo ""
done

echo "========================================"
echo "所有 INTERVAL 推理完成!"
echo "视频目录: $BASE_OUTPUT_DIR"
echo ""
echo "可使用 VBench 评估:"
echo "  python eval/vbench/calc_vbench.py --video_path $BASE_OUTPUT_DIR/interval_X"
echo "========================================"
