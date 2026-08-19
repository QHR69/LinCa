#!/bin/bash
# Generate cached features

# 检查prompt文件
PROMPT_FILE="prompts/DrawBench200.txt"
if [ ! -f "$PROMPT_FILE" ]; then
    echo "错误: prompt文件不存在: $PROMPT_FILE"
    echo "请确认prompts/DrawBench200.txt文件存在"
    exit 1
fi

# Output directory
OUTPUT_DIR="cache_data"

# 模型参数
MODEL_NAME="flux-dev"
NUM_STEPS=50
WIDTH=1024
HEIGHT=1024
SEED=0

# prompt范围（可分批生成）
START_IDX=${1:-0}      # 默认从0开始
END_IDX=${2:-200}      # 默认生成200个（DrawBench200）

echo "========================================"
echo "Generate Flux cached features"
echo "========================================"
echo ""
echo "配置:"
echo "  Prompt文件: $PROMPT_FILE"
echo "  Output directory: $OUTPUT_DIR"
echo "  模型: $MODEL_NAME"
echo "  步数: $NUM_STEPS"
echo "  Prompt范围: $START_IDX - $END_IDX"
echo ""

python src/flux/generate_cache_features.py \
    --prompt_file $PROMPT_FILE \
    --output_dir $OUTPUT_DIR \
    --model_name $MODEL_NAME \
    --num_steps $NUM_STEPS \
    --width $WIDTH \
    --height $HEIGHT \
    --seed $SEED \
    --start_idx $START_IDX \
    --end_idx $END_IDX

echo ""
echo "========================================"
echo "特征生成完成!"
echo "数据保存在: $OUTPUT_DIR"
echo "========================================"
echo ""
echo "下一步: 运行训练"
echo "  bash run_train.sh"
