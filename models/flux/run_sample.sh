#!/bin/bash
# Flux-v4 多阶段推理+评估脚本
# 循环跑多个 INTERVAL，评估后上传指标到 wandb

# === 设置环境变量 ===
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"

# 指向本地FLUX模型文件（避免重新下载）
FLUX_CHECKPOINT_DIR="${FLUX_CHECKPOINT_DIR:-./checkpoints/FLUX.1-dev}"
export FLUX_MODEL="${FLUX_CHECKPOINT_DIR}/flux1-dev.safetensors"
export FLUX_AE="${FLUX_CHECKPOINT_DIR}/ae.safetensors"

# === 多阶段配置 ===
# 从环境变量或命令行参数获取
CHECKPOINT_DIR="${1:-outputs/flux-v4-3-stage-re/flux_v4_3_stage_b_1_h_128_splits_2304x384x384_intervals_3x4x5x6x7x8x9x10x11x12_epochs_60_dropout_0.0_seed_21}"
STAGE_SPLITS="${2:-17,17,16}"
# STAGE_SPLITS="${2:-12,12,13,13}"

# 解析阶段数
IFS=',' read -ra SPLITS_ARR <<< "$STAGE_SPLITS"
NUM_STAGES=${#SPLITS_ARR[@]}

# 构建 checkpoint 路径列表
CHECKPOINT_ARGS=""
echo "========================================"
echo "多阶段推理: ${NUM_STAGES} stages (splits=${STAGE_SPLITS})"
start=0
for i in $(seq 0 $((NUM_STAGES - 1))); do
    CKPT="${CHECKPOINT_DIR}/best_predictor_stage${i}.pt"
    if [ ! -f "$CKPT" ]; then
        echo "错误: stage${i} checkpoint文件不存在: $CKPT"
        echo "Usage: bash run_sample.sh [checkpoint_dir] [stage_splits]"
        echo "示例: bash run_sample.sh outputs/flux-v4-multi-stage/my_exp 12,12,13,13"
        exit 1
    fi
    size=${SPLITS_ARR[$i]}
    end=$((start + size - 1))
    echo "  Stage ${i} (step ${start}-${end}): $CKPT"
    CHECKPOINT_ARGS="${CHECKPOINT_ARGS} ${CKPT}"
    start=$((end + 1))
done
echo "========================================"

# === 从checkpoint路径提取实验名称 ===
EXP_NAME=$(basename "$CHECKPOINT_DIR")
WANDB_RUN_NAME="${EXP_NAME}_infer"
echo "实验名称: $EXP_NAME"
echo "Wandb Run名称: $WANDB_RUN_NAME"
echo ""

# === 推理参数 ===
PROMPT_FILE="prompts/DrawBench200.txt"
SEED=21

# 模型参数
MODEL_NAME="flux-dev"
WIDTH=1024
HEIGHT=1024
NUM_STEPS=50
GUIDANCE=3.5
BATCH_SIZE=1

# Cache参数 (与训练一致)
MAX_ORDER=2
MIN_ORDER=0
FIRST_ENHANCE=3
FORECAST_METHOD="hermite"

# 评估参数
EVAL_SCRIPT="${EVAL_SCRIPT:-./tools/eval_images.py}"
BASELINE_DIR="${BASELINE_DIR:-./baselines/flux_seed21}"
EVAL_PROMPTS_FILE="prompts/DrawBench200.txt"

# Wandb参数
# WANDB_PROJECT="flux-v4-multi-stage"
WANDB_PROJECT="flux-v4-${NUM_STAGES}-stage-ablation"

# 设置 PYTHONPATH
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH="${SCRIPT_DIR}/src:${PYTHONPATH}"

# 检查prompt文件
if [ ! -f "$PROMPT_FILE" ]; then
    echo "错误: prompt文件不存在: $PROMPT_FILE"
    exit 1
fi

# === Output directory (使用实验名称而非时间戳) ===
BASE_OUTPUT_DIR="samples/${EXP_NAME}_infer"
mkdir -p "$BASE_OUTPUT_DIR"

echo "总Output directory: $BASE_OUTPUT_DIR"
echo ""

# ========== 第一阶段: 推理生成 ==========
for INTERVAL in 3 4 5 6 7 8 9 10 11 12; do
    FORECAST_STEPS=$INTERVAL
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/interval_${INTERVAL}"

    echo "========================================"
    echo "Flux-v4 多阶段推理生成 (INTERVAL=$INTERVAL, ${NUM_STAGES} stages)"
    echo "========================================"
    echo ""
    echo "配置:"
    echo "  Stage splits: $STAGE_SPLITS"
    echo "  Checkpoint dir: $CHECKPOINT_DIR"
    echo "  Prompt文件: $PROMPT_FILE"
    echo "  Output directory: $OUTPUT_DIR"
    echo "  模型: $MODEL_NAME"
    echo "  尺寸: ${WIDTH}x${HEIGHT}"
    echo "  步数: $NUM_STEPS"
    echo "  Interval: $INTERVAL"
    echo "  Forecast Steps: $FORECAST_STEPS"
    echo "  Seed: $SEED"
    echo ""

    mkdir -p "$OUTPUT_DIR"

    echo ">>> 开始生成图像 (INTERVAL=$INTERVAL)..."
    python src/sample.py \
        --prompt_file "$PROMPT_FILE" \
        --model_name "$MODEL_NAME" \
        --output_dir "$OUTPUT_DIR" \
        --width $WIDTH \
        --height $HEIGHT \
        --num_steps $NUM_STEPS \
        --guidance $GUIDANCE \
        --seed $SEED \
        --batch_size $BATCH_SIZE \
        --interval $INTERVAL \
        --max_order $MAX_ORDER \
        --min_order $MIN_ORDER \
        --first_enhance $FIRST_ENHANCE \
        --forecast_method "$FORECAST_METHOD" \
        --decompose_method learned \
        --forecast_steps $FORECAST_STEPS \
        --learned_checkpoints $CHECKPOINT_ARGS \
        --stage_splits "$STAGE_SPLITS" \
        --add_sampling_metadata

    echo ""
    echo ">>> INTERVAL=$INTERVAL 生成完成! 输出: $OUTPUT_DIR"
    echo ""
done

echo "========================================"
echo "所有 INTERVAL 推理生成完成!"
echo "========================================"
echo ""

# ========== 第二阶段: 评估 + 上传Wandb ==========
echo "========================================"
echo "开始评估 + 上传Wandb"
echo "========================================"
echo ""

# 用一个Python脚本完成: 评估所有interval + 上传wandb
python -c "
import sys
sys.path.insert(0, '${IMAGEREWARD_DIR:-.}')
sys.path.insert(0, '${EVAL_ROOT:-.}')

import wandb
from eval_images import evaluate_images

# 参数
intervals = [3,4,5,6,7,8,9,10,11,12]
base_output_dir = '${BASE_OUTPUT_DIR}'
wandb_project = '${WANDB_PROJECT}'
wandb_run_name = '${WANDB_RUN_NAME}'
baseline_dir = '${BASELINE_DIR}'
prompts_file = '${EVAL_PROMPTS_FILE}'
checkpoint_dir = '${CHECKPOINT_DIR}'
stage_splits = '${STAGE_SPLITS}'
num_stages = ${NUM_STAGES}

# 初始化wandb
wandb.init(
    project=wandb_project,
    name=wandb_run_name,
    config={
        'exp_name': '${EXP_NAME}',
        'checkpoint_dir': checkpoint_dir,
        'stage_splits': stage_splits,
        'num_stages': num_stages,
        'model_name': '${MODEL_NAME}',
        'width': ${WIDTH},
        'height': ${HEIGHT},
        'num_steps': ${NUM_STEPS},
        'guidance': ${GUIDANCE},
        'seed': ${SEED},
        'max_order': ${MAX_ORDER},
        'min_order': ${MIN_ORDER},
        'first_enhance': ${FIRST_ENHANCE},
        'forecast_method': '${FORECAST_METHOD}',
        'intervals': intervals,
    },
)

print(f'Wandb initialized: project={wandb_project}, run={wandb_run_name}')

# 复用ImageReward模型，避免重复加载
imagereward_model = None
all_metrics = {}

for interval in intervals:
    image_dir = f'{base_output_dir}/interval_{interval}'
    print(f'')
    print(f'========== 评估 interval={interval} ==========')
    print(f'图片目录: {image_dir}')

    result = evaluate_images(
        image_dir=image_dir,
        prompts_file=prompts_file,
        baseline_dir=baseline_dir,
        imagereward_model=imagereward_model,
        skip_clip=False,
        skip_lpips=False,
        verbose=True,
    )

    # 复用模型
    if imagereward_model is None:
        imagereward_model = result.get('imagereward_model')

    # 上传到wandb (以interval为横坐标)
    metrics = {
        'imagereward': result.get('imagereward', 0.0),
        'num_images': result.get('num_images', 0),
    }
    if 'clip_score' in result:
        metrics['clip_score'] = result['clip_score']
    if 'psnr' in result:
        metrics['psnr'] = result['psnr']
    if 'ssim' in result:
        metrics['ssim'] = result['ssim']
    if 'lpips' in result:
        metrics['lpips'] = result['lpips']

    wandb.log(metrics, step=interval)
    print(f'>>> interval={interval} 指标已上传wandb (step={interval})')

    # 同时收集到all_metrics用于保存文件
    prefix = f'interval_{interval}'
    for key, val in metrics.items():
        all_metrics[f'{prefix}/{key}'] = val

print(f'')
print(f'所有指标已上传wandb，横坐标为interval')

# 保存结果到文件
import os
output_file = os.path.join(base_output_dir, 'evaluation_results.txt')
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(f\"{'='*80}\n\")
    f.write(f'评估结果汇总\n')
    f.write(f\"{'='*80}\n\")
    f.write(f'Checkpoint dir: {checkpoint_dir}\n')
    f.write(f'Stage splits: {stage_splits} ({num_stages} stages)\n')
    f.write(f'Baseline: {baseline_dir}\n')
    f.write(f'Prompts: {prompts_file}\n')
    f.write(f'Intervals: {intervals}\n\n')
    for interval in intervals:
        prefix = f'interval_{interval}'
        f.write(f'--- Interval {interval} ---\n')
        for key, val in all_metrics.items():
            if key.startswith(prefix):
                metric_name = key.split('/')[-1]
                f.write(f'  {metric_name}: {val}\n')
        f.write('\n')
print(f'结果已保存到: {output_file}')

wandb.finish()
print('')
print('Wandb run finished')
"

echo ""
echo "========================================"
echo "所有评估完成! 指标已上传Wandb"
echo "总Output directory: $BASE_OUTPUT_DIR"
echo "Wandb: https://wandb.ai (project: $WANDB_PROJECT, run: $WANDB_RUN_NAME)"
echo "========================================"
