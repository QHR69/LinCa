#!/bin/bash
# HunyuanVideo 可逆网络训练脚本
# 方案四: 混合架构 (Glow + RevNet) 去门控版本

# 检查数据目录（使用已收集的VBench特征数据）
DATA_DIR="./cache_data/hunyuan"
if [ ! -d "$DATA_DIR" ]; then
    echo "错误: 数据目录不存在: $DATA_DIR"
    exit 1
fi

# 设置 PYTHONPATH (使用绝对路径)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH="${SCRIPT_DIR}/src:${PYTHONPATH}"

echo "========================================"
echo "HunyuanVideo 可逆网络训练"
echo "========================================"
echo ""
echo "改进特性:"
echo "  - 去门控: 移除可学习门控,简化模型"
echo "  - 可配置分区: split_dims=2304,384,384"
echo "  - 修复归一化: norm_factor 去掉平方"
echo "  - interval随机: [3,4,5,6,7,8,9,10,11,12]均匀采样"
echo "  - VBench 946 prompts"
echo "  - Dropout防过拟合"
echo "  - TensorBoard + Wandb 日志记录"
echo ""

# === 训练参数 ===
BATCH_SIZE=1
EPOCHS=100
LR=2e-5
WEIGHT_DECAY=1e-5
INTERVALS=${INTERVALS:-"3,4,5,6,7,8,9,10"}
USE_AMP="--amp"
GRAD_ACCUM=64
SEED=42

# === 模型参数 (支持环境变量覆盖，用于调参) ===
NUM_BLOCKS=${NUM_BLOCKS:-2}
HIDDEN_DIM=${HIDDEN_DIM:-64}
SPLIT_DIMS=${SPLIT_DIMS:-"2304,384,384"}
DROPOUT=${DROPOUT:-0.0}

# === Loss 权重参数 ===
Z_LOSS_WEIGHT=0.1

# === 训练策略 ===
PROMPTS_PER_EPOCH=20   # 每epoch随机抽30个prompt训练（从训练池中抽，不含验证集）
EARLY_STOP_PATIENCE=20
EVAL_INTERVAL=1
SAVE_INTERVAL=1

# === 数据划分策略 ===
TRAIN_PROMPTS="0-946"    # VBench 946 prompts
NUM_VAL_PROMPTS=20       # 等间隔抽取50个做验证，剩余896个训练

# === Output directory ===
OUTPUT_DIR="outputs"
EXP_NAME="hunyuan_v4_learned_b_${NUM_BLOCKS}_h_${HIDDEN_DIM}_splits_${SPLIT_DIMS//,/x}_intervals_${INTERVALS//,/x}_epochs_${EPOCHS}_dropout_${DROPOUT}_seed_${SEED}"

# === Wandb ===
WANDB_PROJECT="hunyuan-v4"
# NO_WANDB="--no_wandb"

echo ">>> 开始训练 (使用TensorBoard + Wandb记录)"
python src/flux/train_encoder_decoder.py \
    --data_dir $DATA_DIR \
    --train_prompts $TRAIN_PROMPTS \
    --num_val_prompts $NUM_VAL_PROMPTS \
    --prompts_per_epoch $PROMPTS_PER_EPOCH \
    --shuffle_prompts \
    --random_interval \
    --early_stop_patience $EARLY_STOP_PATIENCE \
    --dim 3072 \
    --num_blocks $NUM_BLOCKS \
    --hidden_dim $HIDDEN_DIM \
    --split_dims $SPLIT_DIMS \
    --dropout $DROPOUT \
    --z_loss_weight $Z_LOSS_WEIGHT \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --weight_decay $WEIGHT_DECAY \
    --intervals $INTERVALS \
    --num_workers 4 \
    --grad_accum_steps $GRAD_ACCUM \
    --eval_interval $EVAL_INTERVAL \
    --save_interval $SAVE_INTERVAL \
    --output_dir $OUTPUT_DIR \
    --exp_name $EXP_NAME \
    --seed $SEED \
    --wandb_project $WANDB_PROJECT \
    ${NO_WANDB:-} \
    $USE_AMP

echo ""
echo "========================================"
echo "训练完成!"
echo "查看TensorBoard: tensorboard --logdir=$OUTPUT_DIR"
echo "查看Wandb: https://wandb.ai"
echo "========================================"
