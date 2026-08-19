#!/bin/bash
# Two-stage training: Stage1=0-24, Stage2=25-49
#
# 核心特性:
# - 两个相同架构的可逆网络，分别训练前后25步
# - Early stopping B: stop only after both stages stall for N epochs
# - 其它框架与 without_gate 一致

# 获取 wandb key（可选）
WANDB_KEY=${1:-""}

# 检查数据目录
DATA_DIR="./cache_data"
if [ ! -d "$DATA_DIR" ]; then
    echo "错误: 数据目录不存在: $DATA_DIR"
    exit 1
fi

if [ ! -f "prompts/eval.txt" ]; then
    echo "错误: eval prompts文件不存在"
    exit 1
fi

echo "========================================"
echo "两阶段训练: Stage1(0-24) + Stage2(25-49)"
echo "========================================"
echo ""
echo "核心特性:"
echo "  - 双 predictor: 同架构不同参数"
echo "  - Early Stop B: 两个 stage 都无提升才停"
echo "  - 可配置分区: split_dims=1024,1024,1024"
echo "  - interval随机: [6,7,8,9,10,11,12]"
echo ""

# === 训练参数 ===
BATCH_SIZE=32
EPOCHS=50
LR=1e-4
WEIGHT_DECAY=1e-5
INTERVALS="6,7,8,9,10,11,12"
USE_AMP="--amp"
GRAD_ACCUM=5
SEED=0

# === 模型参数 ===
NUM_BLOCKS=1
HIDDEN_DIM=256
SPLIT_DIMS="2048,512,512"
DROPOUT=0

# === Loss 权重 ===
Z_LOSS_WEIGHT=0.1

# === 训练策略 ===
PROMPTS_PER_EPOCH=50
EARLY_STOP_PATIENCE=20
EVAL_INTERVAL=1
SAVE_INTERVAL=1

# === 数据划分 ===
TRAIN_PROMPTS="200-400"
VAL_SAMPLE_RANGE="0-200"
NUM_VAL_PROMPTS=25

# === Output directory ===
OUTPUT_DIR="outputs/linca_v4_two_stage_2048512512_b${NUM_BLOCKS}_h${HIDDEN_DIM}_32_5_50"
EXP_NAME="two_stage_2048512512_v4_b${NUM_BLOCKS}_h${HIDDEN_DIM}_32_5_50"
WANDB_PROJECT="linca_v4_two_stage_2048512512_1256_32_5_50"

if [ -n "$WANDB_KEY" ]; then
    echo ">>> 使用 wandb 监控训练"
    python train.py \
        --data_dir $DATA_DIR \
        --train_prompts $TRAIN_PROMPTS \
        --val_sample_range $VAL_SAMPLE_RANGE \
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
        --num_workers 0 \
        --grad_accum_steps $GRAD_ACCUM \
        --eval_prompts_file prompts/eval.txt \
        --eval_interval $EVAL_INTERVAL \
        --save_interval $SAVE_INTERVAL \
        --output_dir $OUTPUT_DIR \
        --exp_name $EXP_NAME \
        --wandb_project $WANDB_PROJECT \
        --wandb_key "$WANDB_KEY" \
        --seed $SEED \
        $USE_AMP
else
    echo ">>> 不使用 wandb（无API key）"
    python train.py \
        --data_dir $DATA_DIR \
        --train_prompts $TRAIN_PROMPTS \
        --val_sample_range $VAL_SAMPLE_RANGE \
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
        --num_workers 0 \
        --grad_accum_steps $GRAD_ACCUM \
        --eval_prompts_file prompts/eval.txt \
        --eval_interval $EVAL_INTERVAL \
        --save_interval $SAVE_INTERVAL \
        --output_dir $OUTPUT_DIR \
        --exp_name $EXP_NAME \
        --no_wandb \
        --seed $SEED \
        $USE_AMP
fi

echo ""
echo "========================================"
echo "训练完成！"
echo "模型保存在: $OUTPUT_DIR/"
echo "  - best_predictor_stage1.pt"
echo "  - best_predictor_stage2.pt"
echo "========================================"
