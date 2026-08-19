#!/bin/bash
# qwen_edit LinCA 两阶段训练
# 与 qwen_image 框架一致: loss, interval, cond/uncond, Early Stopping B

WANDB_KEY=${1:-""}

CACHE_DATA_DIR="./cache_data/qwen_edit"
DATASET_PATH="./data/gedit_bench"
if [ ! -d "$CACHE_DATA_DIR" ]; then
    echo "错误: cache 目录不存在: $CACHE_DATA_DIR"
    exit 1
fi
if [ ! -f "$CACHE_DATA_DIR/index.json" ]; then
    echo "错误: index.json 不存在"
    exit 1
fi

echo "========================================"
echo "qwen_edit 两阶段训练: Stage1(0-24) + Stage2(25-49)"
echo "========================================"
echo "数据: 22 val + 180 train + 11 display"
echo ""

BATCH_SIZE=10
EPOCHS=50
LR=1e-4
WEIGHT_DECAY=1e-5
INTERVALS="6,7,8,9,10,11,12"
USE_AMP="--amp"
GRAD_ACCUM=10
SEED=0

NUM_BLOCKS=3
HIDDEN_DIM=256
SPLIT_DIMS="2048,512,512"
DROPOUT=0.1
Z_LOSS_WEIGHT=0.1

PROMPTS_PER_EPOCH=40
EARLY_STOP_PATIENCE=20
EVAL_INTERVAL=1
SAVE_INTERVAL=1

OUTPUT_DIR="outputs/qwen_edit_linca_two_stage"
EXP_NAME="qwen_edit_two_stage_b${NUM_BLOCKS}_h${HIDDEN_DIM}"
WANDB_PROJECT="qwen_edit_linca_two_stage"
MODEL_PATH="Qwen/Qwen-Image-Edit"

cd "$(dirname "$0")"

if [ -n "$WANDB_KEY" ]; then
    python train.py \
        --cache_data_dir $CACHE_DATA_DIR \
        --dataset_path $DATASET_PATH \
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
        --eval_interval $EVAL_INTERVAL \
        --save_interval $SAVE_INTERVAL \
        --model_path $MODEL_PATH \
        --output_dir $OUTPUT_DIR \
        --exp_name $EXP_NAME \
        --wandb_project $WANDB_PROJECT \
        --wandb_key "$WANDB_KEY" \
        --seed $SEED \
        $USE_AMP
else
    python train.py \
        --cache_data_dir $CACHE_DATA_DIR \
        --dataset_path $DATASET_PATH \
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
        --eval_interval $EVAL_INTERVAL \
        --save_interval $SAVE_INTERVAL \
        --model_path $MODEL_PATH \
        --output_dir $OUTPUT_DIR \
        --exp_name $EXP_NAME \
        --no_wandb \
        --seed $SEED \
        $USE_AMP
fi

echo ""
echo "训练完成! 模型保存在: $OUTPUT_DIR/$WANDB_PROJECT/$EXP_NAME/"
