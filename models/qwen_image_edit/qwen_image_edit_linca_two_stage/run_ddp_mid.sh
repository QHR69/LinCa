#!/bin/bash
set -e
set -o pipefail

# qwen_edit LinCA 两阶段多卡训练 (DDP)
# Usage: ./run_ddp_mid.sh [WANDB_KEY]
# 示例: NPROC=8 MASTER_PORT=29602 bash ./run_ddp_mid.sh wandb_v1_xxx
# 端口占用时换 MASTER_PORT；OOM 时确保 8 张 GPU 空闲或减小 BATCH_SIZE

WANDB_KEY=${1:-""}
NPROC=${NPROC:-$(nvidia-smi -L | wc -l)}
MASTER_PORT=${MASTER_PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}

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
echo "qwen_edit 两阶段多卡训练 (DDP): $NPROC GPUs"
echo "Stage1(0-24) + Stage2(25-49)"
echo "========================================"
echo "数据: 22 val + 180 train + 11 display"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<not set>}"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo ""

BATCH_SIZE=8
EPOCHS=${EPOCHS:-50}
LR=1e-4
WEIGHT_DECAY=1e-5
INTERVALS="6,7,8,9,10,11,12"
USE_AMP="--amp"
GRAD_ACCUM=8
SEED=0

NUM_BLOCKS=6
HIDDEN_DIM=512
SPLIT_DIMS="2048,512,512"
DROPOUT=0
Z_LOSS_WEIGHT=0.1

PROMPTS_PER_EPOCH=40
EARLY_STOP_PATIENCE=20
EVAL_INTERVAL=1
SAVE_INTERVAL=1

OUTPUT_DIR="outputs/qwen_edit_linca_two_stage6512_b16_40_mid"
EXP_NAME="qwen_edit_two_stage_b${NUM_BLOCKS}_h${HIDDEN_DIM}_ddp6512_b16_40_mid"
WANDB_PROJECT="qwen_edit_linca_two_stage_6512_b16_40_mid"
MODEL_PATH="Qwen/Qwen-Image-Edit"

cd "$(dirname "$0")"

# eval 仅 rank0 跑 11 张图约 10–15 分钟，其他 rank 在 barrier 等待；避免 NCCL 默认 10 分钟超时
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-7200}

ARGS="--cache_data_dir $CACHE_DATA_DIR \
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
    --seed $SEED \
    $USE_AMP"

# 优先使用 torchrun，否则用 python -m torch.distributed.run
if command -v torchrun &>/dev/null; then
    LAUNCHER="torchrun --nproc_per_node=$NPROC --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT"
else
    LAUNCHER="python -m torch.distributed.run --nproc_per_node=$NPROC --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT"
fi

if [ -n "$WANDB_KEY" ]; then
    $LAUNCHER train_ddp.py $ARGS --wandb_key "$WANDB_KEY"
else
    $LAUNCHER train_ddp.py $ARGS --no_wandb
fi

echo ""
echo "训练完成! 模型保存在: $OUTPUT_DIR/$WANDB_PROJECT/$EXP_NAME/"