#!/bin/bash
# Qwen-Image-Edit: two-stage LinCa predictor training across GPUs (DDP).
# Usage: ./run_train_ddp.sh [WANDB_KEY]
# Uses every visible GPU by default; override with NPROC, e.g. NPROC=2 ./run_train_ddp.sh
# If the port is taken: MASTER_PORT=29501 NPROC=8 ./run_train_ddp.sh <WANDB_KEY>

WANDB_KEY=${1:-""}
NPROC=${NPROC:-$(nvidia-smi -L | wc -l)}

CACHE_DATA_DIR="./cache_data/qwen_edit"
DATASET_PATH="./data/gedit_bench"
if [ ! -d "$CACHE_DATA_DIR" ]; then
    echo "Error: cache directory not found: $CACHE_DATA_DIR"
    exit 1
fi
if [ ! -f "$CACHE_DATA_DIR/index.json" ]; then
    echo "Error: index.json not found"
    exit 1
fi

echo "========================================"
echo "Qwen-Image-Edit two-stage training (DDP): $NPROC GPUs"
echo "Stage1(0-24) + Stage2(25-49)"
echo "========================================"
echo "Data: 180 train + 22 val + 11 display"
echo ""

BATCH_SIZE=10
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

OUTPUT_DIR="outputs/qwen_edit_linca_two_stage6512_b16_40_10_8"
EXP_NAME="qwen_edit_two_stage_b${NUM_BLOCKS}_h${HIDDEN_DIM}_ddp6512_b16_40_10_8"
WANDB_PROJECT="qwen_edit_linca_two_stage_6512_b16_40_10_8"
MODEL_PATH="Qwen/Qwen-Image-Edit"

cd "$(dirname "$0")"

# Rank 0 renders 11 eval images (10-15 min) while the others wait at a barrier, so raise the NCCL timeout above its 10-minute default
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

# Prefer torchrun, falling back to python -m torch.distributed.run
if command -v torchrun &>/dev/null; then
    LAUNCHER="torchrun --nproc_per_node=$NPROC"
else
    LAUNCHER="python -m torch.distributed.run --nproc_per_node=$NPROC"
fi

if [ -n "$WANDB_KEY" ]; then
    $LAUNCHER train_ddp.py $ARGS --wandb_key "$WANDB_KEY"
else
    $LAUNCHER train_ddp.py $ARGS --no_wandb
fi

echo ""
echo "Training complete. Checkpoints: $OUTPUT_DIR/$WANDB_PROJECT/$EXP_NAME/"
