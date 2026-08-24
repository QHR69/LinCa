#!/bin/bash
# Two-stage training: Stage1=0-24, Stage2=25-49
#
# Key points:
# - Two invertible networks of identical architecture, one per half of the schedule
# - Early stopping B: stop only after both stages stall for N epochs
# - Everything else follows the single-predictor recipe

# Optional wandb API key, taken as the first argument.
WANDB_KEY=${1:-""}

# The cache features must already exist.
DATA_DIR="./cache_data"
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: feature directory not found: $DATA_DIR"
    exit 1
fi

if [ ! -f "prompts/eval.txt" ]; then
    echo "Error: eval prompts file not found"
    exit 1
fi

echo "========================================"
echo "Two-stage training: stage 1 = steps 0-24, stage 2 = steps 25-49"
echo "========================================"
echo ""
echo "Key points:"
echo "  - Two predictors: same architecture, separate weights"
echo "  - Early stopping waits until both stages stop improving"
echo "  - Partition sizes: split_dims=1024,1024,1024"
echo "  - Interval randomised per sample: [6,7,8,9,10,11,12]"
echo ""

# === Optimisation ===
BATCH_SIZE=10
EPOCHS=50
LR=1e-4
WEIGHT_DECAY=1e-5
INTERVALS="6,7,8,9,10,11,12"
USE_AMP="--amp"
GRAD_ACCUM=10
SEED=0

# === Predictor architecture ===
NUM_BLOCKS=3
HIDDEN_DIM=512
SPLIT_DIMS="2048,512,512"
DROPOUT=0.1

# === Loss weights ===
Z_LOSS_WEIGHT=0.1

# === Training schedule ===
PROMPTS_PER_EPOCH=60
EARLY_STOP_PATIENCE=20
EVAL_INTERVAL=1
SAVE_INTERVAL=1

# === Data split ===
TRAIN_PROMPTS="200-400"
VAL_SAMPLE_RANGE="0-200"
NUM_VAL_PROMPTS=25

# === Output directory ===
OUTPUT_DIR="outputs/linca_two_stage_2048512512_b10${NUM_BLOCKS}_h${HIDDEN_DIM}"
EXP_NAME="two_stage_2048512512_b10${NUM_BLOCKS}_h${HIDDEN_DIM}"
WANDB_PROJECT="linca_two_stage_2048512512_3512_b10"

if [ -n "$WANDB_KEY" ]; then
    echo ">>> Logging to wandb"
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
    echo ">>> No wandb API key given; training without it"
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
echo "Training complete."
echo "Checkpoints written to: $OUTPUT_DIR/"
echo "  - best_predictor_stage1.pt"
echo "  - best_predictor_stage2.pt"
echo "========================================"
