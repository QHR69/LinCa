#!/bin/bash
# Train the LinCa predictor for HunyuanVideo.
#
# Reads the cache features produced by generate_features.sh and fits a single
# invertible-decomposition predictor covering all 50 sampling steps.

set -e

DATA_DIR=${DATA_DIR:-"./cache_data/hunyuan"}
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: feature directory not found: $DATA_DIR"
    echo "Run generate_features.sh first, or point DATA_DIR at an existing one."
    exit 1
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH="${SCRIPT_DIR}/src:${PYTHONPATH}"

# === Optimisation ===
BATCH_SIZE=1
EPOCHS=100
LR=2e-5
WEIGHT_DECAY=1e-5
GRAD_ACCUM=64
USE_AMP="--amp"
SEED=42

# === Predictor architecture (override via environment for sweeps) ===
NUM_BLOCKS=${NUM_BLOCKS:-2}
HIDDEN_DIM=${HIDDEN_DIM:-64}
SPLIT_DIMS=${SPLIT_DIMS:-"2304,384,384"}   # z0,z1,z2; must sum to --dim
DROPOUT=${DROPOUT:-0.0}
Z_LOSS_WEIGHT=0.1

# Cache intervals to train against. --random_interval draws one per sample so a
# single predictor generalises across all of them.
INTERVALS=${INTERVALS:-"3,4,5,6,7,8,9,10"}

# === Data split (VBench, 946 prompts) ===
TRAIN_PROMPTS="0-946"
NUM_VAL_PROMPTS=20       # taken at regular spacing; the rest are training prompts
PROMPTS_PER_EPOCH=20     # prompts sampled per epoch from the training pool

EARLY_STOP_PATIENCE=20
SAVE_INTERVAL=1
# Mid-training sample previews are not supported for this backbone; keep at 0.
EVAL_INTERVAL=${EVAL_INTERVAL:-0}

# === Output ===
OUTPUT_DIR=${OUTPUT_DIR:-"outputs"}
EXP_NAME="hunyuan_linca_b_${NUM_BLOCKS}_h_${HIDDEN_DIM}_splits_${SPLIT_DIMS//,/x}_intervals_${INTERVALS//,/x}_epochs_${EPOCHS}_dropout_${DROPOUT}_seed_${SEED}"
WANDB_PROJECT="linca-hunyuanvideo"
# Uncomment to train without wandb:
# NO_WANDB="--no_wandb"

echo "========================================"
echo "LinCa predictor training -- HunyuanVideo"
echo "========================================"
echo "  features:   $DATA_DIR"
echo "  blocks:     $NUM_BLOCKS"
echo "  hidden dim: $HIDDEN_DIM"
echo "  splits:     $SPLIT_DIMS"
echo "  intervals:  $INTERVALS (randomised per sample)"
echo "  output:     $OUTPUT_DIR/$EXP_NAME"
echo ""

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
echo "Training complete."
echo "Best predictor: $OUTPUT_DIR/$EXP_NAME/best_predictor.pt"
echo "TensorBoard:    tensorboard --logdir=$OUTPUT_DIR"
echo "========================================"
