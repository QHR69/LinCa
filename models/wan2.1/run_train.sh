#!/bin/bash
# Stage 2 of the Wan2.1 pipeline: train the two-stage invertible predictor on the
# features dumped by generate_features.sh.
#
# Dimensions below target Wan2.1 T2V-1.3B (hidden size 1536). For T2V-14B set
# DIM=5120 and pick SPLIT_DIMS that sum to it.
#
# Usage:
#   bash run_train.sh
#   NUM_BLOCKS=2 HIDDEN_DIM=128 bash run_train.sh     # used by run_sweep.sh

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH="${SCRIPT_DIR}/src:${PYTHONPATH}"

DATA_DIR="${DATA_DIR:-./cache_data/wan}"
if [ ! -d "$DATA_DIR" ]; then
    echo "Cache feature directory not found: $DATA_DIR"
    echo "Run 'bash generate_features.sh <wan_ckpt_dir>' first."
    exit 1
fi

# === Model shape (Wan2.1 T2V-1.3B) ===
DIM=${DIM:-1536}
SPLIT_DIMS=${SPLIT_DIMS:-"1152,192,192"}
NUM_BLOCKS=${NUM_BLOCKS:-2}
HIDDEN_DIM=${HIDDEN_DIM:-64}
DROPOUT=${DROPOUT:-0.0}

# === Optimisation ===
BATCH_SIZE=${BATCH_SIZE:-1}
EPOCHS=${EPOCHS:-100}
LR=${LR:-2e-5}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-5}
GRAD_ACCUM=${GRAD_ACCUM:-64}
INTERVALS=${INTERVALS:-"3,4,5,6,7,8,9,10"}
Z_LOSS_WEIGHT=${Z_LOSS_WEIGHT:-0.1}
SEED=${SEED:-42}

# === Schedule ===
PROMPTS_PER_EPOCH=${PROMPTS_PER_EPOCH:-20}
EARLY_STOP_PATIENCE=${EARLY_STOP_PATIENCE:-20}
SAVE_INTERVAL=${SAVE_INTERVAL:-1}
# Kept at 0: mid-training sample generation is not wired up for Wan.
EVAL_INTERVAL=${EVAL_INTERVAL:-0}

# === Prompt split (VBench 946 prompts) ===
TRAIN_PROMPTS=${TRAIN_PROMPTS:-"0-946"}
NUM_VAL_PROMPTS=${NUM_VAL_PROMPTS:-20}

OUTPUT_DIR=${OUTPUT_DIR:-outputs}
EXP_NAME=${EXP_NAME:-"wan_linca_b${NUM_BLOCKS}_h${HIDDEN_DIM}_splits_${SPLIT_DIMS//,/x}_intervals_${INTERVALS//,/x}_epochs_${EPOCHS}_seed_${SEED}"}
WANDB_PROJECT=${WANDB_PROJECT:-"linca-wan2.1"}

echo "========================================"
echo "Wan2.1 LinCa predictor training"
echo "========================================"
echo "  Data:       $DATA_DIR"
echo "  Dim:        $DIM  (splits $SPLIT_DIMS)"
echo "  Blocks:     $NUM_BLOCKS  hidden $HIDDEN_DIM  dropout $DROPOUT"
echo "  Intervals:  $INTERVALS (sampled per step)"
echo "  Experiment: $EXP_NAME"
echo ""
echo "Pass --no_wandb via NO_WANDB=--no_wandb to train without Weights & Biases."
echo ""

python src/flux/train_encoder_decoder.py \
    --data_dir "$DATA_DIR" \
    --train_prompts $TRAIN_PROMPTS \
    --num_val_prompts $NUM_VAL_PROMPTS \
    --prompts_per_epoch $PROMPTS_PER_EPOCH \
    --shuffle_prompts \
    --random_interval \
    --early_stop_patience $EARLY_STOP_PATIENCE \
    --dim $DIM \
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
    --output_dir "$OUTPUT_DIR" \
    --exp_name "$EXP_NAME" \
    --seed $SEED \
    --wandb_project "$WANDB_PROJECT" \
    ${NO_WANDB:-} \
    --amp

echo ""
echo "========================================"
echo "Training finished."
echo "  Checkpoints: $OUTPUT_DIR/$EXP_NAME/best_predictor_stage{1,2}.pt"
echo "  TensorBoard: tensorboard --logdir=$OUTPUT_DIR"
echo "  Next step:   bash run_sample.sh <wan_ckpt_dir> $OUTPUT_DIR/$EXP_NAME"
echo "========================================"
