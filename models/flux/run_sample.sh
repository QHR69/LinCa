#!/bin/bash
# Cache-accelerated FLUX.1-dev sampling with multi-stage LinCa predictors.
#
# Usage:
#   bash run_sample.sh <predictor_dir> [stage_splits] [interval ...]
#
#   bash run_sample.sh ../../checkpoints/flux                 # interval 6, splits 17,17,16
#   bash run_sample.sh ../../checkpoints/flux 17,17,16 6 8    # two intervals
#   bash run_sample.sh ../../checkpoints/flux 25,25 6         # two-stage predictor
#
# <predictor_dir> holds best_predictor_stage<i>.pt plus the matching
# best_predictor_stage<i>_config.json for every stage. stage_splits must sum to
# NUM_STEPS and have one entry per stage.
#
# The backbone is resolved by src/flux/util.py, which honours FLUX_MODEL and
# FLUX_AE; FLUX_CHECKPOINT_DIR below is just a convenient way to set both.
#
# Scoring is optional and off by default because it needs an ImageReward install
# and a baseline image set. Enable with:
#   RUN_EVAL=1 EVAL_SCRIPT=/path/to/eval_images.py BASELINE_DIR=/path/to/baseline \
#       bash run_sample.sh ...

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH="${SCRIPT_DIR}/src:${PYTHONPATH}"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"

# Point at local FLUX weights when present, so nothing is re-downloaded.
FLUX_CHECKPOINT_DIR="${FLUX_CHECKPOINT_DIR:-./checkpoints/FLUX.1-dev}"
if [ -d "$FLUX_CHECKPOINT_DIR" ]; then
    export FLUX_MODEL="${FLUX_MODEL:-${FLUX_CHECKPOINT_DIR}/flux1-dev.safetensors}"
    export FLUX_AE="${FLUX_AE:-${FLUX_CHECKPOINT_DIR}/ae.safetensors}"
fi

PREDICTOR_DIR="${1:-}"
STAGE_SPLITS="${2:-17,17,16}"
if [ $# -gt 2 ]; then
    shift 2
    INTERVALS=("$@")
else
    IFS=',' read -ra INTERVALS <<< "${INTERVALS:-6}"
fi

if [ -z "$PREDICTOR_DIR" ]; then
    echo "Usage: bash run_sample.sh <predictor_dir> [stage_splits] [interval ...]"
    echo "Example: bash run_sample.sh ../../checkpoints/flux 17,17,16 6"
    echo ""
    echo "Fetch the released predictors with:"
    echo "  huggingface-cli download QHRQQQ/LinCa --include 'flux/*' --local-dir ./checkpoints"
    exit 1
fi

# === Sampling settings (these match the released checkpoints) ===
PROMPT_FILE="${PROMPT_FILE:-prompts/DrawBench200.txt}"
MODEL_NAME="${MODEL_NAME:-flux-dev}"
WIDTH="${WIDTH:-1024}"
HEIGHT="${HEIGHT:-1024}"
NUM_STEPS="${NUM_STEPS:-50}"
GUIDANCE="${GUIDANCE:-3.5}"
BATCH_SIZE="${BATCH_SIZE:-1}"
SEED="${SEED:-21}"

# Cache settings, matched to training.
MAX_ORDER="${MAX_ORDER:-2}"
MIN_ORDER="${MIN_ORDER:-0}"
FIRST_ENHANCE="${FIRST_ENHANCE:-3}"
FORECAST_METHOD="${FORECAST_METHOD:-hermite}"

if [ ! -f "$PROMPT_FILE" ]; then
    echo "Prompt file not found: $PROMPT_FILE"
    exit 1
fi

# === Resolve one checkpoint per stage, and sanity-check the split ===
IFS=',' read -ra SPLITS_ARR <<< "$STAGE_SPLITS"
NUM_STAGES=${#SPLITS_ARR[@]}

split_total=0
for size in "${SPLITS_ARR[@]}"; do
    split_total=$((split_total + size))
done
if [ "$split_total" -ne "$NUM_STEPS" ]; then
    echo "stage_splits '$STAGE_SPLITS' sums to $split_total but NUM_STEPS is $NUM_STEPS."
    echo "They must match: one predictor covers one contiguous run of steps."
    exit 1
fi

echo "========================================"
echo "FLUX.1-dev LinCa sampling"
echo "========================================"
echo "Stages: ${NUM_STAGES} (splits=${STAGE_SPLITS}, ${NUM_STEPS} steps)"

CHECKPOINT_ARGS=""
start=0
for i in $(seq 0 $((NUM_STAGES - 1))); do
    CKPT="${PREDICTOR_DIR}/best_predictor_stage${i}.pt"
    if [ ! -f "$CKPT" ]; then
        echo "Stage ${i} checkpoint not found: $CKPT"
        exit 1
    fi
    # from_pretrained reads the architecture from this file; without it the
    # loader falls back to a different default and the strict load fails.
    if [ ! -f "${PREDICTOR_DIR}/best_predictor_stage${i}_config.json" ]; then
        echo "Missing best_predictor_stage${i}_config.json next to $CKPT"
        echo "Re-download the predictor directory so the config files come with it."
        exit 1
    fi
    size=${SPLITS_ARR[$i]}
    end=$((start + size - 1))
    echo "  stage ${i}: steps ${start}-${end}  <- $(basename "$CKPT")"
    CHECKPOINT_ARGS="${CHECKPOINT_ARGS} ${CKPT}"
    start=$((end + 1))
done

EXP_NAME=$(basename "$PREDICTOR_DIR")
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-samples/${EXP_NAME}_infer}"
mkdir -p "$BASE_OUTPUT_DIR"

echo "Prompts:   $PROMPT_FILE"
echo "Intervals: ${INTERVALS[*]}"
echo "Output:    $BASE_OUTPUT_DIR"
echo "========================================"
echo ""

for INTERVAL in "${INTERVALS[@]}"; do
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/interval_${INTERVAL}"
    mkdir -p "$OUTPUT_DIR"

    echo "---------- interval $INTERVAL ----------"
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
        --forecast_steps $INTERVAL \
        --learned_checkpoints $CHECKPOINT_ARGS \
        --stage_splits "$STAGE_SPLITS" \
        --add_sampling_metadata

    echo ">>> interval $INTERVAL done: $OUTPUT_DIR"
    echo ""
done

echo "========================================"
echo "Sampling finished. Images under: $BASE_OUTPUT_DIR"
echo "========================================"

# ========== Optional scoring ==========
if [ "${RUN_EVAL:-0}" != "1" ]; then
    echo ""
    echo "Scoring skipped. To score ImageReward / CLIP / PSNR / SSIM / LPIPS against a"
    echo "baseline, install ImageReward and re-run with:"
    echo "  RUN_EVAL=1 EVAL_SCRIPT=/path/to/eval_images.py BASELINE_DIR=/path/to/baseline \\"
    echo "      bash run_sample.sh $PREDICTOR_DIR $STAGE_SPLITS ${INTERVALS[*]}"
    exit 0
fi

EVAL_SCRIPT="${EVAL_SCRIPT:-}"
BASELINE_DIR="${BASELINE_DIR:-}"
if [ -z "$EVAL_SCRIPT" ] || [ ! -f "$EVAL_SCRIPT" ]; then
    echo "RUN_EVAL=1 but EVAL_SCRIPT does not point at a readable eval_images.py."
    echo "The scoring helper is not part of this release; supply your own."
    exit 1
fi
if [ -z "$BASELINE_DIR" ] || [ ! -d "$BASELINE_DIR" ]; then
    echo "RUN_EVAL=1 but BASELINE_DIR is not a directory."
    echo "PSNR / SSIM / LPIPS are measured against full-step baseline images."
    exit 1
fi

echo ""
echo "========================================"
echo "Scoring $BASE_OUTPUT_DIR against $BASELINE_DIR"
echo "========================================"

EVAL_DIR="$(cd "$(dirname "$EVAL_SCRIPT")" && pwd)"
INTERVAL_LIST=$(IFS=,; echo "${INTERVALS[*]}")

PYTHONPATH="${EVAL_DIR}:${IMAGEREWARD_DIR:-}:${PYTHONPATH}" python - <<PYEOF
import os
from eval_images import evaluate_images

intervals = [int(x) for x in "${INTERVAL_LIST}".split(",") if x]
base_output_dir = "${BASE_OUTPUT_DIR}"
baseline_dir = "${BASELINE_DIR}"
prompts_file = "${PROMPT_FILE}"

imagereward_model = None
rows = []
for interval in intervals:
    image_dir = os.path.join(base_output_dir, f"interval_{interval}")
    print(f"\n========== interval {interval} :: {image_dir} ==========")
    result = evaluate_images(
        image_dir=image_dir,
        prompts_file=prompts_file,
        baseline_dir=baseline_dir,
        imagereward_model=imagereward_model,
        skip_clip=False,
        skip_lpips=False,
        verbose=True,
    )
    if imagereward_model is None:
        imagereward_model = result.get("imagereward_model")
    metrics = {k: result[k] for k in
               ("imagereward", "clip_score", "psnr", "ssim", "lpips", "num_images")
               if k in result}
    rows.append((interval, metrics))

out = os.path.join(base_output_dir, "evaluation_results.txt")
with open(out, "w", encoding="utf-8") as f:
    f.write(f"baseline: {baseline_dir}\nprompts: {prompts_file}\n\n")
    for interval, metrics in rows:
        f.write(f"--- interval {interval} ---\n")
        for k, v in metrics.items():
            f.write(f"  {k}: {v}\n")
        f.write("\n")
print(f"\nWrote {out}")
PYEOF

echo ""
echo "Scoring complete: $BASE_OUTPUT_DIR/evaluation_results.txt"
