# LinCa on Qwen-Image-Edit

Cache-accelerated instruction-based image editing with a two-stage learned
invertible predictor. Evaluated on GEdit-Bench in the paper.

## Layout

```
models/qwen_image_edit/
├── qwen_edit_data/                     # cache-feature extraction
│   ├── generate_cache_data_edit.py
│   ├── run_generate_cache_ddp.sh
│   └── pipeline/
└── qwen_image_edit_linca_two_stage/    # training and editing
    ├── train.py, train_ddp.py
    ├── run_train.sh, run_train_ddp.sh, run_ddp_mid.sh
    ├── sample_edit_single.py           # one image or a CSV batch
    ├── sample_learned_ddp.py           # full GEdit-Bench sweep
    ├── sample_baseline_ddp.py          # uncached baseline
    ├── export_inference_checkpoint.py
    ├── invertible_net.py               # the LinCa predictor
    ├── dataset.py, data_splits.py
    ├── cache_functions/
    └── demo_pairs/                     # 10 sample images with instructions
```

## Setup

```bash
pip install -r requirements.txt
```

`transformers>=4.51` is required for the Qwen2.5-VL encoder that Qwen-Image-Edit uses.

The backbone (`Qwen/Qwen-Image-Edit`) is fetched from the Hub on first use; pass
`--model_path` for a local copy.

## Editing with the released predictor

```bash
huggingface-cli download QHRQQQ/LinCa qwen-image-edit_checkpoint.pt --local-dir ./checkpoints

cd models/qwen_image_edit/qwen_image_edit_linca_two_stage

# one image
python sample_edit_single.py \
  --checkpoint ../../../checkpoints/qwen-image-edit_checkpoint.pt \
  --image demo_pairs/images/img_1.png \
  --prompt "change the background to a city street" \
  --output output.png

# the bundled 10-pair demo
python sample_edit_single.py \
  --checkpoint ../../../checkpoints/qwen-image-edit_checkpoint.pt \
  --pairs_file demo_pairs/pairs.csv \
  --output_dir samples/demo_edit

# same, but stop after 2 pairs
python sample_edit_single.py \
  --checkpoint ../../../checkpoints/qwen-image-edit_checkpoint.pt \
  --pairs_file demo_pairs/pairs.csv \
  --output_dir samples/demo_edit --limit 2
```

`--checkpoint` is required. The released file embeds its architecture config, so
**keep the code defaults** (`--interval 7`, `--z2_forecast_method lagrange`,
`--num_steps 50`) unless you retrain.

Inputs are resized to 1024×1024 for the model and results are resized back to the
original dimensions before saving.

Instructions may be written in English or Chinese — the paper reports GEdit-Bench in
both languages. `demo_pairs/pairs.csv` ships the English wording.

### Full GEdit-Bench sweep

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29601 \
  sample_learned_ddp.py \
  --checkpoint ../../../checkpoints/qwen-image-edit_checkpoint.pt \
  --dataset_path ./data/gedit_bench \
  --output_dir samples/gedit_full \
  --seed 0 --interval 7
```

Writes `<output_dir>/fullset/<task_type>/<lang>/<key>.png`. The script skips outputs
that already exist, so an interrupted run can simply be restarted.

GEdit-Bench is not redistributed here; fetch it from its own source and point
`--dataset_path` at it.

## Training your own predictor

### 1. Extract cache features

Takes every 6th GEdit-Bench sample (202 in total) and stores the cond/uncond
features from a full 50-step edit.

```bash
cd models/qwen_image_edit/qwen_edit_data

python generate_cache_data_edit.py --limit 1          # single sample, to verify the setup

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
  generate_cache_data_edit.py --ddp                   # all 202
# or
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_generate_cache_ddp.sh
```

Output goes to `--cache_output_dir` (default `./cache_data/qwen_edit`) as
`sample_XXXX/` directories plus an `index.json`.

### 2. Train

```bash
cd models/qwen_image_edit/qwen_image_edit_linca_two_stage

# run_train_ddp.sh expects CACHE_DATA_DIR to hold step 1's output;
# it defaults to ./cache_data/qwen_edit
NPROC=4 MASTER_PORT=29600 bash run_train_ddp.sh [WANDB_KEY]
```

Checkpoints land under `outputs/`.

### 3. Shrink the checkpoint for inference

A training checkpoint is roughly 1.8 GB because of optimizer state; the exported
inference copy is about 600 MB and generates identical results:

```bash
python export_inference_checkpoint.py \
  --input checkpoints/checkpoint_full.pt \
  --output checkpoints/checkpoint.pt
```

## Notes

- `--z2_forecast_method` selects the second-order forecast: `lagrange` (default) or
  `hermite`, which matches the FreqCa baseline.
- `sample_baseline_ddp.py` generates the uncached reference images that quality
  metrics are measured against.
