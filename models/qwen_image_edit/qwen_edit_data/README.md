# Qwen-Image-Edit cache-feature extraction

Produces the cond/uncond feature cache for 202 training samples, which
`qwen_image_edit_linca_two_stage/train.py` then learns to predict.

## What it does

- **Source**: GEdit-Bench, every 6th sample (indices 0, 6, 12, ..., 1206) — 202 in total
- **Resolution**: everything resized to 1024×1024
- **Seed**: `base_seed + original_idx`, matching `sample_edit_single.py` so the
  generated edits line up with accelerated ones for quality comparison
- **Cached tensor**: the 3072-dim activation after `norm_out` and before `proj_out`,
  latent part only (`[:, :4096]`)
- **Sampling config**: `num_steps=50`, `negative_prompt=" "`, `true_cfg_scale=4.0`,
  `guidance_scale=1.0`, `max_sequence_length=512`

GEdit-Bench is not redistributed here. Fetch it separately and point
`--dataset_path` at it (default `./data/gedit_bench`).

## Usage

```bash
cd models/qwen_image_edit/qwen_edit_data

# one sample, to confirm the setup works
python generate_cache_data_edit.py --limit 1

# all 202 across 4 GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
  generate_cache_data_edit.py --ddp

# same thing, with GPU count detected for you
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_generate_cache_ddp.sh
```

## Output

- Features: `<cache_output_dir>/sample_XXXX/{cond,uncond}/step_XX.pt`, default
  `--cache_output_dir ./cache_data/qwen_edit`
- Edited images: `sample_XXXX.png` under `--image_output_dir`
- `index.json`, which the training dataloader reads

Each cond/uncond tensor has shape `(4096, 3072)`. A sample costs roughly 2.4 GB across
50 steps and both branches, so all 202 need on the order of 470 GB. Extract a subset
first if disk is tight.
