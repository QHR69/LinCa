# LinCA 数据生成（cache data）

This directory contains scripts for generating training data for the learned invertible decomposition network.

## Directory Structure

```
linca_data/
├── generate_cache_data.py       # Single-GPU data generation
├── generate_cache_data_ddp.py   # Multi-GPU (DDP) data generation
├── pipeline/
│   ├── pipeline_qwenimage_data.py      # Pipeline for data generation
│   └── transformer_qwenimage_data.py   # Transformer with feature extraction
├── prompts/
│   └── prompts_train.txt        # 400 training prompts
└── data/                        # Generated data (after running scripts)
    ├── cache_data/              # Cached features
    │   ├── prompt_0000/
    │   │   ├── cond/            # noise_pred branch features
    │   │   │   ├── step_00.pt   # [seq_len, 3072]
    │   │   │   └── ...
    │   │   ├── uncond/          # neg_noise_pred branch features
    │   │   │   ├── step_00.pt
    │   │   │   └── ...
    │   │   └── metadata.json
    │   └── ...
    ├── images_first200/         # Images for prompts 0-199 (for PSNR/SSIM)
    └── images_last200/          # Images for prompts 200-399
```

## Usage

### Single-GPU Generation

```bash
cd /root/autodl-tmp/linca_data

# Full generation
python generate_cache_data.py \
    --prompt_file prompts/prompts_train.txt \
    --output_dir data/cache_data \
    --seed 0 \
    --num_steps 50 \
    --true_cfg_scale 4.0
```

### Multi-GPU (DDP) Generation

```bash
cd /root/autodl-tmp/linca_data

# 4 GPUs
torchrun --nproc_per_node=4 generate_cache_data_ddp.py \
    --prompt_file prompts/prompts_train.txt \
    --output_dir data/cache_data \
    --seed 0

# Or specify GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 generate_cache_data_ddp.py \
    --prompt_file prompts/prompts_train.txt
```

## Important: Seed Consistency

The seed logic ensures identical results between single-GPU and multi-GPU:

```python
seed = base_seed + global_prompt_idx
```

- `base_seed`: Command line argument `--seed` (default: 0)
- `global_prompt_idx`: Index of prompt in the file (0-399)

This means:
- Prompt 0 uses seed 0
- Prompt 1 uses seed 1
- Prompt 199 uses seed 199
- etc.

**This matches the seed logic in sample.py**, so generated images can be used for PSNR/SSIM comparison.

## Output Data Format

### Feature Files (`.pt`)

Each feature file contains a tensor of shape `[seq_len, 3072]`:
- `seq_len`: Sequence length (depends on image size, for 1328x1328 it's ~27889)
- `3072`: Feature dimension (after norm_out, before proj_out in transformer)

### Metadata JSON

```json
{
  "prompt_idx": 0,
  "prompt": "A red colored car.",
  "negative_prompt": " ",
  "seed": 0,
  "num_steps": 50,
  "height": 1328,
  "width": 1328,
  "feature_dim": 3072,
  "seq_length": 27889,
  "image_path": "data/images_first200/img_0000.jpg"
}
```

### Index JSON

`data/cache_data/index.json` contains:
- Configuration used for generation
- List of all prompts with their metadata
- Useful for loading data during training

## Image Organization

- **images_first200/**: Prompts 0-199 (matches `test2_drawbench.txt`)
  - Used as ground truth for PSNR/SSIM evaluation
- **images_last200/**: Prompts 200-399
  - Additional training data images

## Estimated Time

- Single A100 80GB: ~3-4 hours for 400 prompts
- 4x A100 80GB: ~1 hour for 400 prompts

## Disk Space

- Features: ~400 prompts × 50 steps × 2 branches × ~200MB = ~8TB (compressed)
- Images: ~400 × 2MB = ~800MB
