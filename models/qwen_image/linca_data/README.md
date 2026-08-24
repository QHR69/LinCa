# Qwen-Image cache-feature extraction

Runs full 50-step Qwen-Image inference over a prompt set and stores the intermediate
DiT features that the LinCa predictor is trained to forecast.

## Layout

```
linca_data/
├── generate_cache_data.py       # single GPU
├── generate_cache_data_ddp.py   # multi-GPU (DDP)
├── pipeline/
│   ├── pipeline_qwenimage_data.py      # pipeline that emits features
│   └── transformer_qwenimage_data.py   # transformer with the extraction hook
├── prompts/
│   ├── prompts_train.txt        # 400 training prompts
│   ├── DrawBench200.txt         # evaluation prompts
│   └── prompts_for_edit.txt     # edit instructions, for the Qwen-Image-Edit data
└── data/                        # created by the scripts
    ├── cache_data/
    │   ├── prompt_0000/
    │   │   ├── cond/            # noise_pred branch, one file per step
    │   │   │   ├── step_00.pt   # [seq_len, 3072]
    │   │   │   └── ...
    │   │   ├── uncond/          # neg_noise_pred branch
    │   │   └── metadata.json
    │   └── index.json
    ├── images_first200/         # images for prompts 0-199, the PSNR/SSIM reference
    └── images_last200/          # images for prompts 200-399
```

## Usage

```bash
cd models/qwen_image/linca_data

python generate_cache_data.py \
    --prompt_file prompts/prompts_train.txt \
    --output_dir data/cache_data \
    --seed 0 \
    --num_steps 50 \
    --true_cfg_scale 4.0
```

Across 4 GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 generate_cache_data_ddp.py \
    --prompt_file prompts/prompts_train.txt \
    --output_dir data/cache_data \
    --seed 0
```

## Seeding

Each prompt is seeded independently, so single-GPU and multi-GPU runs produce
identical output regardless of how prompts are sharded:

```python
seed = base_seed + global_prompt_idx
```

`base_seed` comes from `--seed` (default 0) and `global_prompt_idx` is the prompt's
line number in the file, so prompt 0 uses seed 0, prompt 199 uses seed 199, and so on.

The samplers in [`../qwen_image_linca_two_stage`](../qwen_image_linca_two_stage) seed the
same way, which is what makes the images written here usable as the PSNR/SSIM reference
for accelerated sampling.

## Output format

Each `.pt` file holds one tensor of shape `[seq_len, 3072]`, taken after `norm_out`
and before `proj_out`. `seq_len` follows the image size — about 27889 at 1328×1328 —
and 3072 is the transformer's hidden dimension.

`metadata.json`, per prompt:

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

`data/cache_data/index.json` records the generation config and every prompt's
metadata, and is what the training dataloader reads.

`images_first200/` covers prompts 0-199, matching `test2_drawbench.txt`, and serves as
the ground truth for PSNR/SSIM. `images_last200/` covers prompts 200-399.

## Cost

Roughly 3-4 hours on one A100 80GB for 400 prompts, or about an hour on four.

Plan for the disk: a single `[27889, 3072]` bf16 feature file is around 170 MB, and a
full run writes 400 prompts × 50 steps × 2 branches, so the total reaches several
terabytes. Generate a subset first (200 prompts is enough to train a usable predictor)
and check the footprint before committing to the full set. Images add about 800 MB.
