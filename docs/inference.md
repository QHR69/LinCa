# Inference

LinCa predictors are small. You still need the official backbone weights and the corresponding Python environment.

Released files on [QHRQQQ/LinCa](https://huggingface.co/QHRQQQ/LinCa):

| File | Model |
|---|---|
| `qwen-image_checkpoint.pt` | Qwen-Image (two-stage) |
| `qwen-image-edit_checkpoint.pt` | Qwen-Image-Edit (two-stage) |
| `flux/best_predictor_stage{0,1,2}.pt` | FLUX stages 0-2 |
| `flux/best_predictor_stage{0,1,2}_config.json` | FLUX per-stage architecture |

Download the FLUX `_config.json` files alongside the `.pt` files and keep them in the
same directory. `from_pretrained` reads the architecture from them, and without them it
falls back to different defaults, so the state dict will not load.

Use the **script defaults** shipped with this repo when loading the released checkpoints. Those defaults are what the files were trained/exported with.

## Qwen-Image

```bash
cd models/qwen_image/qwen_image_linca_two_stage
python sample_learned.py \
  --checkpoint /path/to/qwen-image_checkpoint.pt \
  --model_path Qwen/Qwen-Image \
  --prompt_file prompts/DrawBench200.txt \
  --output_dir samples/qwen_image
```

`transformers>=4.51` is required for Qwen2.5-VL.

## Qwen-Image-Edit

Single image:

```bash
cd models/qwen_image_edit/qwen_image_edit_linca_two_stage
python sample_edit_single.py \
  --checkpoint /path/to/qwen-image-edit_checkpoint.pt \
  --image path/to/input.png \
  --prompt "your edit instruction"
```

GEdit-Bench-style batch sampling is in `sample_learned_ddp.py`.

## FLUX.1-dev

This tree is **inference-only** with the released 3-stage predictors.

```bash
export FLUX_CHECKPOINT_DIR=/path/to/FLUX.1-dev   # flux1-dev.safetensors + ae.safetensors
cd models/flux
bash run_sample.sh /path/to/flux-predictor-dir 17,17,16
```

The directory must contain `best_predictor_stage0.pt`, `best_predictor_stage1.pt`, and `best_predictor_stage2.pt`. Stage splits `17,17,16` match a 50-step schedule.

Optional environment variables used by `run_sample.sh`:

- `FLUX_CHECKPOINT_DIR` — official FLUX.1-dev folder
- `HF_HOME` — Hugging Face cache
- `EVAL_SCRIPT`, `BASELINE_DIR` — evaluation only
- `IMAGEREWARD_DIR`, `EVAL_ROOT` — ImageReward / eval helpers if you run scoring

## HunyuanVideo / Wan2.1

Each folder has a `run_sample.sh` wrapping its sampler — `sample_video.py` for
HunyuanVideo, `sample_wan.py` for Wan2.1. We do not publish trained predictors for
these two, so train one first and point the script at it. Note that Wan2.1 needs
*both* predictor stages, while HunyuanVideo takes a single checkpoint.

## Training notes

Qwen-Image, Qwen-Image-Edit, HunyuanVideo, and Wan2.1 include data-generation and training scripts. See the README in each model folder.

FLUX training code is not part of this release.
