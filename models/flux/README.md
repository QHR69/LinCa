# LinCa on FLUX.1-dev

Cache-accelerated FLUX.1-dev inference. This subtree is **inference only** — it
ships the released three-stage predictors and the sampler that uses them. The
predictor training code lives with the other backbones (see
[`models/qwen_image`](../qwen_image) for the reference training recipe).

## Layout

```
models/flux/
├── run_sample.sh              # entry point: multi-stage cache-accelerated sampling
├── prompts/DrawBench200.txt   # the 200-prompt evaluation set used in the paper
├── requirements.txt
└── src/
    ├── sample.py              # single-process sampler
    ├── sample_ddp.py          # multi-GPU sampler
    └── flux/                  # FLUX model code, vendored from Black Forest Labs
        ├── model.py, sampling.py, util.py, math_utils.py
        ├── modules/
        │   ├── invertible_net.py         # LinCa invertible predictor
        │   └── cache_functions/
        │       ├── cache_utils.py        # frequency-decomposition cache baseline
        │       ├── cache_utils_learned.py# LinCa learned decomposition
        │       ├── cache_init.py, cal_type.py
        └── ...
```

## Setup

```bash
pip install -r requirements.txt
```

The tree is used via `PYTHONPATH`, which `run_sample.sh` sets for you. Nothing
needs to be pip-installed from this directory.

### Backbone weights

Get FLUX.1-dev from the official source and point the sampler at it. `src/flux/util.py`
reads `FLUX_MODEL` and `FLUX_AE`; `FLUX_CHECKPOINT_DIR` is a shortcut that sets both:

```bash
huggingface-cli download black-forest-labs/FLUX.1-dev \
  flux1-dev.safetensors ae.safetensors --local-dir ./checkpoints/FLUX.1-dev

export FLUX_CHECKPOINT_DIR=./checkpoints/FLUX.1-dev
```

The text encoders (T5-XXL and CLIP) are pulled from the Hub on first run.

### LinCa predictors

Download the whole `flux/` folder. The `*_config.json` files are **required** —
`LearnedDecompositionPredictor.from_pretrained` reads the architecture from them,
and without them it falls back to a different default and the load fails:

```bash
huggingface-cli download QHRQQQ/LinCa --include 'flux/*' --local-dir ./checkpoints
```

That yields `./checkpoints/flux/best_predictor_stage{0,1,2}.pt` and the matching
`best_predictor_stage{0,1,2}_config.json`.

## Sampling

```bash
cd models/flux
bash run_sample.sh ../../checkpoints/flux              # interval 6, splits 17,17,16
bash run_sample.sh ../../checkpoints/flux 17,17,16 6 8 # sweep two intervals
```

Arguments are `<predictor_dir> [stage_splits] [interval ...]`.

`stage_splits` assigns each predictor a contiguous run of denoising steps and must
sum to `NUM_STEPS`. The released checkpoints are three stages over 50 steps
(`17,17,16`), i.e. steps 0–16, 17–33, 34–49. The script refuses to start if the
splits do not add up, or if a stage checkpoint or its config file is missing.

Everything else is overridable by environment variable, and the defaults below are
the ones the released checkpoints were trained for:

| Variable | Default | Meaning |
|---|---|---|
| `PROMPT_FILE` | `prompts/DrawBench200.txt` | prompt list, one per line |
| `NUM_STEPS` | `50` | denoising steps; must equal the sum of `stage_splits` |
| `WIDTH`, `HEIGHT` | `1024` | output resolution |
| `GUIDANCE` | `3.5` | FLUX guidance scale |
| `SEED` | `21` | base seed |
| `FORECAST_METHOD` | `hermite` | `hermite` or `taylor` |
| `FIRST_ENHANCE` | `3` | leading steps always computed in full |
| `MAX_ORDER`, `MIN_ORDER` | `2`, `0` | forecast order bounds |
| `BASE_OUTPUT_DIR` | `samples/<dir>_infer` | output root |

Images land in `<BASE_OUTPUT_DIR>/interval_<N>/`.

For multi-GPU sampling call `src/sample_ddp.py` directly under `torchrun`; it takes
the same flags as `src/sample.py` apart from the multi-stage predictor options.

## Scoring

Scoring is off by default because it needs an ImageReward install and a set of
full-step baseline images to measure PSNR / SSIM / LPIPS against. The helper
`eval_images.py` is not part of this release; supply your own and enable it with:

```bash
RUN_EVAL=1 \
EVAL_SCRIPT=/path/to/eval_images.py \
BASELINE_DIR=/path/to/full_step_baseline \
IMAGEREWARD_DIR=/path/to/ImageReward \
  bash run_sample.sh ../../checkpoints/flux 17,17,16 6
```

It writes `evaluation_results.txt` next to the generated images.

## Licensing

`src/flux/` is FLUX inference code from Black Forest Labs, kept under its original
license. LinCa's own additions are `modules/invertible_net.py` and
`modules/cache_functions/cache_utils_learned.py`, plus the cache hooks in
`sampling.py`. See the repository [NOTICE](../../NOTICE).
