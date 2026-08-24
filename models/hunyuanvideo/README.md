# LinCa on HunyuanVideo

Cache-accelerated HunyuanVideo text-to-video generation. This folder also holds the
FLUX-derived predictor trainer that the FLUX experiments share, because the trainer
reads cached features off disk and is backbone-agnostic.

We do not publish a trained predictor for HunyuanVideo — train your own with the
recipe below.

## Method

`src/flux/modules/invertible_net.py` holds the method. `InvertibleDecompositionNet`
maps a step's 3072-dim DiT feature through an invertible transform into three
partitions `z0 / z1 / z2`. At inference each partition is extrapolated at a different
order — zeroth for `z0`, first for `z1`, second for `z2` (`FixedPredictionStrategy`) —
and the inverse transform (`compose`) rebuilds the full feature. Most steps therefore
become cache steps instead of full DiT evaluations.

An invertible block pairs `LightweightInvertible1x1Conv` (orthogonally initialised
1×1 convolution, Glow-style channel mixing) with `RevNetResidualBlock` (exactly
invertible, F/G subnetworks) into a `HybridInvertibleBlock`; `num_blocks` of those are
stacked. `LearnedDecompositionPredictor` wraps the network plus the prediction strategy
and exposes `decompose / compose / predict_from_decomposed / save_pretrained /
from_pretrained`. `save_pretrained` precomputes the 1×1 convolution inverses so
inference does not have to.

Inference hooks live in `src/flux/modules/cache_functions/cache_utils_learned.py`.
HunyuanVideo uses a **single predictor** (one checkpoint) covering all steps; the
two-stage variant is in [`models/wan2.1`](../wan2.1) and the Qwen trees.

Measured cost, from [`flops_analysis.md`](flops_analysis.md) — H800 80GB, 480×640,
65 frames, 50 steps: a full step is 595.46 TFLOPs, a cache step is 0.145 TFLOPs.

## Layout

```
models/hunyuanvideo/
├── generate_features.sh          # extract HunyuanVideo cache features
├── generate_features_flux.sh     # extract FLUX cache features (shared trainer)
├── generate_cache_features.py    # HunyuanVideo feature extractor
├── run_train.sh                  # train the predictor
├── run_sweep.sh                  # multi-GPU hyperparameter grid
├── run_sample.sh                 # accelerated video generation
├── sample_video.py               # HunyuanVideo sampler
├── calc_flops.py                 # HunyuanVideo FLOPs table
├── calc_flops_flux.py            # FLUX FLOPs table
├── flops_analysis.md             # measured FLOPs and speedups
├── visualize_intro_figures*.py   # paper figure scripts
├── save_init_model.py
├── requirements.txt, environment.yml
├── prompts/DrawBench200.txt
└── src/
    ├── flux/                     # method code + FLUX-derived trainer
    │   ├── modules/
    │   │   ├── invertible_net.py            # the invertible predictor
    │   │   └── cache_functions/
    │   │       ├── cache_utils_learned.py   # learned-cache hooks (single predictor)
    │   │       ├── cache_utils.py           # frequency-decomposition baseline
    │   │       ├── cache_init.py            # cache dict setup
    │   │       └── cal_type.py              # decides full vs cache for a step
    │   ├── train_encoder_decoder.py         # training entry point
    │   ├── dataset.py                       # builds prediction samples per interval
    │   └── generate_cache_features.py       # FLUX feature extractor
    └── hyvideo/                  # HunyuanVideo backbone
        ├── inference.py          # HunyuanVideoSampler
        ├── config.py             # CLI args, including LinCa's cache flags
        ├── modules/models.py     # DiT
        ├── vae/                  # 3D causal VAE
        ├── diffusion/            # pipelines and schedulers
        └── text_encoder/
```

## Setup

```bash
pip install -r requirements.txt
```

`environment.yml` is an exact conda export (with build hashes) if you want to
reproduce the original environment: `conda env create -f environment.yml`.

Nothing needs pip-installing from this directory. The scripts export
`PYTHONPATH="${SCRIPT_DIR}/src"` so that `flux` and `hyvideo` import.

## What you need to supply

None of this is redistributed here:

1. **HunyuanVideo weights** at `./checkpoints/hunyuanvideo` (override with `MODEL_BASE`),
   with the DiT at
   `${MODEL_BASE}/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt`.
   Get them from Tencent-Hunyuan/HunyuanVideo.
2. **VBench prompts** at `./data/VBench_full_info.json` (946 prompts), from the VBench
   repository. Used for both feature extraction and the evaluation sweep.
3. **Cache features** at `./cache_data/hunyuan`, produced by step 1 below.

## Pipeline

### 1. Extract cache features

```bash
bash generate_features.sh              # all prompts
bash generate_features.sh 0 200        # shard: prompts 0-200
```

Each prompt is run for a full 50 steps with `cache_init(..., first_enhance=num_steps)`
so that no step is skipped, and every step's DiT feature is written to
`prompt_XXXX/step_XX.pt` as bf16 `[seq_len, 3072]`. HunyuanVideo is a single-pass model,
so unlike the Qwen trees there is no cond/uncond split.

Video size, length and step count must match whatever you intend to accelerate later;
the script defaults mirror `run_sample.sh`. To extract FLUX features with the shared
trainer instead, use `generate_features_flux.sh`.

### 2. Train the predictor

```bash
bash run_train.sh
```

This runs `src/flux/train_encoder_decoder.py`. The settings that matter:

- `--data_dir ./cache_data/hunyuan`
- `--dim 3072`, `--split_dims 2304,384,384` — the `z0/z1/z2` partition, which must sum
  to `dim`
- `--num_blocks` (default 2, override with `NUM_BLOCKS=`), `--hidden_dim`
  (default 64, override with `HIDDEN_DIM=`), `--dropout 0.0`
- `--intervals 3,4,5,6,7,8,9,10` with `--random_interval`, so each sample draws an
  interval at random and one predictor generalises across them
- `--train_prompts 0-946`, `--num_val_prompts 20` (taken at regular spacing),
  `--prompts_per_epoch 20`
- `--epochs 100`, `--batch_size 1`, `--lr 2e-5`, `--grad_accum_steps 64`, `--amp`,
  `--early_stop_patience 20`
- Logging to TensorBoard (`outputs/<exp>/tensorboard`) and wandb (project
  `linca-hunyuanvideo`; disable with `--no_wandb`)

The loss decomposes both the target and the cached feature, predicts `z0/z1/z2` at
orders 0/1/2, and combines the `compose` reconstruction MSE with
`z_loss_weight` (0.1) times the per-partition MSE. The invertible transform is forced
to float32 for numerical stability.

Best weights land at `outputs/<exp>_<timestamp>/best_predictor.pt` alongside
`best_predictor_config.json`, which `from_pretrained` needs — keep the two together.

### 3. Optional: sweep hyperparameters

```bash
bash run_sweep.sh --gpus 0,1,2,3,4,5,6,7 --blocks 1,2,3,4 --hiddens 256 --intervals 6
```

Each GPU takes one `(NUM_BLOCKS, HIDDEN_DIM)` combination at a time and picks up the
next when it finishes. Results are collected in `sweep_logs_*/results.txt`.

### 4. Generate video with the predictor

```bash
bash run_sample.sh outputs/<exp_name>/best_predictor.pt
```

This runs `sample_video.py` with `--decompose-method learned --learned-checkpoint
<best_predictor.pt>`. Cache flags (`--interval 6`, `--max-order 2`, `--min-order 0`,
`--first-enhance 3`, `--forecast-method hermite`, `--forecast-steps <interval>`) must
match training. It sweeps the 946 VBench prompts via
`--vbench-json-path --index-start 0 --index-end 945`, distributing work across GPUs
through a `.dispatch_counter` file.

Videos land in `samples/<exp_name>_infer/interval_6/<prompt>-0.mp4`.

The cache flags are registered in `src/hyvideo/config.py`.

### 5. Evaluate

`run_sample.sh` finishes by printing the VBench command to run:

```bash
python eval/vbench/calc_vbench.py --video_path samples/<exp_name>_infer/interval_6
```

There is no `eval/` directory here — VBench scoring is an external toolchain and has to
be set up separately.

## Known limitations

- `src/flux/train_encoder_decoder.py` contains an `eval_generate_images` helper that
  builds an image pipeline for mid-training previews. It is gated behind
  `--eval_interval` (default 0, so off) and wrapped in `try/except`; leave it off for
  video training.
- This tree trains a single predictor. For the two-stage variant, where separate
  predictors cover separate timestep ranges, see [`models/wan2.1`](../wan2.1) or the
  Qwen trees.
