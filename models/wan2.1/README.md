# LinCa on Wan2.1-T2V

Cache-accelerated Wan2.1 text-to-video generation, using a **two-stage** predictor.

We do not publish a trained predictor for Wan2.1 — train your own with the recipe
below.

## Method

The invertible network is the same one used everywhere else in LinCa
(`src/flux/modules/invertible_net.py`): a step's DiT feature is decomposed into
`z0 / z1 / z2`, each partition is extrapolated at order 0 / 1 / 2, and the inverse
transform rebuilds the feature so the step can skip the DiT entirely. See
[`models/hunyuanvideo`](../hunyuanvideo#method) for the block-level description.

What differs here:

- **Two predictors instead of one.** The 50-step schedule is split at
  `STAGE1_MAX_STEP = 24` / `STAGE2_MIN_STEP = 25`: steps 0–24 use
  `predictor_stage1`, steps 25–49 use `predictor_stage2`. Both train at once, with
  each sample's loss routed by which range its `target_step` falls in. Early stopping
  waits until *both* stages have gone `--early_stop_patience` epochs without
  improving.
- **Feature dimension 1536**, so `--dim 1536 --split_dims 1152,192,192` — not
  HunyuanVideo's 3072 / `2304,384,384`.
- **Two CFG streams.** Wan runs classifier-free guidance, so feature extraction
  stores `cond` and `uncond` separately and the predictor learns both.

At inference, `get_predictor(step)` in
`src/flux/modules/cache_functions/cache_utils_learned.py` picks the stage by step
index. Steps 0–24 store their decomposition under both stages, since a step in that
range can be consumed by either predictor; steps 25–49 store stage 2 only.

The hook into Wan's sampler lives in `wan/taylorseer/forwards/wan_forward.py`, which
switches to the learned cache when `decompose_method == 'learned'` and otherwise runs
the TaylorSeer baseline.

## Layout

```
models/wan2.1/
├── generate_features.sh          # extract cond/uncond cache features
├── generate_cache_features.py    # feature extractor
├── run_train.sh                  # train both predictor stages
├── run_sweep.sh                  # multi-GPU hyperparameter grid
├── run_sample.sh                 # accelerated video generation
├── sample_wan.py                 # Wan sampler with two-stage learned cache
├── requirements.txt              # what this tree needs
├── requirements_wan.txt          # upstream Wan2.1's own declared minimums
├── prompts/DrawBench200.txt
├── src/flux/                     # the method + trainer
│   ├── modules/
│   │   ├── invertible_net.py            # invertible predictor
│   │   └── cache_functions/
│   │       └── cache_utils_learned.py   # two-stage learned cache
│   ├── train_encoder_decoder.py         # two-stage training entry point
│   └── dataset.py                       # builds cond/uncond prediction samples
└── wan/                          # Wan2.1 backbone
    ├── text2video.py             # WanT2V pipeline
    ├── image2video.py            # image-to-video (unused by LinCa)
    ├── configs/                  # wan_t2v_1_3B / wan_t2v_14B / wan_i2v_14B
    ├── modules/                  # DiT, VAE, T5, CLIP, tokenizers, attention
    ├── distributed/              # FSDP, xDiT context parallel
    ├── utils/                    # fm_solvers (UniPC), prompt_extend
    └── taylorseer/               # TaylorSeer baseline + LinCa hook
        ├── forwards/wan_forward.py
        ├── generates/
        └── cache_functions/
```

## Setup

```bash
pip install -r requirements.txt
```

`requirements_wan.txt` is upstream Wan2.1's own dependency declaration (with
`flash_attn`, `dashscope`, `easydict`, `ftfy` and friends) — useful for
cross-checking, but `requirements.txt` is what this tree actually needs.

Nothing needs pip-installing from this directory; the scripts set `PYTHONPATH` so that
`flux` and `wan` import.

## What you need to supply

1. **Wan2.1 weights**, from `Wan-AI/Wan2.1-T2V-1.3B` on HuggingFace:

   ```
   checkpoints/Wan2.1-T2V-1.3B/
   ├── config.json
   ├── diffusion_pytorch_model*.safetensors   # DiT
   ├── models_t5_umt5-xxl-enc-bf16.pth        # T5 text encoder
   ├── Wan2.1_VAE.pth
   └── google/umt5-xxl/                       # tokenizer
   ```

   Pass the directory with `--ckpt_dir` (the scripts default to the path above).
2. **VBench prompts** — `VBench_full_info.json`, from the VBench repository, passed
   with `--vbench-json-path`.
3. **Cache features** at `./cache_data/wan`, produced by step 1 below.

## Pipeline

### 1. Extract cache features

```bash
# bash generate_features.sh <wan_ckpt_dir> [start_idx] [end_idx]
bash generate_features.sh ./checkpoints/Wan2.1-T2V-1.3B            # all prompts
bash generate_features.sh ./checkpoints/Wan2.1-T2V-1.3B 0 200      # shard 0-200
```

Each prompt runs full inference with `first_enhance` set to the step count so nothing
is skipped, and both CFG streams are written per step:

```
cache_data/wan/
└── prompt_0000/
    ├── step_00_cond.pt      # [seq_len, 1536], bf16
    ├── step_00_uncond.pt
    ├── step_01_cond.pt
    └── ...
```

`dataset.py` accepts either this flat layout or the nested
`prompt_XXXX/cond/step_XX.pt` form.

### 2. Train both predictor stages

```bash
bash run_train.sh
```

Wan's settings differ from HunyuanVideo's, and `run_train.sh` already carries them:

```bash
python src/flux/train_encoder_decoder.py \
    --data_dir ./cache_data/wan \
    --dim 1536 --split_dims 1152,192,192 \
    --num_blocks 6 --hidden_dim 512 --dropout 0.1 \
    --intervals 6,7,8,9,10 --random_interval \
    --train_prompts 0-946 --num_val_prompts 20 --prompts_per_epoch 30 \
    --epochs 50 --batch_size 4 --lr 1e-4 --grad_accum_steps 5 \
    --early_stop_patience 10 --amp \
    --output_dir outputs --wandb_project linca-wan
```

Two checkpoints come out, each with the `_config.json` that `from_pretrained` needs:

- `outputs/<exp>_<ts>/best_predictor_stage1.pt` (+ `_config.json`)
- `outputs/<exp>_<ts>/best_predictor_stage2.pt` (+ `_config.json`)

`best_model.pt` additionally holds both state dicts in one file for resuming.

### 3. Optional: sweep hyperparameters

```bash
bash run_sweep.sh --gpus 0,1,2,3 --blocks 1,2,3,4 --hiddens 256 --intervals 6
```

Each GPU takes one `(NUM_BLOCKS, HIDDEN_DIM)` combination at a time; results collect in
`sweep_logs_*/results.txt`.

### 4. Generate video with the predictors

```bash
# bash run_sample.sh <wan_ckpt_dir> <predictor_dir> [interval ...]
bash run_sample.sh ./checkpoints/Wan2.1-T2V-1.3B outputs/<exp>_<ts>

# sweep several intervals in one go
bash run_sample.sh ./checkpoints/Wan2.1-T2V-1.3B outputs/<exp>_<ts> 6 8 10
```

`<predictor_dir>` is a training output directory; the script picks up
`best_predictor_stage1.pt` and `best_predictor_stage2.pt` from it. Videos land under
`samples/<exp>_infer/interval_<N>/`. Underneath it runs:

```bash
python sample_wan.py \
    --task t2v-1.3B --size 832*480 \
    --ckpt_dir checkpoints/Wan2.1-T2V-1.3B \
    --vbench-json-path /path/to/VBench_full_info.json \
    --index-start 0 --index-end -1 \
    --save_folder samples/wan_learned_infer \
    --decompose-method learned \
    --learned-checkpoint-stage1 outputs/<exp>/best_predictor_stage1.pt \
    --learned-checkpoint-stage2 outputs/<exp>/best_predictor_stage2.pt \
    --interval 3 --max-order 2 --min-order 0 --first-enhance 3 \
    --forecast-method hermite --base_seed 42
```

Notes:

- `--decompose-method learned` **requires both** stage checkpoints; `sample_wan.py`
  asserts on it. The default is `taylor`, i.e. the TaylorSeer baseline.
- Cache settings are pushed into `wan_t2v.model.learned_cache_config`
  (`interval`, `max_order`, `min_order`, `first_enhance`, `forecast_method`,
  `forecast_steps`, `use_z_cache`) and must match training. `forecast_steps`
  defaults to `interval`.
- Single prompts (`--prompt`) and prompt files (`--prompt_file`) work too. VBench
  batches distribute across GPUs through a `.dispatch_counter` file.

### 5. Evaluate

VBench scoring is an external toolchain; there is no `eval/` directory here. Point it
at the videos under `samples/`.

## Known limitations

- `wan/image2video.py` and the i2v configs ship with the backbone but are unused by
  LinCa, which targets text-to-video.
- Mid-training sample previews are disabled (`--eval_interval 0`): rendering one
  would need the full generation pipeline resident alongside the trainer.
