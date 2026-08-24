# LinCa on Qwen-Image

Cache-accelerated Qwen-Image text-to-image generation with a two-stage learned
invertible predictor. Evaluated on DrawBench in the paper.

## Layout

```
models/qwen_image/
├── linca_data/                    # cache-feature extraction
│   ├── generate_cache_data.py     # single GPU
│   ├── generate_cache_data_ddp.py # multi-GPU
│   ├── prompts/
│   └── pipeline/
└── qwen_image_linca_two_stage/    # training and sampling
    ├── train.py, run_train.sh, run_tiny.sh, run_large.sh
    ├── sample_learned.py          # cache-accelerated sampling
    ├── sample.py, sample_ddp.py   # baselines
    ├── export_inference_checkpoint.py
    ├── invertible_net.py          # the LinCa predictor
    ├── dataset.py
    ├── cache_functions/
    └── prompts/
```

## Setup

```bash
pip install -r requirements.txt
```

`transformers>=4.51` is required: below that, `Qwen2_5_VLForConditionalGeneration`
cannot be imported and the pipeline fails to load.

Qwen-Image itself is pulled from the Hub on first use (`Qwen/Qwen-Image`, about
40 GB). Pass `--model_path` to point at a local copy instead.

## Sampling with the released predictor

Skip training entirely if you just want accelerated generation:

```bash
huggingface-cli download QHRQQQ/LinCa qwen-image_checkpoint.pt --local-dir ./checkpoints

cd models/qwen_image/qwen_image_linca_two_stage
python sample_learned.py \
  --checkpoint ../../../checkpoints/qwen-image_checkpoint.pt \
  --prompt_file prompts/DrawBench200.txt \
  --output_dir samples/qwen_image
```

The released checkpoint carries its own architecture config, so nothing else has to
be configured. **Keep the code defaults** (`--interval 7`, `--forecast_method hermite`,
`--z2_forecast_method lagrange`, `--num_steps 50`): they are what this checkpoint was
trained for. Only change them if you retrain.

Useful flags:

| Flag | Default | Meaning |
|---|---|---|
| `--interval` | `7` | steps between full computations; higher is faster and less exact |
| `--limit` | all | stop after N prompts, for a quick smoke test |
| `--width`, `--height` | `1328` | output resolution |
| `--num_steps` | `50` | denoising steps |
| `--true_cfg_scale` | `4.0` | classifier-free guidance |
| `--seed` | `0` | base seed |
| `--z2_forecast_method` | `lagrange` | second-order forecast; `hermite` matches the FreqCa baseline |
| `--test_FLOPs` | off | report FLOPs instead of only wall-clock (needs `calflops`) |

Multi-GPU sampling goes through `sample_ddp.py` under `torchrun`.

## Training your own predictor

### 1. Extract cache features

Runs full 50-step inference over a prompt set and stores the intermediate DiT
features the predictor learns to forecast.

```bash
cd models/qwen_image/linca_data

python generate_cache_data.py \
  --prompt_file prompts/prompts_train.txt \
  --output_dir data/cache_data \
  --seed 0

# or across 4 GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 generate_cache_data_ddp.py \
  --prompt_file prompts/prompts_train.txt \
  --output_dir data/cache_data
```

This writes `data/cache_data/prompt_XXXX/{cond,uncond}/`. Around 200-400 prompts is
enough; the paper uses 400.

### 2. Train

```bash
cd models/qwen_image/qwen_image_linca_two_stage

# Point DATA_DIR at step 1's output, then:
bash run_train.sh          # optionally pass a wandb key
bash run_tiny.sh           # small config, for checking the loop runs
bash run_large.sh          # the large config
```

Checkpoints land in `outputs/linca_two_stage_*/checkpoint_epoch_*.pt`. One predictor
is trained per timestep segment, hence "two-stage".

### 3. Shrink the checkpoint for inference

Training checkpoints carry optimizer state. Export an inference-only copy:

```bash
python export_inference_checkpoint.py \
  --input checkpoints/checkpoint_full.pt \
  --output checkpoints/checkpoint.pt
```

The exported file embeds the predictor config, and produces identical images.

## Troubleshooting

- `ImportError: Qwen2_5_VLForConditionalGeneration` — upgrade `transformers` to 4.51 or newer.
- "data directory does not exist" during training — run step 1 first, and check `DATA_DIR`.
- Out of memory while training — lower `batch_size`, or start from `run_tiny.sh`.
- Out of memory while sampling — lower `--width`/`--height`; 1328×1328 at 50 steps is the paper setting.
