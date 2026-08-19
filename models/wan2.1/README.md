# Wan2.1-T2V Learned-Cache 加速项目（代码归档）

This directory is a lightweight code archive of Wan2.1 + LinCa: it keeps the code, scripts, configs, prompts, and analysis notes needed to reproduce experiments. It does **not** include training data, trained weights, Wan backbone weights (`checkpoints/`), wandb logs, sample videos, or VBench evaluation artifacts. This README is based only on the archived code.

---

## 1. 项目是做什么的

这是一个 **Diffusion Cache 推理加速**项目（FreqCa 方法的 v4 迭代），落地在 **Wan2.1 文生视频（T2V）** 上。方法与 `hunyuanvideo` 项目同源：训练**可逆分解网络** `InvertibleDecompositionNet`，把 DiT 中间特征分解成 `z0 / z1 / z2` 三分区，分别用 0/1/2 阶外推预测、再可逆重建，从而跳过大部分 full step。

**与 hunyuanvideo 的关键差异 —— 两阶段（two-stage）预测器**（见 `src/flux/train_encoder_decoder.py`、`src/flux/modules/cache_functions/cache_utils_learned.py`）：
- 50 步采样被切成两段：**步 0–24 用 `predictor_stage1`，步 25–49 用 `predictor_stage2`**（`STAGE1_MAX_STEP = 24`、`STAGE2_MIN_STEP = 25`）。
- 训练时两个 predictor 同时训练，按 `target_step` 落在哪一段路由到对应 stage 计算损失；早停策略为“两个 stage 都连续 N epoch 无提升才停”（Early Stopping B）。
- 推理时 `cache_utils_learned.py` 的 `get_predictor(step)` 按步数返回对应 stage；步 0–24 的特征分解会同时存 stage1 与 stage2 两份（因可能被两阶段预测器分别使用），步 25–49 只存 stage2。
- 数据采集区分 CFG 的 **cond / uncond 两条流**（Wan 是带 CFG 的模型），特征维度为 **1536**，默认 `--split_dims 1152,192,192`（注意与 hunyuanvideo 的 3072 / 2304,384,384 不同）。

可逆网络结构本身（`src/flux/modules/invertible_net.py`）与 hunyuanvideo 完全一致（已校验 diff 无差异）：`LightweightInvertible1x1Conv` + `RevNetResidualBlock` 组成 `HybridInvertibleBlock`，对外类 `LearnedDecompositionPredictor`。

Wan 推理的 cache 挂接在 `wan/taylorseer/`（`forwards/wan_forward.py` 检测 `decompose_method=='learned'` 后调用 `flux.modules.cache_functions.cache_utils_learned`）。

---

## 2. 目录结构

```
wan2.1/
├── README.md                       # 本文件
├── environment.yml                 # conda 完整环境导出（带 build hash）
├── environment.history.yml         # conda --from-history
├── requirements.txt                # pip freeze（本环境精确版本）
├── requirements_wan.txt            # 上游 Wan2.1 官方依赖声明（版本下限）
├── pyproject.toml / setup.py       # 包配置
│
├── generate_features.sh            # 壳脚本（注意：内容为 FLUX 版，见 §5.1 提醒）
├── generate_cache_features.py      # 【数据准备】Wan 离线特征采集主程序（cond/uncond 双流）
├── run_train.sh                    # 【训练】壳脚本（注意：内容指向 hunyuan，见 §5.2 提醒）
├── run_sweep.sh                    # 【训练】多 GPU 超参网格搜索
├── run_sample.sh                   # 壳脚本（注意：内容为 HunyuanVideo 版，见 §5.4 提醒）
├── sample_wan.py                   # ★【推理】Wan 真实推理入口（两阶段 learned cache）
├── sample_video.py                 # HunyuanVideo 推理脚本（随 src 携带，Wan 不用）
│
├── calc_flops.py / calc_flops_flux.py / flops_analysis.md
├── visualize_intro_figures*.py     # 论文配图脚本（3 个）
├── save_init_model.py
│
├── prompts/
│   ├── DrawBench200.txt
│   └── parti_prompts.txt
│
├── wan/                            # Wan2.1 实现
│   ├── text2video.py               # WanT2V 管线
│   ├── image2video.py
│   ├── configs/                    # wan_t2v_1_3B / wan_t2v_14B / wan_i2v_14B / shared_config
│   ├── modules/                    # model(DiT) / vae / t5 / clip / tokenizers / attention / xlm_roberta
│   ├── distributed/                # fsdp / xdit_context_parallel
│   ├── utils/                      # fm_solvers(_unipc) / prompt_extend / qwen_vl_utils / utils
│   └── taylorseer/                 # ★ TaylorSeer + learned cache 挂接
│       ├── forwards/wan_forward.py # ★ 检测 learned 并调用自研 cache
│       ├── generates/              # wan_t2v_generate / wan_i2v_generate
│       └── cache_functions/        # cache_init / cal_type / force_scheduler
│
└── src/                            # 与 hunyuanvideo 同源的方法代码（+ 上游 FLUX/hyvideo）
    ├── flux/
    │   ├── modules/
    │   │   ├── invertible_net.py           # ★ 可逆分解网络（与 hunyuan 一致）
    │   │   └── cache_functions/
    │   │       └── cache_utils_learned.py  # ★ 两阶段 learned cache 挂接
    │   ├── train_encoder_decoder.py        # ★ 两阶段训练入口
    │   ├── dataset.py                       # ★ 数据集（cond/uncond 双流，dim=3072 默认；Wan 用 1536）
    │   └── ...                              # cli / model / sampling / trt 等上游代码
    └── hyvideo/                    # HunyuanVideo 底模实现（随 src 携带，Wan 推理不用）
```

---

## 3. 运行环境

与 `hunyuanvideo` 项目共用同一 conda 环境：

- Environment: `conda activate hunyuanvideo`
- 激活：`conda activate hunyuanvideo`
- 关键版本（摘自 `requirements.txt`）：`python 3.10`、`torch==2.7.1+cu126`、`torchvision==0.22.1+cu126`、`diffusers==0.30.3`、`transformers==4.45.2`、`accelerate==1.13.0`、`einops==0.8.2`、`loguru==0.7.3`、`opencv-python-headless==4.11.0.86`、`wandb==0.26.1`。
- `requirements_wan.txt` 为上游 Wan 官方声明的依赖下限（如 `flash_attn`、`dashscope`、`easydict`、`ftfy` 等），供参考对照。

环境文件（本目录已附）：
- `environment.yml` —— `conda env create -f environment.yml` 精确复现。
- `environment.history.yml` —— 仅显式安装项，跨平台参考。
- `requirements.txt` —— `pip install -r requirements.txt`。

---

## 4. 权重 / 数据存放结构（本归档不含，需自行准备）

1. **Wan2.1 底模**（源项目放在 `wan_v4/checkpoints/Wan2.1-T2V-1.3B/`，归档已**整体剔除**）。需从 HuggingFace `Wan-AI/Wan2.1-T2V-1.3B` 重新下载，目录结构（源仅残留 config/tokenizer，下载后应含）：
   ```
   checkpoints/Wan2.1-T2V-1.3B/
   ├── config.json
   ├── diffusion_pytorch_model*.safetensors   # DiT 权重（大）
   ├── models_t5_umt5-xxl-enc-bf16.pth         # T5 文本编码器（大）
   ├── Wan2.1_VAE.pth                          # VAE（大）
   ├── google/umt5-xxl/                        # tokenizer（spiece.model / tokenizer.json 等）
   ├── README.md / LICENSE.txt
   └── ...
   ```
   推理/采集脚本通过 `--ckpt_dir <该目录>` 指定。
2. **训练数据**：`src/flux/train_encoder_decoder.py` 默认 `--data_dir ./cache_data/wan`。由 `generate_cache_features.py` 生成，每个 prompt 每步存**两份**（cond / uncond）：
   ```
   cache_data_vbench/
   └── prompt_0000/
       ├── step_00_cond.pt     # [seq_len, 1536]，bf16
       ├── step_00_uncond.pt
       ├── step_01_cond.pt
       └── ...
   ```
   （`dataset.py` 同时兼容 `prompt/cond/step_XX.pt` 嵌套式与 `prompt/step_XX_cond.pt` 扁平式两种布局。）
3. **VBench prompts**：通过 `--vbench-json-path` 指定 `VBench_full_info.json`（外部 TaylorSeer / VBench）。
4. **训练输出**（`--output_dir outputs`）：两阶段各存一份——
   - `outputs/<exp>_<ts>/best_predictor_stage1.pt`（+ `_config.json`）
   - `outputs/<exp>_<ts>/best_predictor_stage2.pt`（+ `_config.json`）
   - 另有 `best_model.pt`（含 `predictor_stage1`/`predictor_stage2` 两个 state_dict）、周期 checkpoint。

---

## 5. 全流程指南（命令均与归档代码一致）

> 先 `conda activate hunyuanvideo` 并 `cd` 到本目录。

### 5.1 数据准备（生成缓存特征，Wan 双流）

**重要提醒**：归档中的 `generate_features.sh` 内容是 FLUX 版（调 `src/flux/generate_cache_features.py`），**不是** Wan 流程。Wan 的真实采集入口是顶层 `generate_cache_features.py`。参数取自其 `parse_args()`：

```bash
python generate_cache_features.py \
    --vbench-json-path /path/to/VBench_full_info.json \
    --output_dir cache_data_vbench \
    --index-start 0 --index-end 945 \
    --task t2v-1.3B \
    --size 832*480 \
    --frame_num 81 \
    --ckpt_dir checkpoints/Wan2.1-T2V-1.3B \
    --sample_steps 50 --sample_shift 8.0 --sample_solver unipc \
    --sample_guide_scale 6.0 --base_seed 42 \
    --offload_model --t5_cpu
```

该脚本对每个 prompt 跑 full 推理，对 **cond_stream 与 uncond_stream 各保存一次**每步特征（`save_stream_feature` 写 `step_XX_cond.pt` / `step_XX_uncond.pt`），通过 `reset_collect_cache` 设 `first_enhance=num_steps` 使所有步皆 full。也支持 `--prompt_file`。

### 5.2 训练两阶段 learned predictor

**重要提醒**：归档中的 `run_train.sh` 与 hunyuanvideo 的完全相同（已校验 diff 无差异），其 `--data_dir` 写死为 `hunyuan_v4/cache_data_vbench`、`--dim 3072 --split_dims 2304,384,384`，这是 **HunyuanVideo 的配置**。Wan 的真实训练应直接调用 `src/flux/train_encoder_decoder.py` 并使用 Wan 的默认值（见该文件 `argparse`）：

```bash
python src/flux/train_encoder_decoder.py \
    --data_dir ./cache_data/wan \
    --dim 1536 --split_dims 1152,192,192 \
    --num_blocks 6 --hidden_dim 512 --dropout 0.1 \
    --intervals 6,7,8,9,10 --random_interval \
    --train_prompts 0-946 --num_val_prompts 20 --prompts_per_epoch 30 \
    --epochs 50 --batch_size 4 --lr 1e-4 --grad_accum_steps 5 \
    --early_stop_patience 10 --amp \
    --output_dir outputs --wandb_project linca_v4_two_stage
```

要点（均来自代码）：
- 两阶段：`STAGE1_MAX_STEP=24` / `STAGE2_MIN_STEP=25`；`stage1_mask = target_steps<=24`、`stage2_mask = target_steps>=25` 路由损失。
- 维度：`--dim 1536`、`--split_dims 1152,192,192`（和须等于 dim）。
- 早停：两个 stage 都连续 `--early_stop_patience` epoch 无提升才停。
- 产物：`best_predictor_stage1.pt`、`best_predictor_stage2.pt`（各带 `_config.json`）。

### 5.3 超参网格搜索（可选）

```bash
bash run_sweep.sh --gpus 0,1,2,3 --blocks 1,2,3,4 --hiddens 256 --intervals 6
```
机制同 hunyuanvideo（原子任务队列，`results.txt` 记录结论）。注意它内部仍调用 `run_train.sh`，受 §5.2 的配置提醒影响。

### 5.4 加载权重推理（生成视频，两阶段）

**重要提醒**：归档中的 `run_sample.sh` 内容是 HunyuanVideo 版（调 `sample_video.py`、单 `--learned-checkpoint`），**不能**直接用于 Wan。Wan 的真实推理入口是 `sample_wan.py`，参数取自其 `_parse_args()`：

```bash
python sample_wan.py \
    --task t2v-1.3B \
    --size 832*480 \
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

要点（来自 `sample_wan.py`）：
- `learned` 模式**强制要求**同时给 `--learned-checkpoint-stage1` 与 `--learned-checkpoint-stage2`（断言）。
- 加载后通过 `load_predictor(stage1, stage2, device)` 注册两阶段预测器，并把 cache 配置写入 `wan_t2v.model.learned_cache_config`（`interval/max_order/min_order/first_enhance/forecast_method/forecast_steps/use_z_cache`）。
- `forecast_steps` 默认取 `interval`。
- 也支持单 prompt（`--prompt`）、`--prompt_file`、`--save_folder`；VBench 批量用 `.dispatch_counter` 抢占式多 GPU 分发。
- 默认 `--decompose-method taylor`（原始 TaylorSeer），传 `learned` 才走自研可逆 cache。

### 5.5 评测（VBench）

源项目的 VBench 评测产物原在 `wan_v4/samples/<exp>/vbench/*.json`（归档已剔除）。评测脚本属外部 VBench / TaylorSeer 工具链，本归档与源项目均**不含** `eval/` 目录，需自行准备后对 `samples/` 下生成的视频评测。

---

## 6. 已知注意点（基于代码事实）

1. `run_train.sh`、`run_sample.sh`、`generate_features.sh` 三个壳脚本均与 hunyuanvideo 雷同，内容指向 HunyuanVideo/FLUX 流程，**不可直接用于 Wan**；Wan 的真实入口分别是 `src/flux/train_encoder_decoder.py`、`sample_wan.py`、`generate_cache_features.py`（见 §5）。
2. Wan 是两阶段预测器（stage1/stage2）、特征维度 1536、数据分 cond/uncond 两流；与 hunyuanvideo（单预测器、3072、单流）不同。
3. 多处数据/底模路径为绝对写死路径，新位置运行需手动修改。
4. `wan/` 内仍保留 i2v（图生视频）相关实现与 `sample_video.py`（HunyuanVideo），属随仓携带，T2V learned-cache 主线不使用。
5. `cache_utils_learned.py` 中 `pipeline_with_learned_cache` 引用了 `pipeline.transformer_qwenimage`（Qwen-Image 相关），属跨项目遗留，Wan T2V 推理路径不经过它。
