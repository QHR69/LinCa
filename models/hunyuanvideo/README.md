# HunyuanVideo Learned-Cache 加速项目（代码归档）

This directory is a lightweight code archive of HunyuanVideo + LinCa: it keeps the code, scripts, configs, prompts, and analysis notes needed to reproduce experiments. It does **not** include training data, trained weights, backbone weights, wandb logs, or sample videos. This README is based only on the archived code.

---

## 1. 项目是做什么的

这是一个 **Diffusion Cache 推理加速**项目（原始方法 FreqCa 的 v4 迭代），落地在 **HunyuanVideo（文生视频 T2V）** 上。

核心思想（见 `src/flux/modules/invertible_net.py`）：训练一个**可逆分解网络** `InvertibleDecompositionNet`，把扩散模型某一步的 DiT 中间特征（维度 3072）通过可逆变换分解成 3 个分区 `z0 / z1 / z2`；推理时对这三个分区分别用 **0 阶 / 1 阶 / 2 阶外推**（`FixedPredictionStrategy`）预测，再经可逆逆变换 `compose` 重建该步特征，从而跳过大部分 full step（cache step）。

- 可逆块结构：`LightweightInvertible1x1Conv`（正交初始化 1x1 卷积，Glow 风格通道混合）+ `RevNetResidualBlock`（RevNet 残差块，精确可逆，F/G 双子网络）组合成 `HybridInvertibleBlock`，堆叠 `num_blocks` 层。
- 对外主类：`LearnedDecompositionPredictor`，封装网络 + 预测策略，提供 `decompose / compose / predict_from_decomposed / save_pretrained / from_pretrained`。`save_pretrained` 会预计算 1x1 卷积逆矩阵以加速推理。
- 推理挂接：`src/flux/modules/cache_functions/cache_utils_learned.py`（HunyuanVideo 为**单预测器**，即一个 checkpoint）。

加速比与 FLOPs 的实测分析见 `flops_analysis.md`（H800 80GB，480×640，65 帧，50 步：单 full step 595.46 TFLOPS，cache step 仅 0.145 TFLOPS）。

> 说明：`src/` 同时包含上游 FLUX 图像推理代码与 HunyuanVideo 底模实现。本项目的“方法代码”集中在 `src/flux/modules/invertible_net.py`、`src/flux/train_encoder_decoder.py`、`src/flux/dataset.py`、`src/flux/modules/cache_functions/`。

---

## 2. 目录结构

```
hunyuanvideo/
├── README.md                       # 本文件
├── environment.yml                 # conda 完整环境导出（带 build hash，精确复现）
├── environment.history.yml         # conda --from-history（仅显式安装，跨平台）
├── requirements.txt                # pip freeze（精确 pip 版本）
├── pyproject.toml / setup.py       # 包配置（上游 FLUX 遗留）
│
├── generate_features.sh            # 【数据准备】壳脚本（注意：调的是 FLUX 版，见 §5.1 提醒）
├── generate_cache_features.py      # 【数据准备】HunyuanVideo 离线特征采集主程序
├── run_train.sh                    # 【训练】训练 learned predictor
├── run_sweep.sh                    # 【训练】多 GPU 超参网格搜索
├── run_sample.sh                   # 【推理+评测】用训练好的 predictor 加速生成视频
├── sample_video.py                 # 【推理】HunyuanVideo 推理主程序
│
├── calc_flops.py / calc_flops_flux.py   # FLOPs 计算
├── flops_analysis.md               # FLOPs / 加速比分析表
├── visualize_intro_figures*.py     # 论文配图脚本（3 个）
├── save_init_model.py              # 保存初始化模型
├── Cache方向数据.docx              # flops_analysis.md 的数据来源
├── img.jpg / mask.jpg              # FLUX inpainting CLI 的示例输入（被 src/sample*.py 引用）
│
├── prompts/
│   ├── DrawBench200.txt
│   └── parti_prompts.txt
│
├── sweep_logs_*/results.txt        # 各轮超参搜索结论（纯文本，原始 .log 已剔除）
│
└── src/
    ├── flux/                       # 方法代码 + 上游 FLUX 推理
    │   ├── modules/
    │   │   ├── invertible_net.py           # ★ 可逆分解网络 + 预测策略（方法核心）
    │   │   └── cache_functions/
    │   │       ├── cache_utils_learned.py  # ★ learned cache 推理挂接（单预测器）
    │   │       ├── cache_utils.py          # 原始 FreqCa cache（fallback）
    │   │       ├── cache_init.py            # cache 字典初始化
    │   │       └── cal_type.py              # 判断当前 step 是 full / cache
    │   ├── train_encoder_decoder.py        # ★ 训练入口
    │   ├── dataset.py                       # ★ 训练数据集（按 interval 构造预测样本）
    │   ├── generate_cache_features.py       # FLUX 版特征采集
    │   ├── model.py / sampling.py / util.py / math_utils.py
    │   ├── cli*.py                          # 上游 FLUX 图像生成命令行
    │   └── trt/                             # 上游 TensorRT 引擎封装
    └── hyvideo/                    # HunyuanVideo 底模实现
        ├── inference.py                     # HunyuanVideoSampler
        ├── config.py                        # 命令行参数（含本项目自定义 cache 参数）
        ├── modules/models.py                # DiT 主干
        ├── vae/                             # 3D 因果 VAE
        ├── diffusion/pipelines/ / schedulers/
        └── text_encoder/
```

---

## 3. 运行环境

两个项目（hunyuanvideo / wan2.1）共用同一个 conda 环境：

- Environment: `conda activate hunyuanvideo`
- 激活：`conda activate hunyuanvideo`
- 关键版本（摘自 `requirements.txt`）：`python 3.10`、`torch==2.7.1+cu126`、`torchvision==0.22.1+cu126`、`diffusers==0.30.3`、`transformers==4.45.2`、`accelerate==1.13.0`、`einops==0.8.2`、`loguru==0.7.3`、`opencv-python-headless==4.11.0.86`、`wandb==0.26.1`、`torch-dct==0.1.6`。

环境文件（本目录已附）：
- `environment.yml` —— `conda env create -f environment.yml` 精确复现（含 build hash）。
- `environment.history.yml` —— 仅含显式安装项（`python=3.10`），跨平台参考。
- `requirements.txt` —— `pip install -r requirements.txt`（pip 精确版本）。

> 注：脚本里 `export PYTHONPATH="${SCRIPT_DIR}/src:${PYTHONPATH}"`，使 `flux` 与 `hyvideo` 可被 import。

---

## 4. 权重 / 数据存放结构（本归档不含，需自行准备）

以下路径来自实际脚本（`run_train.sh`、`run_sample.sh`），归档中**均未包含**对应数据/权重：

1. **HunyuanVideo 底模**（外部，`run_sample.sh` 写死）：
   - `MODEL_BASE = ./checkpoints/hunyuanvideo`
   - DiT 权重：`${MODEL_BASE}/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt`
2. **训练数据**（`run_train.sh` 写死）：
   - `DATA_DIR = ./cache_data/hunyuan`
   - 由 `generate_cache_features.py` 生成，目录格式（见 `src/flux/dataset.py` 与采集脚本）：
     ```
     cache_data_vbench/
     └── prompt_0000/
         ├── step_00.pt   # 张量 [seq_len, 3072]，bf16 保存
         ├── step_01.pt
         └── ... step_49.pt
     ```
     每个 `.pt` 为单步 DiT 输出特征（HunyuanVideo 单 pass，无 CFG cond/uncond 之分）。
3. **VBench prompts**（`run_sample.sh` 写死）：
   - `./data/VBench_full_info.json`（946 条 prompt）。
4. **训练输出**（`run_train.sh` → `OUTPUT_DIR=outputs`）：
   - `outputs/<exp_name>_<timestamp>/best_predictor.pt` + `best_predictor_config.json`（由 `LearnedDecompositionPredictor.save_pretrained` 写出），另有 `best_model.pt`、`config.json`、`tensorboard/`。

---

## 5. 全流程指南（命令均与归档脚本一致）

> 先 `conda activate hunyuanvideo` 并 `cd` 到本目录。

### 5.1 数据准备（生成缓存特征）

**重要提醒**：归档中的 `generate_features.sh` 内容实际调用的是 FLUX 版（`src/flux/generate_cache_features.py`，`MODEL_NAME=flux-dev`、`prompts/DrawBench200.txt`、输出 `cache_data`），**并非** HunyuanVideo 流程。HunyuanVideo 的真实特征采集入口是顶层 `generate_cache_features.py`，需手动按其 docstring 调用，例如（参数取自该文件 docstring 与 `argparse`）：

```bash
python generate_cache_features.py \
    --vbench-json-path ./data/VBench_full_info.json \
    --output_dir cache_data_vbench \
    --index-start 0 --index-end 945 \
    --model-base ./checkpoints/hunyuanvideo \
    --video-size 480 640 --video-length 65 --infer-steps 50 \
    --flow-reverse --seed 42
```

该脚本对每个 prompt 跑 50 步 full 推理（`cache_init(..., first_enhance=num_steps)` 使所有步都是 full），把每步 DiT 特征存为 `prompt_XXXX/step_XX.pt`。也支持 `--prompt_file` 文本格式输入。

### 5.2 训练 learned predictor

```bash
bash run_train.sh
```

`run_train.sh` → `python src/flux/train_encoder_decoder.py`，关键参数（脚本内写死/可经环境变量覆盖）：
- `--data_dir`：`./cache_data/hunyuan`（写死，如改归档路径需同步改）。
- `--dim 3072`、`--split_dims 2304,384,384`（z0/z1/z2 分区，和须等于 dim）、`--num_blocks`（默认 2，可 `NUM_BLOCKS=` 覆盖）、`--hidden_dim`（默认 64，可 `HIDDEN_DIM=` 覆盖）、`--dropout 0.0`。
- `--intervals 3,4,5,6,7,8,9,10` + `--random_interval`（每样本随机选 interval）。
- `--train_prompts 0-946`、`--num_val_prompts 20`（等间隔抽验证集）、`--prompts_per_epoch 20`。
- `--epochs 100`、`--batch_size 1`、`--lr 2e-5`、`--grad_accum_steps 64`、`--amp`、`--early_stop_patience 20`。
- 日志：TensorBoard（`outputs/<exp>/tensorboard`）+ Wandb（project `hunyuan-v4`，可 `--no_wandb` 关闭）。
- 产物：最优权重 `outputs/<exp>_<ts>/best_predictor.pt`（+ `_config.json`）。

损失（`compute_loss`）：分解 target 与 cache 特征后，z0 做 0 阶、z1 做 1 阶、z2 做 2 阶预测，主损失为 `compose` 重建 MSE + `z_loss_weight`（0.1）× 各分量 MSE，可逆变换强制 float32。

### 5.3 超参网格搜索（可选）

```bash
bash run_sweep.sh --gpus 0,1,2,3,4,5,6,7 --blocks 1,2,3,4 --hiddens 256 --intervals 6
```

多 GPU 原子任务队列：每张卡跑完一个 `(NUM_BLOCKS, HIDDEN_DIM)` 组合再领下一个，结果写 `sweep_logs_8gpu_interval3-10/results.txt`，逐任务 log 写各自 `.log`（归档已剔除 `.log`，仅留 `results.txt`）。

### 5.4 加载权重推理（生成视频）

```bash
bash run_sample.sh outputs/<exp_name>/best_predictor.pt
```

`run_sample.sh` → `python sample_video.py`，关键参数：
- `--decompose-method learned --learned-checkpoint <best_predictor.pt>`：启用 learned cache，`sample_video.py` 调 `load_predictor()` 加载预测器。
- `--interval 6`、`--max-order 2`、`--min-order 0`、`--first-enhance 3`、`--forecast-method hermite`、`--forecast-steps <interval>`（cache 参数，需与训练一致）。
- `--vbench-json-path`、`--index-start 0 --index-end 945`：批量跑 VBench 946 prompt（多 GPU 通过 `.dispatch_counter` 抢占式分发）。
- `--video-size 480 640 --video-length 65 --infer-steps 50 --flow-reverse --use-cpu-offload --seed 42`。
- 输出：`samples/<exp_name>_infer/interval_6/<prompt>-0.mp4`。

自定义 cache 参数在 `src/hyvideo/config.py` 注册（`--interval/--max-order/--min-order/--first-enhance/--forecast-method/--decompose-method/--forecast-steps/--learned-checkpoint`）。

### 5.5 评测（VBench）

`run_sample.sh` 末尾提示用 VBench 评测生成的视频：
```bash
python eval/vbench/calc_vbench.py --video_path samples/<exp_name>_infer/interval_6
```
**注意**：本归档与源项目中都**不含** `eval/` 目录；该评测脚本属外部 TaylorSeer 仓库 / VBench 工具链，需自行准备。

---

## 6. 已知注意点（基于代码事实）

1. `generate_features.sh`、`run_train.sh` 中的数据路径为绝对写死路径（`hunyuan_v4/cache_data_vbench`），在新位置运行需手动修改。
2. `generate_features.sh` 实为 FLUX 流程，**不是** HunyuanVideo 特征采集；HunyuanVideo 用顶层 `generate_cache_features.py`（见 §5.1）。
3. `run_sample.sh` / `run_train.sh` 中底模、VBench JSON 路径指向外部 `TaylorSeer/TaylorSeer-HunyuanVideo`。
4. `src/flux/train_encoder_decoder.py` 中的 `eval_generate_images` 在训练时被显式跳过（注释为需完整 pipeline），训练阶段不生成图片。
5. 本归档为单预测器版本；两阶段（stage1/stage2）预测器见 `wan2.1` 项目。
