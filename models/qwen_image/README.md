# LinCA - Qwen-Image 文生图加速

基于 LinCA 两阶段可逆分解网络的 Qwen-Image 推理加速，支持 DrawBench 等文生图任务。

**运行环境**：`conda activate qwen_image`

## 环境配置

```bash
# 创建 conda 环境
conda create -n qwen_image python=3.10 -y
conda activate qwen_image

# 安装依赖（需与 Qwen-Image 官方要求一致）
pip install torch torchvision torchaudio
pip install transformers>=4.49.0   # 需 >=4.49 以支持 Qwen2.5-VL
pip install diffusers
pip install qwen-vl-utils
# 其他依赖见项目 requirements
```

**注意**：`transformers>=4.49.0` 为必需，否则无法导入 `Qwen2_5_VLForConditionalGeneration`。

## 全流程

### 1. 生成 cache 数据

在 `linca_data/` 下运行，生成训练用的中间特征：

```bash
cd qwen_image/linca_data

# 单卡
python generate_cache_data.py \
    --prompt_file prompts/prompts_train.txt \
    --output_dir data/cache_data \
    --seed 0

# 多卡
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 generate_cache_data_ddp.py \
    --prompt_file prompts/prompts_train.txt \
    --output_dir data/cache_data
```

Output directory结构：`data/cache_data/prompt_XXXX/cond/`, `uncond/` 等。

### 2. 训练可逆网络

使用生成的 cache 数据训练两阶段 predictor：

```bash
cd qwen_image/qwen_image_linca_two_stage

# 修改 run_train.sh 中的 DATA_DIR 指向 cache Output directory，例如：
# DATA_DIR="./cache_data"

bash run_train.sh          # 可选传入 wandb key
# 或
bash run_tiny.sh           # 小规模测试
bash run_large.sh          # 大规模配置
```

训练输出在 `outputs/linca_v4_two_stage_*/`，得到 `checkpoint_epoch_*.pt`。

### 3. 推理生成图片

```bash
cd qwen_image/qwen_image_linca_two_stage

# 单卡，使用 checkpoint.pt
CUDA_VISIBLE_DEVICES=0 python sample_learned.py \
    --checkpoint checkpoints/checkpoint.pt \
    --prompt_file prompts/DrawBench200.txt \
    --interval 6 \
    --seed 0 \
    --output_dir samples/test_inter6

# 快速验证（仅生成 2 张）
CUDA_VISIBLE_DEVICES=0 python sample_learned.py \
    --checkpoint checkpoints/checkpoint.pt \
    --prompt_file prompts/DrawBench200.txt \
    --output_dir samples/test_inter6 --limit 2

# 多卡
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 sample_ddp.py \
    --checkpoint checkpoints/checkpoint.pt \
    --prompt_file prompts/DrawBench200.txt \
    --interval 6 \
    --output_dir samples/test_ddp
```

## 目录结构

```
qwen_image/
├── README.md                 # 本文件
├── linca_data/               # cache 数据生成
│   ├── generate_cache_data.py
│   ├── generate_cache_data_ddp.py
│   ├── prompts/
│   └── pipeline/
└── qwen_image_linca_two_stage/   # 训练与推理
    ├── train.py
    ├── run_train.sh
    ├── sample_learned.py
    ├── sample_ddp.py
    ├── checkpoints/          # 存放 checkpoint.pt
    ├── prompts/
    │   ├── DrawBench200.txt
    │   └── prompts_train.txt
    └── dataset.py
```

- **2阶预测方法**：可通过 `--z2_forecast_method` 选择 `lagrange`（默认）或 `hermite`（与 freqca 一致）。

## 推理 checkpoint 瘦身

训练保存的 checkpoint 含 optimizer 等，体积较大。仅推理时可导出瘦身版：

```bash
cd qwen_image_linca_two_stage
python export_inference_checkpoint.py --input checkpoints/checkpoint_full.pt --output checkpoints/checkpoint.pt
```

推理结果与完整 checkpoint 一致。

## 常见问题

- **ImportError: Qwen2_5_VLForConditionalGeneration**：升级 `transformers>=4.49.0`
- **数据目录不存在**：先完成步骤 1 生成 cache，再训练
- **OOM**：减小 `batch_size` 或使用 `run_tiny.sh`
