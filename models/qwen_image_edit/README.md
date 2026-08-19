# LinCA - Qwen-Image-Edit 图编辑加速

基于 LinCA 两阶段可逆分解网络的 Qwen-Image-Edit 推理加速，支持 GEdit-Bench 等图编辑任务。

**运行环境**：`conda activate qwen_image`

## 环境配置

```bash
conda activate qwen_image
pip install transformers>=4.49.0   # 需 >=4.49 以支持 Qwen2.5-VL
# 其他依赖与 qwen_image 一致
```

## 全流程

### 1. 生成 cache 数据

从 `gedit_bench_numbered` 每隔 6 取 1（202 条）生成训练 cache：

```bash
cd qwen_edit/qwen_edit_data

# 单卡（测试）
python generate_cache_data_edit.py --limit 1

# 多卡（完整 202 条）
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 generate_cache_data_edit.py --ddp

# 或
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_generate_cache_ddp.sh
```

输出：`./cache_data/qwen_edit/sample_XXXX/` 及 `index.json`。

### 2. 训练可逆网络

```bash
cd qwen_edit/qwen_image_edit_linca_two_stage

# 确保 CACHE_DATA_DIR 指向步骤 1 的输出
# run_train_ddp.sh 中默认: CACHE_DATA_DIR="./cache_data/qwen_edit"

NPROC=4 MASTER_PORT=29600 bash run_train_ddp.sh [WANDB_KEY]
```

输出：`outputs/qwen_edit_linca_two_stage*/.../checkpoint_epoch_50.pt`。

### 3. 推理生成编辑图片

**3.1 针对 gedit_bench_numbered 全量推理**

```bash
cd qwen_edit/qwen_image_edit_linca_two_stage

# 单卡
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 sample_learned_ddp.py \
  --checkpoint_dir checkpoints/checkpoint_epoch_50.pt \
  --dataset_path ./data/gedit_bench \
  --output_dir samples/qwen_edit_1212_full \
  --seed 0 --interval 7

# 多卡
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29601 sample_learned_ddp.py \
  --checkpoint_dir checkpoints/checkpoint_epoch_50.pt \
  --dataset_path ./data/gedit_bench \
  --output_dir samples/qwen_edit_1212_full \
  --seed 0 --interval 7
```

Output directory：`fullset/<task_type>/<lang>/<key>.png`，**输出会自动 resize 回原图尺寸**。

**3.2 普适单张/批量编辑（不依赖 gedit_bench）**

```bash
cd qwen_edit/qwen_image_edit_linca_two_stage

# 单张
python sample_edit_single.py --checkpoint checkpoints/checkpoint.pt \
  --image demo_pairs/images/img_1.png \
  --prompt "将背景改为城市街道" \
  --output output.png

# 批量（demo_pairs 内含 10 组）
python sample_edit_single.py --checkpoint checkpoints/checkpoint.pt \
  --pairs_file demo_pairs/pairs.csv \
  --output_dir samples/demo_edit

# 快速验证（仅处理 2 条）
python sample_edit_single.py --checkpoint checkpoints/checkpoint.pt \
  --pairs_file demo_pairs/pairs.csv --output_dir samples/demo_edit --limit 2
```

## 目录结构

```
qwen_edit/
├── README.md                    # 本文件
├── qwen_edit_data/              # cache 数据生成
│   ├── generate_cache_data_edit.py
│   └── run_generate_cache_ddp.sh
└── qwen_image_edit_linca_two_stage/   # 训练与推理
    ├── train.py
    ├── train_ddp.py
    ├── run_train_ddp.sh
    ├── sample_learned_ddp.py    # gedit_bench 全量推理
    ├── sample_edit_single.py    # 普适单张/批量推理
    ├── sample_baseline_ddp.py   # 原始模型 baseline（无 cache）
    ├── checkpoints/
    ├── demo_pairs/              # 示例 10 组 图片+prompt
    │   ├── images/
    │   └── pairs.csv
    └── cache_functions/
```

## 推理 checkpoint 瘦身

训练保存的 checkpoint 含 optimizer 等，体积约 1.8GB。仅推理时可导出瘦身版（约 600MB）：

```bash
cd qwen_image_edit_linca_two_stage
python export_inference_checkpoint.py --input checkpoints/checkpoint_full.pt --output checkpoints/checkpoint.pt
```

推理结果与完整 checkpoint 一致。

## 重要说明

- **2阶预测方法**：可通过 `--z2_forecast_method` 选择 `lagrange`（默认）或 `hermite`（与 freqca 一致）。
- **输出 resize**：推理时输入会 resize 到 1024×1024，生成结果会**自动 resize 回原图尺寸**后保存。
- **断点续跑**：`sample_learned_ddp.py` 会检查输出文件是否存在，已存在则跳过，支持中断后继续。
- **transformers 版本**：需 `>=4.49.0`，否则无法导入 Qwen2.5-VL。
