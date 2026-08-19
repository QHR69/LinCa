# qwen_edit LinCA Cache 数据生成

生成 202 条训练样本的 cond/uncond cache，用于 LinCA 训练。

## 环境

需使用 `qwen_image` conda 环境：
```bash
conda activate qwen_image
```

## 数据与流程

- **数据源**: `gedit_bench_numbered`，每隔 6 取 1（indices 0, 6, 12, ..., 1206）
- **尺寸**: 统一 resize 到 1024×1024
- **seed**: `base_seed + original_idx`（与 LinCA sample_edit 一致）
- **Cache**: norm_out 后、proj_out 前 3072 维，仅 latent 部分 `[:, :4096]`
- **超参**: num_steps=50, negative_prompt=" ", true_cfg_scale=4.0, guidance_scale=1.0, max_sequence_length=512

## 输出

- **Cache**: `./cache_data/qwen_edit/sample_XXXX/cond/step_XX.pt`, `uncond/step_XX.pt`
- **编辑图**: 当前目录 `sample_XXXX.png`
- **索引**: `index.json`

## 运行

### 单卡（测试 1 条）

```bash
cd models/qwen_image_edit/qwen_edit_data
conda activate qwen_image
python generate_cache_data_edit.py --limit 1
```

### 多卡（202 条）

```bash
cd models/qwen_image_edit/qwen_edit_data
conda activate qwen_image

# 4 卡
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 generate_cache_data_edit.py --ddp

# 或使用脚本（自动 conda activate + 检测 GPU 数）
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_generate_cache_ddp.sh
```

### 验证

```bash
conda activate qwen_image
python verify_cache_sample.py 0
```

预期输出：cond/uncond shape `(4096, 3072)`，每样本约 2.4GB（50 步×cond+uncond），202 条约 473GB。
