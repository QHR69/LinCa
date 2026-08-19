# LinCa

**通过可学习分解特征缓存加速扩散模型**

[论文](https://arxiv.org/abs/2608.17973) · [代码](https://github.com/QHR69/LinCa) · [权重](https://huggingface.co/QHRQQQ/LinCa) · [English](README.md)

LinCa 用轻量可逆网络把缓存特征分解成连续性不同的子分量，按分量匹配预测阶数，再重建回原特征空间。映射严格可逆，重建无损。

额外参数 **&lt;0.2%**，在 **5–7×** 加速下保持近无损质量。下方 teaser 为 Qwen-Image + LinCa，加速比 **6.95×**。

<p align="center">
  <img src="assets/head.png" width="92%" alt="Qwen-Image + LinCa，6.95× 加速">
</p>

## 支持的模型

| 模型 | 代码 | 已发布权重 | 说明 |
|---|---|---|---|
| FLUX.1-dev | [`models/flux`](models/flux) | 三阶段 predictor | 使用发布权重做推理 |
| Qwen-Image | [`models/qwen_image`](models/qwen_image) | 两阶段 checkpoint | 代码默认值与发布权重一致 |
| Qwen-Image-Edit | [`models/qwen_image_edit`](models/qwen_image_edit) | 两阶段 checkpoint | 代码默认值与发布权重一致 |
| HunyuanVideo | [`models/hunyuanvideo`](models/hunyuanvideo) | — | 需自行训练 |
| Wan2.1 | [`models/wan2.1`](models/wan2.1) | — | 需自行训练 |

Predictor 权重在 [QHRQQQ/LinCa](https://huggingface.co/QHRQQQ/LinCa)。底模权重不随本仓库分发，请从各官方渠道下载。

## 快速开始

先按对应底模的官方环境安装依赖，再加载 LinCa predictor。更完整的命令见 [README.md](README.md) 与 [`docs/inference.md`](docs/inference.md)。

### Qwen-Image

```bash
huggingface-cli download QHRQQQ/LinCa qwen-image_checkpoint.pt --local-dir ./checkpoints

cd models/qwen_image/qwen_image_linca_two_stage
python sample_learned.py \
  --checkpoint ../../../checkpoints/qwen-image_checkpoint.pt \
  --prompt_file prompts/DrawBench200.txt \
  --output_dir samples/qwen_image
```

### FLUX.1-dev

```bash
huggingface-cli download QHRQQQ/LinCa \
  flux/best_predictor_stage0.pt \
  flux/best_predictor_stage1.pt \
  flux/best_predictor_stage2.pt \
  --local-dir ./checkpoints

export FLUX_CHECKPOINT_DIR=/path/to/FLUX.1-dev
cd models/flux
bash run_sample.sh ../../checkpoints/flux 17,17,16
```

HunyuanVideo / Wan2.1 提供训练与采样脚本，但不发布已训练权重。

## 引用

```bibtex
@article{liu2026linca,
  title={LinCa: Accelerating Diffusion Models via Learnable Decomposed Feature Caching},
  author={Liu, Jinshan and Qin, Haoran and Tu, Xiaobing and Liu, Jiacheng and Hu, Jiahui and Yan, Zhengan and Xie, Yukun and Shen, Kerui and Ren, Jinkui and Lin, Yuqi and Zhang, Xiantao and Zhang, Linfeng},
  journal={arXiv preprint arXiv:2608.17973},
  year={2026}
}
```

联系：[qinhaoran68@gmail.com](mailto:qinhaoran68@gmail.com)
