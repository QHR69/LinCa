# LinCa

**通过可学习分解特征缓存加速扩散模型**

<p align="center">
  <a href="https://arxiv.org/abs/2608.17973"><img src="https://img.shields.io/badge/arXiv-2608.17973-b31b1b.svg" alt="arXiv"></a>
  <a href="https://github.com/QHR69/LinCa"><img src="https://img.shields.io/badge/GitHub-QHR69/LinCa-181717?logo=github" alt="GitHub"></a>
  <a href="https://huggingface.co/QHRQQQ/LinCa"><img src="https://img.shields.io/badge/HuggingFace-Weights-FFD21E?logo=huggingface" alt="Weights"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"></a>
  <a href="README.md"><img src="https://img.shields.io/badge/README-English-blue.svg" alt="English"></a>
</p>

<p align="center">
<a href="https://arxiv.org/abs/2608.17973">论文</a> ·
<a href="https://github.com/QHR69/LinCa">代码</a> ·
<a href="https://huggingface.co/QHRQQQ/LinCa">权重</a> ·
<a href="README.md">English</a>
</p>

LinCa 用轻量可逆网络加速扩散推理：把缓存特征**分解**成连续性不同的子分量，按分量匹配预测阶数，再**重建**回原特征空间。映射严格可逆，重建无损。

**额外参数 &lt;0.2%。加速 5–7×。近无损质量。** 下方 teaser 为 Qwen-Image + LinCa，加速比 **6.95×**。

<p align="center">
  <img src="assets/fig1_teaser.png" width="100%" alt="Qwen-Image + LinCa，6.95× 加速">
</p>
<p align="center"><em>图 1. Qwen-Image + LinCa 在 6.95× 加速下的生成结果。</em></p>

## 更新

- **2026-08** — 代码、论文与 predictor 权重公开。FLUX / Qwen-Image / Qwen-Image-Edit 权重见 [Hugging Face](https://huggingface.co/QHRQQQ/LinCa)。

## 核心结果

| 设置 | FLOPs 加速 | 质量 | 相对原始 50 步 |
|---|---:|---|---|
| **FLUX.1-dev** `N=6` | **4.52×** | ImageReward **1.0228** | **+3.0%**（原始 0.9930） |
| **FLUX.1-dev** `N=8` | **5.51×** | ImageReward **1.0162** | **+2.3%** |
| **Qwen-Image** `N=10` | **6.95×** | ImageReward **1.0524** · CLIP **35.17** | 缓存方法中最优 |
| **HunyuanVideo** `N=6` | **5.50×** | VBench **80.16** | **−0.6%**（原始 80.66） |
| **Qwen-Image-Edit** `N=7` | **5.52×** | GEdit-EN OS **7.56** | 持平 / 超过原始 |
| **Wan2.1-1.3B** `N=6` | **5.00×** | VBench **82.56** | 高于 TaylorSeer 81.07 @ 4.17× |

同样适用于 **FLUX.1-lite-8B**、**FLUX.1-schnell**、**FLUX.1-dev-int8**、**Qwen-Image-Lightning**。用 100–200 条缓存特征，单卡 **12 GB / 约 1 小时**即可训完 predictor，无需加载底模。

## 为什么现有缓存会掉点

训练无关的缓存方法对所有模型、所有时间步、所有特征维度用**同一套**预测规则。但连续性并不均匀：有的维度突变，有的平滑，而且这种混合会随模型和去噪阶段变化。

<p align="center">
  <img src="assets/fig2_dynamics.png" width="86%" alt="特征动态失配的 PCA 分析">
</p>
<p align="center"><em>图 2. 模型、阶段、维度之间的动态失配。(b) 不稳定维度；(c) 稳定维度。</em></p>

## 方法

LinCa 是 **分解 → 预测 → 重建** 流水线：

1. **分解**：可逆网络把缓存的 DiT 特征映射到连续性不同的分区。
2. **预测**：0 阶直接复用最近缓存；更高阶分区用匹配的多项式预测。
3. **重建**：逆映射回到原空间。每个 block 是可逆 1×1 卷积 + 加性耦合，满足 \(\mathcal{E}_\theta^{-1}\circ\mathcal{E}_\theta = I\)。

不同模型和不同时间段可以各自训练 predictor。

<p align="center">
  <img src="assets/fig3_method.png" width="100%" alt="LinCa 框架总览">
</p>
<p align="center"><em>图 3. LinCa 总览：特征缓存、分解–预测–重建流水线、可逆 block。</em></p>

## 定性结果

**FLUX.1-dev**：更快，同时保住文字、细部和空间关系，其它缓存方法会丢。

<p align="center">
  <img src="assets/fig4_flux.png" width="100%" alt="FLUX.1-dev 定性对比">
</p>

**Qwen-Image**：加速比升高时质量仍稳；TaylorSeer 发糊、丢细节。

<p align="center">
  <img src="assets/fig5_qwen.png" width="100%" alt="Qwen-Image 定性对比">
</p>

**HunyuanVideo**：更高加速下仍保住背景、布局和运动；22% steps / TaylorSeer / TeaCache 会丢。

<p align="center">
  <img src="assets/fig6_hunyuan.png" width="100%" alt="HunyuanVideo 定性对比">
</p>

**Qwen-Image-Edit**：指令理解更稳，未编辑区域保持一致。

<p align="center">
  <img src="assets/fig7_edit.png" width="100%" alt="Qwen-Image-Edit 定性对比">
</p>

## 定量结果

下列表格均从论文 PDF 精裁。

### FLUX.1-dev

`N=4 / 6 / 8` 均在速度–质量曲线上最优。`N=6` 时 FLOPs **4.52×**，ImageReward **1.0228（+3.0%）**。同等预算下 TeaCache / DBCache 掉点明显。

<p align="center">
  <img src="assets/tab1_flux.png" width="100%" alt="FLUX.1-dev 定量表">
</p>

### Qwen-Image

`N=10` 达到 FLOPs **6.95×**，ImageReward / CLIP / PSNR / SSIM / LPIPS 均为缓存方法最优。FORA / ToCa / DuCa / TaylorSeer 在此加速比下掉点严重。

<p align="center">
  <img src="assets/tab2_qwen.png" width="100%" alt="Qwen-Image 定量表">
</p>

### HunyuanVideo

`N=6` → FLOPs **5.50×**，VBench **80.16**（原始 80.66）。优于 FORA、ToCa、DuCa、TeaCache、TaylorSeer、Speca、Clusca、FoCa。

<p align="center">
  <img src="assets/tab3_hunyuan.png" width="92%" alt="HunyuanVideo 定量表">
</p>

### Qwen-Image-Edit（GEdit-Bench）

`N=7` → FLOPs **5.52×**，GEdit-EN OS **7.56**（原始 7.54）。`N=10` → FLOPs **7.08×**，OS **7.40**；DuCa / TaylorSeer 降到约 6.3。

<p align="center">
  <img src="assets/tab4_edit.png" width="100%" alt="Qwen-Image-Edit 定量表">
</p>

### 蒸馏 / 量化 FLUX

| 底模 | 步数 | FLOPs 加速 | ImageReward | CLIP |
|---|---:|---:|---:|---:|
| FLUX.1-lite-8B | 28 → LinCa `N=3` | **2.32×** | 0.8936 → **0.9070** | 32.12 → **32.36** |
| FLUX.1-schnell | 4 → LinCa `N=3` | **1.99×** | 0.9692 → **0.9843** | 32.54 → **32.67** |
| FLUX.1-dev-int8 | 50 → LinCa `N=3` | **2.63×** | 0.9744 → **1.0036** | 32.55 → **32.81** |

<p align="center">
  <img src="assets/tab5_distill.png" width="92%" alt="蒸馏与量化 FLUX 表">
</p>

### 对比蒸馏模型与训练式加速

FLUX：14 步 LinCa（**3.57×**，IR **1.02**）优于 FLUX.1-lite-8B（28 步，1.79×，IR 0.89），8–10 步也优于 LESA。

<p align="center">
  <img src="assets/tab7_more_flux.png" width="62%" alt="更多 FLUX 对比">
</p>

Qwen-Image：14 步 LinCa（**3.57×**，IR **1.19**）优于 Distill-Full / LoRA（15 步，IR 1.02 / 0.95），在 5.56× 与 7.14× 上也优于 LESA。

<p align="center">
  <img src="assets/tab8_more_qwen.png" width="62%" alt="更多 Qwen-Image 对比">
</p>

### 少步蒸馏与 Wan2.1

Qwen-Image-Lightning（8 步）+ LinCa `N=3` → **2.00×**，ImageReward 1.26（原始 1.28），GenEval **0.85**（原始 0.84）。

<p align="center">
  <img src="assets/tab9_lightning.png" width="58%" alt="Qwen-Image-Lightning 兼容性">
</p>

Wan2.1-1.3B：LinCa `N=6` → **5.00×**，VBench **82.56**；TaylorSeer 为 4.17× / 81.07。

<p align="center">
  <img src="assets/tab10_wan.png" width="58%" alt="Wan2.1 定量表">
</p>

## 分析

### 消融

可训练的可逆网络在 FLUX 与 Qwen-Image 上优于 MLP 和未训练可逆网络。差分多阶 `0+1+2` 随 `N` 增大仍优于任一单阶。

<p align="center">
  <img src="assets/fig8_ablation.png" width="96%" alt="结构与预测阶数消融">
</p>

### 预测误差

缓存间隔变大时，LinCa 的特征 MSE 仍低。FORA 稳步上升；TaylorSeer 在 `N=6` 后急剧恶化。

<p align="center">
  <img src="assets/fig9_mse.png" width="62%" alt="特征预测 MSE">
</p>

### 超参

论文扫描：时间段 `S=3`、损失权重 `λ=1`、结构 `L=2`、`h=128` 最强。`S` 再加大收益很小；`λ` 过小或过大都会伤质量。

<p align="center">
  <img src="assets/fig10_hyper.png" width="96%" alt="超参敏感性">
</p>

### 高阶预测器

同等 **5.51×** 下 Hermite 最好（IR **1.0162**，CLIP **32.72**），优于 Lagrange / Taylor / Chebyshev / Laguerre。对完整特征做均匀 Taylor（TaylorSeer）即使在 4.16× 也更弱。

<p align="center">
  <img src="assets/tab6_predictor.png" width="72%" alt="高阶预测器对比">
</p>

发布的推理脚本请保持**与已发布权重一致的代码默认值**，不要为了对齐论文扫描而改默认超参。

## 支持的模型

| 模型 | 代码 | 已发布权重 | 说明 |
|---|---|---|---|
| FLUX.1-dev | [`models/flux`](models/flux) | 三阶段 predictor | 使用发布权重推理 |
| Qwen-Image | [`models/qwen_image`](models/qwen_image) | 两阶段 checkpoint | 默认值与发布权重一致 |
| Qwen-Image-Edit | [`models/qwen_image_edit`](models/qwen_image_edit) | 两阶段 checkpoint | 默认值与发布权重一致 |
| HunyuanVideo | [`models/hunyuanvideo`](models/hunyuanvideo) | — | 需自行训练 |
| Wan2.1 | [`models/wan2.1`](models/wan2.1) | — | 需自行训练 |

Predictor 权重：[QHRQQQ/LinCa](https://huggingface.co/QHRQQQ/LinCa)。底模请从各官方渠道下载。

## 快速开始

先按对应底模安装官方环境，再加载 LinCa predictor。更完整参数见 [`docs/inference.md`](docs/inference.md)。

### Qwen-Image

```bash
huggingface-cli download QHRQQQ/LinCa qwen-image_checkpoint.pt --local-dir ./checkpoints

cd models/qwen_image/qwen_image_linca_two_stage
python sample_learned.py \
  --checkpoint ../../../checkpoints/qwen-image_checkpoint.pt \
  --prompt_file prompts/DrawBench200.txt \
  --output_dir samples/qwen_image
```

### Qwen-Image-Edit

```bash
huggingface-cli download QHRQQQ/LinCa qwen-image-edit_checkpoint.pt --local-dir ./checkpoints

cd models/qwen_image_edit/qwen_image_edit_linca_two_stage
python sample_edit_single.py \
  --checkpoint ../../../checkpoints/qwen-image-edit_checkpoint.pt \
  --image path/to/input.png \
  --prompt "your edit instruction"
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

### HunyuanVideo / Wan2.1

各目录含训练与采样脚本，但不发布已训练权重。

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
