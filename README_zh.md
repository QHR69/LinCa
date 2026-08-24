# LinCa

**通过可学习分解特征缓存加速扩散模型**

<p align="center">
  <a href="https://arxiv.org/abs/2608.17973"><img src="https://img.shields.io/badge/ECCV%202026-Accepted-8A2BE2.svg" alt="ECCV 2026"></a>
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

<p align="center"><b>ECCV 2026 中稿论文官方实现。</b></p>

LinCa 用轻量可逆网络加速扩散推理：把缓存特征**分解**成连续性不同的子分量，按分量匹配预测阶数，再**重建**回原特征空间。映射严格可逆，重建无损。

**额外参数 &lt;0.2%。FLOPs 最高降低 7.08×。近无损质量。**

<p align="center">
  <img src="assets/fig1_teaser.png" width="100%" alt="Qwen-Image + LinCa，6.95x 加速">
</p>
<p align="center"><em>图 1. Qwen-Image + LinCa 在 6.95× 加速下的生成结果。</em></p>

> **本文档中所有加速比均为 FLOPs 加速比**，相对原始全步数模型，数据直接取自论文表格。FLOPs 与硬件无关；实测延迟见下方论文表格截图。

## 更新

- **2026-08** — **论文被 ECCV 2026 接收。** 代码、论文与 predictor 权重公开。FLUX / Qwen-Image / Qwen-Image-Edit 权重见 [Hugging Face](https://huggingface.co/QHRQQQ/LinCa)。

## 核心结果

各模型上质量保持最好的工作点：

| 模型 | 设置 | FLOPs 加速 | 质量 | 相对原始 |
|---|---|---:|---|---|
| **FLUX.1-dev** | `N=6` | **4.52×** | ImageReward **1.0228** · CLIP **32.97** | IR **+3.0%**（原始 0.9930） |
| **FLUX.1-dev** | `N=8` | **5.51×** | ImageReward **1.0162** · CLIP 32.72 | IR **+2.3%** |
| **Qwen-Image** | `N=6` | **4.90×** | ImageReward **1.2163** · CLIP **35.34** · PSNR **29.84** | IR **−2.9%**（原始 1.2532） |
| **Qwen-Image-Edit** | `N=7` | **5.52×** | GEdit-EN OS **7.56** · CN OS **7.45** | **超过**原始（7.54 / 7.41） |
| **HunyuanVideo** | `N=6` | **5.50×** | VBench **80.16** | **−0.6%**（原始 80.66） |
| **Wan2.1-1.3B** | `N=6` | 5.00× ¹ | VBench **82.56** | 高于 TaylorSeer 81.07 @ 4.17× |

¹ 论文表 10 对 Wan2.1 给出的是步数加速比，该实验未列 FLOPs。

继续拉高：**Qwen-Image 在 6.95×** 仍保持 ImageReward 1.0524、CLIP 35.17；**Qwen-Image-Edit 在 7.08×** 仍保持 GEdit-EN OS 7.40 —— 在这些加速比下其它缓存方法已经崩掉。

LinCa 还能叠加在 **FLUX.1-lite-8B**、**FLUX.1-schnell**、**FLUX.1-dev-int8**、**Qwen-Image-Lightning** 之上。用 100–200 条缓存特征，单卡 **12 GB / 约 1 小时**即可训完 predictor，无需加载底模。

## 为什么现有缓存会掉点

训练无关的缓存方法对所有模型、所有时间步、所有特征维度用**同一套**预测规则。但连续性并不均匀：有的维度突变，有的平滑，而且这种混合会随模型和去噪阶段变化。

<p align="center">
  <img src="assets/fig2_dynamics.png" width="86%" alt="特征动态失配的 PCA 分析">
</p>
<p align="center"><em>图 2. 模型、阶段、维度之间的动态失配。(b) 不稳定维度；(c) 稳定维度。</em></p>

## 方法

LinCa 是 **分解 → 预测 → 重建** 流水线：

1. **分解**：可学习可逆网络把缓存的 DiT 特征映射到连续性不同的分区。
2. **预测**：0 阶直接复用最近缓存；更高阶分区用匹配阶数的多项式预测。
3. **重建**：逆映射回到原空间。每个 block 是可逆 1×1 卷积 + 加性耦合层，网络严格可逆，不引入重建误差。

不同模型、不同时间段各自训练 predictor，同一套框架就能适配异质的特征动态，代价极低。

<p align="center">
  <img src="assets/fig3_method.png" width="100%" alt="LinCa 框架总览">
</p>
<p align="center"><em>图 3. LinCa 总览：特征缓存、分解–预测–重建流水线、可逆 block。</em></p>

## 实验结果

### 文生图：FLUX.1-dev

在 **FLOPs 降低 4.52×** 时，LinCa 反而**超过**原始 50 步（ImageReward +3.0%，CLIP +1.1%）。其它缓存方法要么牺牲质量，要么牺牲速度。

| 方法 | FLOPs 加速 | ImageReward ↑ | CLIP ↑ |
|---|---:|---:|---:|
| 原始 50 步 | 1.00× | 0.9930 | 32.61 |
| TaylorSeer `N=4` | 3.84× | 0.9887 | 32.58 |
| DBCache `F=4,B=4` | 4.10× | 0.6372 | 32.10 |
| HyCa `N=5` | 4.16× | 1.0096 | 32.87 |
| TeaCache `l=0.8` | 4.17× | 0.8710 | 31.89 |
| **LinCa `N=6`** | **4.52×** | **1.0228** | **32.97** |
| FORA `N=5` | 4.16× | 0.8276 | 31.95 |
| DuCa `N=7` | 4.89× | 0.9785 | 32.26 |
| HyCa `N=6` | 5.00× | 1.0043 | 32.64 |
| TeaCache `l=1` | 5.01× | 0.8421 | 32.01 |
| **LinCa `N=8`** | **5.51×** | **1.0162** | 32.72 |

<p align="center">
  <img src="assets/tab1_flux.png" width="100%" alt="FLUX.1-dev 定量表">
</p>

LinCa 更快，同时保住文字、细部和空间关系，其它缓存方法会丢。

<p align="center">
  <img src="assets/fig4_flux.png" width="100%" alt="FLUX.1-dev 定性对比">
</p>

### 文生图：Qwen-Image

优势最明显的一组。**4.90×** 时 LinCa 的 ImageReward 只掉 **2.9%**，而 TaylorSeer 在接近的 5.00× 上掉 **19.2%**。LinCa 在质量指标和感知指标上同时全面领先。

| 方法 | FLOPs 加速 | ImageReward ↑ | CLIP ↑ | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|---:|---:|---:|
| 原始 50 步 | 1.00× | 1.2532 | 35.52 | ∞ | 1.00 | 0.00 |
| FORA `N=4` | 3.84× | 0.9315 | 34.33 | 28.65 | 0.60 | 0.50 |
| ToCa `N=8, R=75%` | 4.32× | 1.0218 | 34.97 | 28.94 | 0.63 | 0.45 |
| DuCa `N=9, R=80%` | 4.37× | 0.7709 | 34.59 | 28.44 | 0.58 | 0.54 |
| TaylorSeer `N=6` | 5.00× | 1.0125 | 34.77 | 28.59 | 0.62 | 0.45 |
| HyCa `N=6` | 5.00× | 1.1967 | 34.89 | 29.68 | 0.71 | 0.31 |
| **LinCa `N=6`** | **4.90×** | **1.2163** | **35.34** | **29.84** | **0.73** | **0.29** |
| TaylorSeer `N=9` | 6.25× | 0.7318 | 32.89 | 28.24 | 0.56 | 0.57 |
| HyCa `N=8` | 6.25× | 1.0432 | 34.82 | 28.86 | 0.62 | 0.44 |
| **LinCa `N=10`** | **6.95×** | **1.0524** | **35.17** | **29.10** | **0.69** | **0.39** |

<p align="center">
  <img src="assets/tab2_qwen.png" width="100%" alt="Qwen-Image 定量表">
</p>

加速比升高时质量仍稳；TaylorSeer 发糊、丢细节。

<p align="center">
  <img src="assets/fig5_qwen.png" width="100%" alt="Qwen-Image 定性对比">
</p>

### 图像编辑：Qwen-Image-Edit（GEdit-Bench）

**5.52×** 时 LinCa 在中英文总分上都**超过原始模型**。**7.08×** 时仍保持 7.40 / 7.27，而 DuCa 和 TaylorSeer 已降到约 6.3。

| 方法 | FLOPs 加速 | GEdit-CN OS ↑ | GEdit-EN OS ↑ |
|---|---:|---:|---:|
| 原始 50 步 | 1.00× | 7.41 | 7.54 |
| FORA `N=5` | 5.00× | 7.25 | 7.28 |
| TaylorSeer `N=6` | 5.00× | 6.92 | 6.89 |
| DuCa `N=7, R=95%` | 4.95× | 7.44 | 7.45 |
| **LinCa `N=7`** | **5.52×** | **7.45** | **7.56** |
| DuCa `N=10, R=95%` | 5.46× | 6.39 | 6.41 |
| FORA `N=7` | 6.24× | 7.06 | 7.06 |
| TaylorSeer `N=9` | 6.24× | 6.31 | 6.31 |
| **LinCa `N=10`** | **7.08×** | **7.27** | **7.40** |

<p align="center">
  <img src="assets/tab4_edit.png" width="100%" alt="Qwen-Image-Edit 定量表">
</p>

指令理解更稳，未编辑区域保持一致。

<p align="center">
  <img src="assets/fig7_edit.png" width="100%" alt="Qwen-Image-Edit 定性对比">
</p>

### 文生视频：HunyuanVideo

加速比**和** VBench 分数在对比中同时最高，仅比原始 50 步低 0.6%。

| 方法 | FLOPs 加速 | VBench ↑ |
|---|---:|---:|
| 原始 50 步 | 1.00× | 80.66 |
| ToCa `N=5, R=90%` | 4.25× | 78.86 |
| DuCa `N=5, R=90%` | 4.48× | 78.72 |
| TeaCache `l=0.4` | 4.55× | 79.36 |
| FoCa `N=5` | 4.99× | 79.96 |
| Clusca `N=5, K=32` | 4.99× | 79.99 |
| TaylorSeer `N=5` | 5.00× | 79.93 |
| Speca | 5.23× | 79.98 |
| **LinCa `N=6`** | **5.50×** | **80.16** |

<p align="center">
  <img src="assets/tab3_hunyuan.png" width="92%" alt="HunyuanVideo 定量表">
</p>

更高加速下仍保住背景、布局和运动；22% steps / TaylorSeer / TeaCache 会丢。

<p align="center">
  <img src="assets/fig6_hunyuan.png" width="100%" alt="HunyuanVideo 定性对比">
</p>

### 文生视频：Wan2.1-1.3B

| 方法 | 步数 | 加速比 ¹ | VBench ↑ |
|---|---:|---:|---:|
| 原始 | 50 | 1.00× | 83.72 |
| TaylorSeer `N=5` | 12 | 4.17× | 81.07 |
| **LinCa `N=6`** | 10 | **5.00×** | **82.56** |

<p align="center">
  <img src="assets/tab10_wan.png" width="58%" alt="Wan2.1 定量表">
</p>

### 与蒸馏 / 量化叠加

LinCa 与模型压缩互补：在蒸馏和 INT8 底模上继续加速，**同时提升质量**。

| 底模 | 设置 | FLOPs 加速 | ImageReward ↑ | CLIP ↑ |
|---|---|---:|---:|---:|
| FLUX.1-lite-8B，28 步 | + LinCa `N=3` | **2.32×** | 0.8936 → **0.9070** | 32.12 → **32.36** |
| FLUX.1-schnell，4 步 | + LinCa `N=3` | **1.99×** | 0.9692 → **0.9843** | 32.54 → **32.67** |
| FLUX.1-dev-int8，50 步 | + LinCa `N=3` | **2.63×** | 0.9744 → **1.0036** | 32.55 → **32.81** |

<p align="center">
  <img src="assets/tab5_distill.png" width="92%" alt="蒸馏与量化 FLUX 表">
</p>

### 对比蒸馏模型与训练式加速

论文表 7–9 给出的是步数加速比 ¹。FLUX 上，14 步的 LinCa 在速度和质量上都优于 28 步的 FLUX.1-lite-8B；Qwen-Image 上在每个对齐的加速比下都优于官方蒸馏模型和 LESA。

| 模型 | 方法 | 步数 | 加速比 ¹ | ImageReward ↑ |
|---|---|---:|---:|---:|
| FLUX.1-dev | 原始 | 50 | 1.00× | 0.99 |
| FLUX.1-dev | FLUX.1-lite-8B | 28 | 1.79× | 0.89 |
| FLUX.1-dev | **LinCa `N=4`** | 14 | **3.57×** | **1.02** |
| FLUX.1-dev | LESA `N=7` | 9 | 5.56× | 0.98 |
| FLUX.1-dev | **LinCa `N=8`** | 8 | **6.25×** | **1.02** |
| Qwen-Image | 原始 | 50 | 1.00× | 1.25 |
| Qwen-Image | Qwen-Image-Distill-Full | 15 | 3.33× | 1.02 |
| Qwen-Image | Qwen-Image-Distill-LoRA | 15 | 3.33× | 0.95 |
| Qwen-Image | **LinCa `N=4`** | 14 | **3.57×** | **1.19** |
| Qwen-Image | LESA `N=10` | 7 | 7.14× | 1.01 |
| Qwen-Image | **LinCa `N=10`** | 7 | **7.14×** | **1.05** |

<p align="center">
  <img src="assets/tab7_more_flux.png" width="49%" alt="更多 FLUX 对比">
  <img src="assets/tab8_more_qwen.png" width="49%" alt="更多 Qwen-Image 对比">
</p>

即便在 8 步的少步模型上，LinCa 仍能再加速 **2.00×**，ImageReward 基本持平且 GenEval 提升（0.84 → **0.85**）。

<p align="center">
  <img src="assets/tab9_lightning.png" width="58%" alt="Qwen-Image-Lightning 兼容性">
</p>

## 分析

### 消融

可训练的可逆网络在 FLUX 与 Qwen-Image 上优于 MLP 和未训练可逆网络。差分多阶 `0+1+2` 随 `N` 增大仍优于任一单阶。

<p align="center">
  <img src="assets/fig8_ablation.png" width="96%" alt="结构与预测阶数消融">
</p>

### 预测误差

缓存间隔变大时，LinCa 的特征 MSE 仍低。FORA 稳步上升；TaylorSeer 在 `N=6` 后急剧恶化。这正是高加速比下质量差距的根因。

<p align="center">
  <img src="assets/fig9_mse.png" width="72%" alt="特征预测 MSE">
</p>

### 超参

论文扫描：时间段 `S=3`、损失权重 `λ=1`、结构 `L=2`、`h=128` 最强。`S` 再加大收益很小；`λ` 过小或过大都会伤质量。

<p align="center">
  <img src="assets/fig10_hyper.png" width="96%" alt="超参敏感性">
</p>

### 高阶预测器

同等 **5.51×** 下 Hermite 最好，优于 Lagrange / Taylor / Chebyshev / Laguerre。对完整特征做均匀 Taylor（TaylorSeer）即使在 4.16× 也更弱。

| 方法 | 预测器 | FLOPs 加速 | ImageReward ↑ | CLIP ↑ |
|---|---|---:|---:|---:|
| TaylorSeer | Taylor | 4.16× | 0.9793 | 32.63 |
| **LinCa** | **Hermite** | 5.51× | **1.0162** | **32.72** |
| LinCa | Lagrange | 5.51× | 1.0144 | 32.60 |
| LinCa | Taylor | 5.51× | 1.0139 | 32.58 |
| LinCa | Chebyshev | 5.51× | 1.0023 | 32.60 |
| LinCa | Laguerre | 5.51× | 0.9264 | 32.50 |

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
@inproceedings{liu2026linca,
  title={LinCa: Accelerating Diffusion Models via Learnable Decomposed Feature Caching},
  author={Liu, Jinshan and Qin, Haoran and Tu, Xiaobing and Liu, Jiacheng and Hu, Jiahui and Yan, Zhengan and Xie, Yukun and Shen, Kerui and Ren, Jinkui and Lin, Yuqi and Zhang, Xiantao and Zhang, Linfeng},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2026}
}
```

## 许可

LinCa 代码使用 [Apache-2.0](LICENSE)。上游底模保持各自许可，见 [NOTICE](NOTICE)。

联系：[qinhaoran68@gmail.com](mailto:qinhaoran68@gmail.com)
