# LinCa ⚡️

**Accelerating Diffusion Models via Learnable Decomposed Feature Caching**

Welcome to the official (anonymous) repository for **LinCa**.

## 📢 News
* **[2026-06]** Repository created!The full PyTorch codebase, including training scripts and pre-trained predictors, will be officially released here after the unblinding process. Stay tuned!

## 📖 About
Diffusion models have achieved remarkable success, but the high computational cost of iterative sampling remains a bottleneck. **LinCa** is a novel feature caching framework based on learnable invertible networks designed to accelerate diffusion models. 

Unlike training-free methods that apply uniform prediction strategies, LinCa decomposes cached features into sub-components with distinct continuity properties via a lightweight invertible network. It forms a unified **Decompose-Predict-Reconstruct** pipeline with strict invertibility guarantees for lossless reconstruction.

### 🔥 Key Features:
- **High Acceleration Ratio**: Maintains near-lossless generation quality at **5-7× speedup**.
- **Lightweight**: Introduces less than **0.2%** additional parameters.
- **Broad Compatibility**: Demonstrated effectiveness on state-of-the-art models including **FLUX**, **Qwen-Image**, and **HunyuanVideo**.

## 🚀 Todo / Roadmap
- [ ] Release the core PyTorch implementation of the Invertible Network and Predictors
- [ ] Release inference scripts for FLUX, Qwen-Image, and HunyuanVideo
- [ ] Release pre-trained predictor weights
