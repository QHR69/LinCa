#!/bin/bash
# Multi-GPU cache generation for qwen_edit (202 samples)
# seed = base_seed + original_idx, ensures identical results across single/multi-GPU
#
# Usage:
#   # 4 GPUs (need qwen_image conda env)
#   conda activate qwen_image
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_generate_cache_ddp.sh
#
#   # 2 GPUs
#   CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 generate_cache_data_edit.py --ddp

cd "$(dirname "$0")"

# Activate qwen_image env if conda available
if command -v conda &>/dev/null; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate qwen_image 2>/dev/null || true
fi

# Default: use all visible GPUs
NPROC=${NPROC:-$(nvidia-smi -L 2>/dev/null | wc -l)}
NPROC=${NPROC:-4}

echo "========================================"
echo "qwen_edit cache generation (202 samples)"
echo "  Cache: ./cache_data/qwen_edit"
echo "  Images: ./ (current dir)"
echo "  GPUs: $NPROC"
echo "  seed = base_seed + original_idx"
echo "========================================"

torchrun --nproc_per_node=$NPROC generate_cache_data_edit.py --ddp

echo ""
echo "Done. Verify with: python verify_cache_sample.py 0"
