"""
Compute FLOPs for every model at interval=3..12 and emit a Markdown table
"""
import torch
import torch.nn as nn
import sys
import math

sys.path.insert(0, 'src')
from flux.modules.invertible_net import InvertibleDecompositionNet

# ============ Constants ============
FLUX_FULL_STEP_TFLOPS = 74.39       # FLUX full step FLOPs (TFLOPs)
FREQCA_CACHE_STEP_TFLOPS = 0.02275  # Freqca cache step FLOPs (TFLOPs)
NUM_STEPS = 50
FIRST_ENHANCE = 3
SEQ_LEN = 4096  # 1024x1024 image tokens
DIM = 3072
INTERVALS = list(range(3, 13))  # 3 ~ 12


def count_steps(interval):
    """Count full-compute vs cached steps"""
    full = FIRST_ENHANCE + math.ceil((NUM_STEPS - FIRST_ENHANCE) / interval)
    cache = NUM_STEPS - full
    return full, cache


def measure_invertible_net_flops(num_blocks, hidden_dim, dim=DIM, seq_len=SEQ_LEN):
    """Measure InvertibleNet forward FLOPs with calflops"""
    from calflops import calculate_flops

    net = InvertibleDecompositionNet(
        dim=dim, num_blocks=num_blocks, hidden_dim=hidden_dim, dropout=0.0
    )
    net.eval().cuda()

    x = torch.randn(1, seq_len, dim, device='cuda', dtype=torch.float32)

    flops_str, macs_str, params_str = calculate_flops(
        model=net, args=[x], print_results=False
    )

    # InvertibleNet forward costs the same as inverse
    flops_val = parse_calflops_str(flops_str)

    # Read the parameter count
    total_params = sum(p.numel() for p in net.parameters())

    del net, x
    torch.cuda.empty_cache()

    return flops_val, total_params


def parse_calflops_str(s):
    """Parse a calflops string such as '644.295 GFLOPS' into TFLOPs"""
    parts = s.strip().split()
    value = float(parts[0])
    unit = parts[1].upper() if len(parts) > 1 else ''
    multipliers = {'K': 1e3, 'M': 1e6, 'G': 1e9, 'T': 1e12, 'P': 1e15}
    for prefix, mult in multipliers.items():
        if unit.startswith(prefix):
            return value * mult * 1e-12  # convert to TFLOPs
    return value * 1e-12


def generate_model_configs():
    """Enumerate every (num_blocks, hidden_dim) pair"""
    blocks_list = [1, 2, 3, 4, 5, 6, 7, 8]
    hidden_list = [64, 128, 256, 512, 1024, 2048]
    models = []
    for b in blocks_list:
        for h in hidden_list:
            models.append({
                'num_blocks': b,
                'hidden_dim': h,
            })
    return models


def main():
    models = generate_model_configs()

    print(f"Testing {len(models)} architecture combinations")

    # Group by (num_blocks, hidden_dim) and compute FLOPs
    arch_flops = {}  # (num_blocks, hidden_dim) -> (invertible_net_tflops, params)
    for m in models:
        key = (m['num_blocks'], m['hidden_dim'])
        if key not in arch_flops:
            print(f"Measuring FLOPs for blocks={key[0]}, hidden={key[1]}...")
            flops, params = measure_invertible_net_flops(key[0], key[1])
            arch_flops[key] = (flops, params)
            print(f"  -> InvertibleNet FLOPs = {flops:.6f} TFLOPs, Params = {params:,}")

    # ============ Emit Markdown ============
    md_lines = []
    md_lines.append("# FLOPs Calculation Results")
    md_lines.append("")
    md_lines.append("## Basic Constants")
    md_lines.append("")
    md_lines.append(f"- FLUX model: **flux-dev** (11.9B params)")
    md_lines.append(f"- Resolution: **1024 x 1024** (seq_len = {SEQ_LEN} tokens)")
    md_lines.append(f"- num_steps = **{NUM_STEPS}**, first_enhance = **{FIRST_ENHANCE}**")
    md_lines.append(f"- Full step cost (FLUX only): **{FLUX_FULL_STEP_TFLOPS} TFLOPs**")
    md_lines.append(f"- Freqca cache step cost: **{FREQCA_CACHE_STEP_TFLOPS} TFLOPs**")
    md_lines.append(f"- Baseline (no cache): **{NUM_STEPS * FLUX_FULL_STEP_TFLOPS:.2f} TFLOPs**")
    md_lines.append("")
    md_lines.append(f"- Step count formula: `full = first_enhance + ceil((num_steps - first_enhance) / interval)`")
    md_lines.append("")

    # Step count table
    md_lines.append("## Step Counts per Interval")
    md_lines.append("")
    md_lines.append("| interval | full steps | cache steps |")
    md_lines.append("|----------|-----------|------------|")
    for interval in INTERVALS:
        full, cache = count_steps(interval)
        md_lines.append(f"| {interval} | {full} | {cache} |")
    md_lines.append("")

    # Freqca baseline
    md_lines.append("## Freqca FLOPs (Baseline Method)")
    md_lines.append("")
    md_lines.append(f"Formula: `total = full_steps x {FLUX_FULL_STEP_TFLOPS} + cache_steps x {FREQCA_CACHE_STEP_TFLOPS}`")
    md_lines.append("")
    md_lines.append("| interval | full | cache | Total FLOPs (T) | Speedup |")
    md_lines.append("|----------|------|-------|----------------|---------|")
    baseline = NUM_STEPS * FLUX_FULL_STEP_TFLOPS
    for interval in INTERVALS:
        full, cache = count_steps(interval)
        total = full * FLUX_FULL_STEP_TFLOPS + cache * FREQCA_CACHE_STEP_TFLOPS
        speedup = baseline / total
        md_lines.append(f"| {interval} | {full} | {cache} | {total:.2f} | {speedup:.2f}x |")
    md_lines.append(f"| baseline | {NUM_STEPS} | 0 | {baseline:.2f} | 1.00x |")
    md_lines.append("")

    # Each model variant
    md_lines.append("## flux_v4 FLOPs (Ours - Learned Invertible Decomposition)")
    md_lines.append("")

    for m in sorted(models, key=lambda x: (x['num_blocks'], x['hidden_dim'])):
        key = (m['num_blocks'], m['hidden_dim'])
        inv_flops, inv_params = arch_flops[key]

        v4_full = FLUX_FULL_STEP_TFLOPS + inv_flops
        v4_cache = FREQCA_CACHE_STEP_TFLOPS + inv_flops

        md_lines.append(f"### blocks={m['num_blocks']}, hidden={m['hidden_dim']}")
        md_lines.append("")
        md_lines.append(f"- InvertibleNet FLOPs (forward = inverse): **{inv_flops:.4f} TFLOPs**")
        md_lines.append(f"- InvertibleNet Params: **{inv_params / 1e6:.2f} M**")
        md_lines.append(f"- Full step cost: {FLUX_FULL_STEP_TFLOPS} + {inv_flops:.4f} = **{v4_full:.4f} TFLOPs**")
        md_lines.append(f"- Cache step cost: {FREQCA_CACHE_STEP_TFLOPS} + {inv_flops:.4f} = **{v4_cache:.4f} TFLOPs**")
        md_lines.append("")
        md_lines.append("| interval | full | cache | Total FLOPs (T) | Speedup | vs Freqca |")
        md_lines.append("|----------|------|-------|----------------|---------|-----------|")

        for interval in INTERVALS:
            full, cache = count_steps(interval)
            total_v4 = full * v4_full + cache * v4_cache
            total_freqca = full * FLUX_FULL_STEP_TFLOPS + cache * FREQCA_CACHE_STEP_TFLOPS
            speedup = baseline / total_v4
            overhead = total_v4 - total_freqca
            md_lines.append(
                f"| {interval} | {full} | {cache} | {total_v4:.2f} | {speedup:.2f}x | +{overhead:.2f}T |"
            )

        md_lines.append(f"| baseline | {NUM_STEPS} | 0 | {baseline:.2f} | 1.00x | - |")
        md_lines.append("")

    # Summary comparison table
    md_lines.append("## Summary Comparison (All Models, interval=3/7/10)")
    md_lines.append("")
    header = "| Method | Params (M) | InvNet (T)"
    for iv in [3, 7, 10]:
        header += f" | interval={iv} (T)"
    header += " |"
    md_lines.append(header)
    sep = "|--------|-----------|----------"
    for _ in [3, 7, 10]:
        sep += "|------------------"
    sep += "|"
    md_lines.append(sep)

    # Freqca row
    row = f"| Freqca | - | -"
    for iv in [3, 7, 10]:
        full, cache = count_steps(iv)
        total = full * FLUX_FULL_STEP_TFLOPS + cache * FREQCA_CACHE_STEP_TFLOPS
        row += f" | {total:.2f}"
    row += " |"
    md_lines.append(row)

    # Each model
    seen_keys = set()
    for m in sorted(models, key=lambda x: (x['num_blocks'], x['hidden_dim'])):
        key = (m['num_blocks'], m['hidden_dim'])
        if key in seen_keys:
            continue
        seen_keys.add(key)
        inv_flops, inv_params = arch_flops[key]
        v4_full = FLUX_FULL_STEP_TFLOPS + inv_flops
        v4_cache = FREQCA_CACHE_STEP_TFLOPS + inv_flops
        row = f"| Ours (b={key[0]},h={key[1]}) | {inv_params/1e6:.2f} | {inv_flops:.4f}"
        for iv in [3, 7, 10]:
            full, cache = count_steps(iv)
            total = full * v4_full + cache * v4_cache
            row += f" | {total:.2f}"
        row += " |"
        md_lines.append(row)

    # Baseline
    row = f"| No Cache | - | -"
    for _ in [3, 7, 10]:
        row += f" | {baseline:.2f}"
    row += " |"
    md_lines.append(row)
    md_lines.append("")

    # Write file
    md_content = '\n'.join(md_lines)
    output_path = './flops_results.md'
    with open(output_path, 'w') as f:
        f.write(md_content)

    print(f"\nResults saved to {output_path}")
    print("\n" + md_content)


if __name__ == '__main__':
    main()
