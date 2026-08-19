"""
HunyuanVideo FLOPs calculator

Contains:
1. HunyuanVideo transformer FLOPs (full step / Taylor cache step)
2. Encoder/Decoder (InvertibleDecompositionNet) FLOPs
3. Total FLOPs and speedup for (b, hidden_dim, interval) combinations

Model layout:
- hidden_size = 3072, heads = 24, head_dim = 128
- mlp_ratio = 4 → mlp_hidden_dim = 12288
- 20 double-stream blocks + 40 single-stream blocks
- Input: 480x640, 65 frames -> img_seq_len = 20400, txt_seq_len = 256
- guidance-distilled: batch_size = 1 (no CFG)

Formulae:
  Linear(in, out) applied to [B, L, in]: FLOPs = 2 * B * L * in * out
  Attention(L, head_dim, heads): FLOPs = 2 * heads * L^2 * head_dim * 2
  LayerNorm/RMSNorm: ignored
  Element-wise (add/mul/gate): ignored

Step count formula:
  full = first_enhance(=3) + ceil((num_steps - first_enhance) / interval)
  cache = num_steps - full
"""

import math


def calc_hunyuan_flops(
    img_seq_len=20400,
    txt_seq_len=256,
    hidden_size=3072,
    heads_num=24,
    mlp_ratio=4,
    num_double_blocks=20,
    num_single_blocks=40,
    text_states_dim=4096,
    text_states_dim_2=768,
    max_order=1,
):
    head_dim = hidden_size // heads_num  # 128
    mlp_hidden_dim = hidden_size * mlp_ratio  # 12288
    total_seq = img_seq_len + txt_seq_len  # 20656

    def linear_flops(seq_len, in_dim, out_dim):
        return 2 * seq_len * in_dim * out_dim

    def attention_flops(seq_len):
        return 4 * heads_num * seq_len * seq_len * head_dim

    # ---- Fixed overhead (embedding, modulation, final_layer) ----
    fixed_flops = linear_flops(1, 256, hidden_size) + linear_flops(1, hidden_size, hidden_size)
    fixed_flops += linear_flops(1, text_states_dim_2, hidden_size) + linear_flops(1, hidden_size, hidden_size)
    fixed_flops += linear_flops(1, 256, hidden_size) + linear_flops(1, hidden_size, hidden_size)
    fixed_flops += linear_flops(img_seq_len, 16 * 4, hidden_size)
    fixed_flops += linear_flops(txt_seq_len, text_states_dim, hidden_size)
    fixed_flops += linear_flops(img_seq_len, hidden_size, hidden_size)
    fixed_flops += linear_flops(img_seq_len, hidden_size, 16 * 4)

    # ---- Double-stream block - FULL ----
    double_full = 0
    double_full += linear_flops(1, hidden_size, hidden_size * 6)
    double_full += linear_flops(1, hidden_size, hidden_size * 6)
    double_full += linear_flops(img_seq_len, hidden_size, hidden_size * 3)
    double_full += linear_flops(txt_seq_len, hidden_size, hidden_size * 3)
    double_full += attention_flops(total_seq)
    double_full += linear_flops(img_seq_len, hidden_size, hidden_size)
    double_full += linear_flops(txt_seq_len, hidden_size, hidden_size)
    double_full += linear_flops(img_seq_len, hidden_size, mlp_hidden_dim)
    double_full += linear_flops(img_seq_len, mlp_hidden_dim, hidden_size)
    double_full += linear_flops(txt_seq_len, hidden_size, mlp_hidden_dim)
    double_full += linear_flops(txt_seq_len, mlp_hidden_dim, hidden_size)

    # ---- Single-stream block - FULL ----
    single_full = 0
    single_full += linear_flops(1, hidden_size, hidden_size * 3)
    single_full += linear_flops(total_seq, hidden_size, hidden_size * 3 + mlp_hidden_dim)
    single_full += attention_flops(total_seq)
    single_full += linear_flops(total_seq, hidden_size + mlp_hidden_dim, hidden_size)

    # ---- Double-stream block - CACHE (Taylor) ----
    double_cache = 0
    double_cache += linear_flops(1, hidden_size, hidden_size * 6)
    double_cache += linear_flops(1, hidden_size, hidden_size * 6)
    taylor_ops = max_order + 1
    double_cache += 2 * taylor_ops * img_seq_len * hidden_size  # img_attn
    double_cache += 2 * taylor_ops * img_seq_len * hidden_size  # img_mlp
    double_cache += 2 * taylor_ops * txt_seq_len * hidden_size  # txt_attn
    double_cache += 2 * taylor_ops * txt_seq_len * hidden_size  # txt_mlp
    double_cache += 2 * img_seq_len * hidden_size  # img gates
    double_cache += 2 * txt_seq_len * hidden_size  # txt gates

    # ---- Single-stream block - CACHE (Taylor) ----
    single_cache = 0
    single_cache += linear_flops(1, hidden_size, hidden_size * 3)
    single_cache += taylor_ops * total_seq * hidden_size
    single_cache += total_seq * hidden_size

    full_step_flops = fixed_flops + num_double_blocks * double_full + num_single_blocks * single_full
    cache_step_flops = fixed_flops + num_double_blocks * double_cache + num_single_blocks * single_cache

    return {
        'full_step_flops': full_step_flops,
        'cache_step_flops': cache_step_flops,
        'fixed_flops': fixed_flops,
        'double_full': double_full,
        'single_full': single_full,
        'double_cache': double_cache,
        'single_cache': single_cache,
    }


def calc_encoder_decoder_flops(num_blocks, hidden_dim, seq_len=20400, dim=3072):
    """
    FLOPs of InvertibleDecompositionNet.

    Each HybridInvertibleBlock contains:
    - LightweightInvertible1x1Conv: Linear(dim, dim)
    - RevNetResidualBlock:
        - F-net: Linear(half_dim, hidden_dim) + Linear(hidden_dim, hidden_dim) + Linear(hidden_dim, half_dim)
        - G-net: same as F-net

    Forward (decompose): a matmul per token
    Inverse (compose): same as forward + a matrix inverse (2/3 * dim^3 per block, negligible)
    """
    half_dim = dim // 2  # 1536

    per_block = 0
    # 1x1Conv: Linear(dim, dim)
    per_block += 2 * seq_len * dim * dim
    # F-net: 3 linear layers
    per_block += 2 * seq_len * (half_dim * hidden_dim + hidden_dim * hidden_dim + hidden_dim * half_dim)
    # G-net: same as F-net
    per_block += 2 * seq_len * (half_dim * hidden_dim + hidden_dim * hidden_dim + hidden_dim * half_dim)

    forward_flops = num_blocks * per_block  # decompose
    inv_overhead = num_blocks * (2 * dim * dim * dim // 3)  # matrix inverse
    inverse_flops = forward_flops + inv_overhead  # compose

    # parameter count
    params_per_block = dim * dim + 2 * (half_dim * hidden_dim + hidden_dim * hidden_dim + hidden_dim * half_dim)
    total_params = num_blocks * params_per_block

    return {
        'forward': forward_flops,
        'inverse': inverse_flops,
        'per_block': per_block,
        'inv_overhead': inv_overhead,
        'params': total_params,
    }


def calc_step_counts(num_steps, interval):
    """
    TaylorSeer: full step at 0, N, 2N, 3N, ...
    full = floor((num_steps - 1) / interval) + 1
    cache = num_steps - full

    N=5: steps 0,5,10,...,45 → 10 full, 40 cache ✓
    N=6: steps 0,6,12,...,48 → 9 full, 41 cache ✓
    """
    num_full = (num_steps - 1) // interval + 1
    num_cache = num_steps - num_full
    return num_full, num_cache


def format_params(params):
    if params >= 1e6:
        return f'{params / 1e6:.2f}M'
    elif params >= 1e3:
        return f'{params / 1e3:.1f}K'
    else:
        return str(params)


def generate_markdown(output_path='flops_analysis.md'):
    T = lambda x: x / 1e12
    G = lambda x: x / 1e9

    # ---- Measured FLOPs (from internal notes) ----
    # HunyuanVideo: 480x640, 65 frames, 50 steps, H800 80GB
    full_tf  = 595.46e12   # Full step: 595.46 TFLOPS
    cache_tf = 0.145e12    # Cache step (Taylor O=1): 0.145 TFLOPS
    baseline = 29773.0e12  # 50-step baseline: 29773.0 TFLOPS

    num_steps = 50
    intervals = [3,4,5, 6, 7, 8, 9]

    b_values = [1, 2, 3, 4, 5, 6, 7, 8]
    hidden_dim_values = [32, 64, 128, 256, 512, 1024, 2048]

    lines = []

    # ================================================================
    # Part 1: Baseline HunyuanVideo Transformer FLOPs
    # ================================================================
    lines.append('# HunyuanVideo FLOPs Analysis')
    lines.append('')
    lines.append('## 1. HunyuanVideo Transformer FLOPs (Measured)')
    lines.append('')
    lines.append('Source: measured on H800 80GB, 480x640, 65 frames, 50 steps')
    lines.append('')
    lines.append(f'- **Full step**: {T(full_tf):.4f} TFLOPS (595.46 T)')
    lines.append(f'- **Cache step (Taylor O=1)**: {T(cache_tf):.4f} TFLOPS (0.145 T)')
    lines.append(f'- **Baseline (50 full steps)**: {T(baseline):.2f} TFLOPS (29773.0 T)')
    lines.append(f'- **Cache/Full ratio**: {cache_tf / full_tf * 100:.4f}%')
    lines.append('')

    # ================================================================
    # Part 2: TaylorSeer step count and speedup (no encoder/decoder)
    # ================================================================
    lines.append('## 2. TaylorSeer Speedup (No Encoder/Decoder)')
    lines.append('')
    lines.append('Step count formula: full step at 0, N, 2N, ... → `full = floor((num_steps-1) / N) + 1`')
    lines.append('')
    lines.append('| N (interval) | Full Steps | Cache Steps | Total FLOPs (TFLOPS) | Speedup |')
    lines.append('|-------------|-----------|-------------|---------------------|---------|')
    lines.append(f'| baseline | {num_steps} | 0 | {T(baseline):.2f} | 1.00x |')

    for interval in [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
        nf, nc = calc_step_counts(num_steps, interval)
        total = nf * full_tf + nc * cache_tf
        speedup = baseline / total
        lines.append(f'| N={interval} | {nf} | {nc} | {T(total):.2f} | {speedup:.2f}x |')
    lines.append('')

    # ================================================================
    # Part 3: Encoder/Decoder FLOPs for all (b, hidden_dim) combos
    # ================================================================
    lines.append('## 3. Encoder/Decoder (InvertibleDecompositionNet) FLOPs')
    lines.append('')
    lines.append('Applied to model output: seq_len=20400 (img tokens), dim=3072')
    lines.append('')
    lines.append('Architecture per block: 1x1Conv(3072→3072) + RevNet(F-net + G-net, bottleneck=hidden_dim)')
    lines.append('')
    lines.append('| b (blocks) | hidden_dim | Decompose (GFLOPS) | Compose (GFLOPS) | Params |')
    lines.append('|-----------|-----------|-------------------|-----------------|--------|')

    for b in b_values:
        for hd in hidden_dim_values:
            ed = calc_encoder_decoder_flops(b, hd)
            lines.append(
                f'| {b} | {hd} | {G(ed["forward"]):.2f} | {G(ed["inverse"]):.2f} | {format_params(ed["params"])} |'
            )
    lines.append('')

    # ================================================================
    # Part 4: Combined FLOPs for interval=5,6,7 with all (b, hidden_dim)
    # ================================================================
    lines.append('## 4. Combined FLOPs & Speedup (TaylorSeer + Encoder/Decoder)')
    lines.append('')
    lines.append('- **Full step** = transformer full + decompose (1 forward pass through invertible net)')
    lines.append('- **Cache step** = Taylor cache + compose (1 inverse pass through invertible net)')
    lines.append(f'- **Baseline** = {num_steps} steps × {T(full_tf):.2f} TFLOPS/step = {T(baseline):.2f} TFLOPS')
    lines.append('')

    for interval in intervals:
        nf, nc = calc_step_counts(num_steps, interval)

        lines.append(f'### Interval N={interval} (full={nf}, cache={nc})')
        lines.append('')
        lines.append(
            '| b | hidden_dim | Decompose (T) | Compose (T) '
            '| Full+Decompose (T) | Cache+Compose (T) '
            '| 50-step Total (T) | Speedup |'
        )
        lines.append(
            '|---|-----------|--------------|------------ '
            '|-------------------|------------------ '
            '|------------------|---------|'
        )

        for b in b_values:
            for hd in hidden_dim_values:
                ed = calc_encoder_decoder_flops(b, hd)
                full_with_ed = full_tf + ed['forward']
                cache_with_ed = cache_tf + ed['inverse']
                total_50 = nf * full_with_ed + nc * cache_with_ed
                speedup = baseline / total_50
                lines.append(
                    f'| {b} | {hd} '
                    f'| {T(ed["forward"]):.4f} | {T(ed["inverse"]):.4f} '
                    f'| {T(full_with_ed):.4f} | {T(cache_with_ed):.4f} '
                    f'| {T(total_50):.2f} | {speedup:.2f}x |'
                )
        lines.append('')

    # ================================================================
    # Part 5: Summary - best combos
    # ================================================================
    lines.append('## 5. Summary: Encoder/Decoder Overhead Ratio')
    lines.append('')
    lines.append('Compose FLOPs as percentage of full transformer step:')
    lines.append('')
    lines.append('| b | hidden_dim | Compose (TFLOPS) | % of Full Step |')
    lines.append('|---|-----------|-----------------|---------------|')

    for b in b_values:
        for hd in hidden_dim_values:
            ed = calc_encoder_decoder_flops(b, hd)
            pct = ed['inverse'] / full_tf * 100
            lines.append(f'| {b} | {hd} | {T(ed["inverse"]):.4f} | {pct:.2f}% |')
    lines.append('')

    # Write to file
    content = '\n'.join(lines)
    with open(output_path, 'w') as f:
        f.write(content)
    print(f'Results saved to {output_path}')
    print()
    # Print a condensed summary to console
    print('=' * 60)
    print('Quick Summary')
    print('=' * 60)
    print(f'Full step:  {T(full_tf):.4f} TFLOPS')
    print(f'Cache step: {T(cache_tf):.4f} TFLOPS')
    print(f'Baseline (50 full): {T(baseline):.2f} TFLOPS')
    print()
    for interval in intervals:
        nf, nc = calc_step_counts(num_steps, interval)
        taylor_only = nf * full_tf + nc * cache_tf
        print(f'N={interval}: full={nf}, cache={nc}, Taylor-only={T(taylor_only):.2f}T, speedup={baseline/taylor_only:.2f}x')
    print()
    print('Encoder/Decoder overhead examples:')
    for b, hd in [(1, 64), (2, 64), (2, 256), (3, 128), (5, 2048)]:
        ed = calc_encoder_decoder_flops(b, hd)
        pct = ed['inverse'] / full_tf * 100
        print(f'  b={b}, N={hd}: compose={G(ed["inverse"]):.1f} GFLOPS ({pct:.2f}% of full step)')


if __name__ == '__main__':
    generate_markdown()
