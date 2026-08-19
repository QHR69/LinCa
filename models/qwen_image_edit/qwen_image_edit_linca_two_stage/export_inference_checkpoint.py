"""
Export inference-only weights from a full training checkpoint (drop optimizer, etc.)

Usage:
  python export_inference_checkpoint.py \\
    --input checkpoints/checkpoint.pt \\
    --output checkpoints/checkpoint_inference.pt
"""
import argparse
import os
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="full training checkpoint")
    ap.add_argument("--output", required=True, help="output inference checkpoint")
    args = ap.parse_args()

    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"Input not found: {args.input}")

    ckpt = torch.load(args.input, map_location="cpu", weights_only=False)
    if not (isinstance(ckpt, dict) and "predictor_stage1" in ckpt and "predictor_stage2" in ckpt):
        raise ValueError("Checkpoint must contain predictor_stage1 and predictor_stage2")

    slim = {
        "predictor_stage1": ckpt["predictor_stage1"],
        "predictor_stage2": ckpt["predictor_stage2"],
        "config": ckpt["config"],
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(slim, args.output)

    orig_size = os.path.getsize(args.input) / (1024 * 1024)
    new_size = os.path.getsize(args.output) / (1024 * 1024)
    print(f"Exported: {args.output}")
    print(f"  Original: {orig_size:.1f} MB -> Inference: {new_size:.1f} MB")


if __name__ == "__main__":
    main()
