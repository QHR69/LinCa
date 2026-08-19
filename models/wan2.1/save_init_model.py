"""Save an untrained randomly-initialised invertible net for a baseline (HunyuanVideo)"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from flux.modules.invertible_net import LearnedDecompositionPredictor

# Matches the current run_train.sh configuration
dim = 3072
num_blocks = 6
hidden_dim = 128
split_dims = [2304, 384, 384]
dropout = 0.0

model = LearnedDecompositionPredictor(
    dim=dim,
    num_blocks=num_blocks,
    hidden_dim=hidden_dim,
    split_dims=split_dims,
    dropout=dropout,
)

split_str = "x".join(map(str, split_dims))

save_path = f"outputs/init_model_blocks_{num_blocks}_hidden_{hidden_dim}_split_{split_str}_dim_{dim}_dropout_{dropout}/best_predictor.pt"

os.makedirs(os.path.dirname(save_path), exist_ok=True)
model.save_pretrained(save_path)
print(f"Saved untrained model to {save_path}")
