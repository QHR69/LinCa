"""
Invertible Decomposition Network - hybrid architecture (gate-free variant).

Hybrid design: lightweight Glow-style mixing + RevNet residual coupling,
configurable partitioning, and a fixed per-partition prediction policy.

Feature dimension: 3072
Prediction policy: fixed partitioning (0th / 1st / 2nd order) with
configurable partition sizes.

Invertibility: exact by construction, no reconstruction error.

Architecture notes:
- Lightweight invertible 1x1 convolution: strengthens channel mixing
- RevNet residual block: cheap exact inversion
- Configurable partitioning: arbitrary split_dims
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
import math


class LightweightInvertible1x1Conv(nn.Module):
    """
    Lightweight invertible 1x1 convolution.

    Orthogonal initialisation keeps the condition number well behaved.

    Forward: y = W @ x
    Inverse: x = W^{-1} @ y = W^T @ y (orthogonal matrix)
    """
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
        # Orthogonal initialisation gives a condition number of 1.
        W = torch.eye(dim)
        # Perturb slightly, then QR-decompose to recover an orthogonal matrix.
        W = W + 0.01 * torch.randn(dim, dim)
        q, r = torch.linalg.qr(W)
        # Force a positive determinant (rotation rather than reflection).
        d = torch.diag(r)
        sign = torch.sign(d)
        W_init = q * sign.unsqueeze(0)
        
        self.weight = nn.Parameter(W_init)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward transform: y = W @ x"""
        return F.linear(x, self.weight.t())
    
    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Inverse transform: x = W^{-1} @ y. Uses the cached inverse once
        register_inverse_weight() has been called, otherwise inverts on the
        fly (training path)."""
        if hasattr(self, 'weight_inv') and self.weight_inv is not None:
            return F.linear(y, self.weight_inv.t())
        W_inv = torch.linalg.inv(self.weight)
        return F.linear(y, W_inv.t())

    def register_inverse_weight(self):
        """Call before inference: precompute and cache the inverse weight so
        that torch.linalg.inv is never invoked during sampling."""
        with torch.no_grad():
            self.register_buffer('weight_inv', torch.linalg.inv(self.weight.detach()))


class RevNetResidualBlock(nn.Module):
    """
    Reversible residual block, RevNet style (with dropout for regularisation).

    Forward:
    y1 = x1 + scale_F * F(x2)
    y2 = x2 + scale_G * G(y1)

    Inverse:
    x2 = y2 - scale_G * G(y1)
    x1 = y1 - scale_F * F(x2)

    The key property is that F depends only on x2 and G only on y1, which
    makes the mapping exactly invertible.
    """
    
    def __init__(self, half_dim: int, hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.half_dim = half_dim
        self.dropout_rate = dropout
        
        # F network: takes x2, produces a residual (with dropout).
        self.F = nn.Sequential(
            nn.Linear(half_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, half_dim),
        )
        
        # G network: takes y1, produces a residual (with dropout).
        self.G = nn.Sequential(
            nn.Linear(half_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, half_dim),
        )
        
        # Learnable residual scaling factors (stabilise training).
        self.scale_F = nn.Parameter(torch.tensor(0.1))
        self.scale_G = nn.Parameter(torch.tensor(0.1))
        
        # Zero-initialise the last layer so the block starts near identity.
        self._init_zero_last_layer()
    
    def _init_zero_last_layer(self):
        """Zero-initialise the final linear layer."""
        for module in [self.F, self.G]:
            for m in reversed(list(module.modules())):
                if isinstance(m, nn.Linear):
                    nn.init.zeros_(m.weight)
                    nn.init.zeros_(m.bias)
                    break
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward: x1, x2 -> y1, y2"""
        y1 = x1 + self.scale_F * self.F(x2)
        y2 = x2 + self.scale_G * self.G(y1)
        return y1, y2
    
    def inverse(self, y1: torch.Tensor, y2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inverse: y1, y2 -> x1, x2"""
        x2 = y2 - self.scale_G * self.G(y1)
        x1 = y1 - self.scale_F * self.F(x2)
        return x1, x2


class HybridInvertibleBlock(nn.Module):
    """
    Hybrid invertible block = lightweight 1x1 convolution + RevNet residual
    block (with dropout).

    Combines Glow's channel-mixing capacity with RevNet's cheap invertibility.
    """
    
    def __init__(self, dim: int, hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.half_dim = dim // 2
        
        # Lightweight 1x1 convolution (optional, strengthens channel mixing).
        self.conv1x1 = LightweightInvertible1x1Conv(dim)
        
        # Dropout, applied after the 1x1 convolution.
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # RevNet residual block (with dropout).
        self.revnet = RevNetResidualBlock(self.half_dim, hidden_dim, dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: x -> z"""
        # 1. Mix channels with the 1x1 convolution.
        x = self.conv1x1(x)
        
        # 2. Dropout (training only).
        x = self.dropout(x)
        
        # 3. Split into two halves.
        x1, x2 = x[..., :self.half_dim], x[..., self.half_dim:]
        
        # 4. RevNet residual block.
        y1, y2 = self.revnet(x1, x2)
        
        # 5. Concatenate the output.
        return torch.cat([y1, y2], dim=-1)
    
    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """Inverse: z -> x"""
        # 1. Split.
        y1, y2 = z[..., :self.half_dim], z[..., self.half_dim:]
        
        # 2. Invert the RevNet block.
        x1, x2 = self.revnet.inverse(y1, y2)
        
        # 3. Concatenate.
        x = torch.cat([x1, x2], dim=-1)
        
        # 4. Invert the 1x1 convolution.
        x = self.conv1x1.inverse(x)
        
        return x


class InvertibleDecompositionNet(nn.Module):
    """
    Invertible decomposition network built from the hybrid architecture
    (with dropout for regularisation).

    A stack of HybridInvertibleBlock modules.

    Args:
        dim: feature dimension (3072)
        num_blocks: number of hybrid blocks
        hidden_dim: MLP hidden dimension
        dropout: dropout rate (regularisation)
    """
    
    def __init__(
        self,
        dim: int = 3072,
        num_blocks: int = 6,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.num_blocks = num_blocks
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        # Stack of hybrid invertible blocks (with dropout).
        self.blocks = nn.ModuleList([
            HybridInvertibleBlock(dim, hidden_dim, dropout)
            for _ in range(num_blocks)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward transform: x -> z"""
        z = x
        for block in self.blocks:
            z = block(z)
        return z
    
    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """Inverse transform: z -> x (exact)"""
        x = z
        for block in reversed(self.blocks):
            x = block.inverse(x)
        return x
    
    def decompose(self, x: torch.Tensor, split_dims: list = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Decompose the feature into three partitions.

        Partitioning is configurable: split_dims gives the width of each
        partition.
        """
        if split_dims is None:
            split_dims = [1024, 1024, 1024]
        z = self.forward(x)
        z0, z1, z2 = torch.split(z, split_dims, dim=-1)
        return z0, z1, z2
    
    def compose(self, z0: torch.Tensor, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct the original feature from the three partitions.
        """
        z = torch.cat([z0, z1, z2], dim=-1)
        return self.inverse(z)


class FixedPredictionStrategy(nn.Module):
    """
    Fixed per-partition prediction policy.

    The three partitions are extrapolated differently:
    - z0: 0th order (plain reuse)
    - z1: 1st order (linear extrapolation)
    - z2: 2nd order (Lagrange interpolation)
    """
    
    def __init__(self):
        super().__init__()
    
    def predict_z0(self, z_prev: torch.Tensor) -> torch.Tensor:
        """0th-order prediction: reuse the cached value."""
        return z_prev.clone()
    
    def predict_z1(
        self,
        z_prev: torch.Tensor,
        z_prev2: torch.Tensor,
        dt1: float,
        dt2: float,
    ) -> torch.Tensor:
        """1st-order prediction: linear extrapolation."""
        # f(t) ~= f(t-dt1) + (f(t-dt1) - f(t-dt2)) * dt1 / (dt2 - dt1)
        if abs(dt2 - dt1) < 1e-8:
            return z_prev.clone()
        slope = (z_prev - z_prev2) / (dt2 - dt1)
        return z_prev + slope * dt1
    
    def predict_z2(
        self,
        z_prev: torch.Tensor,
        z_prev2: torch.Tensor,
        z_prev3: torch.Tensor,
        dt1: float,
        dt2: float,
        dt3: float,
    ) -> torch.Tensor:
        """2nd-order prediction: Lagrange interpolation."""
        # Quadratic interpolation through three points.
        t0, t1, t2 = 0, -dt1, -dt2
        t3 = -dt3
        
        # Lagrange basis functions.
        denom01 = (t1 - t0) if abs(t1 - t0) > 1e-8 else 1e-8
        denom02 = (t2 - t0) if abs(t2 - t0) > 1e-8 else 1e-8
        denom12 = (t2 - t1) if abs(t2 - t1) > 1e-8 else 1e-8
        
        L0 = ((t0 - t1) * (t0 - t2)) / (denom01 * denom02)
        L1 = ((t0 - t0) * (t0 - t2)) / ((-denom01) * denom12)
        L2 = ((t0 - t0) * (t0 - t1)) / ((-denom02) * (-denom12))
        
        # Simplified form: use finite differences.
        if abs(dt2 - dt1) < 1e-8 or abs(dt3 - dt2) < 1e-8:
            return self.predict_z1(z_prev, z_prev2, dt1, dt2)
        
        # First-order difference.
        d1 = (z_prev - z_prev2) / (dt2 - dt1)
        d2 = (z_prev2 - z_prev3) / (dt3 - dt2)
        
        # Second-order difference.
        d2_diff = (d1 - d2) / (dt3 - dt1)
        
        # Prediction.
        pred = z_prev + d1 * dt1 + 0.5 * d2_diff * dt1 * dt1
        return pred

    def predict_z2_hermite(
        self,
        z_prev: torch.Tensor,
        z_prev2: torch.Tensor,
        z_prev3: torch.Tensor,
        dt1: float,
        dt2: float,
        dt3: float,
    ) -> torch.Tensor:
        """2nd-order prediction: Hermite polynomial expansion."""
        if abs(dt2 - dt1) < 1e-8 or abs(dt3 - dt2) < 1e-8:
            return self.predict_z1(z_prev, z_prev2, dt1, dt2)
        d1 = (z_prev - z_prev2) / (dt2 - dt1)
        d2 = (z_prev2 - z_prev3) / (dt3 - dt2)
        d2_diff = (d1 - d2) / (dt3 - dt1)
        module_dict = {0: z_prev.clone(), 1: d1, 2: d2_diff}
        return _hermite_formula_z2(module_dict, dt1, 2, z_prev.dtype, z_prev.device)


def _hermite_polynomial(x: torch.Tensor, n: int) -> torch.Tensor:
    """Physicist's Hermite polynomial H_n(x)"""
    if n == 0:
        return torch.ones_like(x)
    elif n == 1:
        return 2 * x
    H_prev, H_curr = torch.ones_like(x), 2 * x
    for k in range(2, n + 1):
        H_next = 2 * x * H_curr - 2 * (k - 1) * H_prev
        H_prev, H_curr = H_curr, H_next
    return H_curr


def _hermite_formula_z2(module_dict: dict, distance: float, max_order: int, dtype, device) -> torch.Tensor:
    """Hermite expansion used to extrapolate the z2 partition."""
    pred = module_dict[0].clone()
    x_tensor = torch.tensor(float(distance), dtype=dtype, device=device)
    scale_factor = 0.71
    x_scaled = x_tensor * scale_factor
    for k in range(1, min(max_order, len(module_dict) - 1) + 1):
        diff_k = module_dict[k]
        Hk = _hermite_polynomial(x_scaled, k)
        alpha = Hk / math.factorial(k) * (scale_factor ** k)
        pred = pred + alpha * diff_k
    return pred


class LearnedDecompositionPredictor(nn.Module):
    """
    Hybrid architecture + configurable partitioning + fixed prediction policy
    (gate-free variant, with dropout for regularisation).

    Composed of:
    1. Hybrid invertible decomposition network (Glow + RevNet)
    2. Fixed prediction policy (0th / 1st / 2nd order)
    3. Configurable partition sizes (split_dims)
    """
    
    def __init__(
        self,
        dim: int = 3072,
        num_blocks: int = 6,
        hidden_dim: int = 512,
        split_dims: list = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.num_blocks = num_blocks
        self.hidden_dim = hidden_dim
        self.split_dims = split_dims if split_dims is not None else [1024, 1024, 1024]
        self.dropout = dropout
        
        # split_dims must sum to dim.
        assert sum(self.split_dims) == dim, \
            f"split_dims sum ({sum(self.split_dims)}) must equal dim ({dim})"
        
        # Hybrid invertible decomposition network (with dropout).
        self.net = InvertibleDecompositionNet(
            dim=dim,
            num_blocks=num_blocks,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        
        # Fixed prediction policy.
        self.prediction_strategy = FixedPredictionStrategy()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: decompose the feature."""
        return self.net.forward(x)
    
    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """Inverse: reconstruct the feature."""
        return self.net.inverse(z)
    
    def decompose(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decompose the feature."""
        return self.net.decompose(x, split_dims=self.split_dims)
    
    def compose(self, z0: torch.Tensor, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Reconstruct the feature."""
        return self.net.compose(z0, z1, z2)

    def register_inverse_weights(self):
        """Call before inference: precompute the inverse weight of every 1x1
        convolution so that torch.linalg.inv is never invoked during sampling."""
        for block in self.net.blocks:
            block.conv1x1.register_inverse_weight()
    
    def predict(
        self,
        cache_list: List[Tuple[float, torch.Tensor]],
        order: int = 2,
        z2_forecast_method: str = "lagrange",
    ) -> torch.Tensor:
        """
        Predict the next-step feature from cached history.

        Args:
            cache_list: cache entries [(dt1, feat1), (dt2, feat2), ...] where
                        dt is the temporal distance from the current step
            order: maximum prediction order

        Returns:
            predicted: the predicted feature
        """
        if len(cache_list) == 0:
            raise ValueError("Cache list is empty")
        
        # Most recent cache entry.
        dt1, feat_prev = cache_list[0]
        
        # Decompose the most recent feature (using split_dims).
        z_prev = self.forward(feat_prev)
        z0_prev, z1_prev, z2_prev = torch.split(z_prev, self.split_dims, dim=-1)
        
        # 0th-order prediction for z0.
        z0_pred = self.prediction_strategy.predict_z0(z0_prev)
        
        # 1st-order prediction for z1.
        if len(cache_list) >= 2 and order >= 1:
            dt2, feat_prev2 = cache_list[1]
            z_prev2 = self.forward(feat_prev2)
            _, z1_prev2, _ = torch.split(z_prev2, self.split_dims, dim=-1)
            z1_pred = self.prediction_strategy.predict_z1(z1_prev, z1_prev2, dt1, dt2)
        else:
            z1_pred = self.prediction_strategy.predict_z0(z1_prev)
        
        # 2nd-order prediction for z2.
        if len(cache_list) >= 3 and order >= 2:
            dt2, feat_prev2 = cache_list[1]
            dt3, feat_prev3 = cache_list[2]
            z_prev2 = self.forward(feat_prev2)
            z_prev3 = self.forward(feat_prev3)
            _, _, z2_prev2 = torch.split(z_prev2, self.split_dims, dim=-1)
            _, _, z2_prev3 = torch.split(z_prev3, self.split_dims, dim=-1)
            if z2_forecast_method == "hermite":
                z2_pred = self.prediction_strategy.predict_z2_hermite(
                    z2_prev, z2_prev2, z2_prev3, dt1, dt2, dt3
                )
            else:
                z2_pred = self.prediction_strategy.predict_z2(
                    z2_prev, z2_prev2, z2_prev3, dt1, dt2, dt3
                )
        elif len(cache_list) >= 2 and order >= 1:
            dt2, feat_prev2 = cache_list[1]
            z_prev2 = self.forward(feat_prev2)
            _, _, z2_prev2 = torch.split(z_prev2, self.split_dims, dim=-1)
            z2_pred = self.prediction_strategy.predict_z1(z2_prev, z2_prev2, dt1, dt2)
        else:
            z2_pred = self.prediction_strategy.predict_z0(z2_prev)
        
        # Concatenate the three partition predictions.
        z_pred = torch.cat([z0_pred, z1_pred, z2_pred], dim=-1)
        
        # Reconstruct.
        predicted = self.inverse(z_pred)
        
        return predicted
    
    def predict_from_decomposed(
        self,
        decomposed_cache_list: List[Tuple[float, torch.Tensor, torch.Tensor, torch.Tensor]],
        order: int = 2,
        z2_forecast_method: str = "lagrange",
    ) -> torch.Tensor:
        """
        Predict the next-step feature from an already decomposed cache, which
        avoids recomputing the forward transform.

        Args:
            decomposed_cache_list: decomposed cache entries [(dt, z0, z1, z2), ...]
                                   where dt is the temporal distance from the
                                   current step
            order: maximum prediction order

        Returns:
            predicted: the predicted feature
        """
        if len(decomposed_cache_list) == 0:
            raise ValueError("Decomposed cache list is empty")
        
        # Most recent cache entry.
        dt1, z0_prev, z1_prev, z2_prev = decomposed_cache_list[0]
        
        # 0th-order prediction for z0.
        z0_pred = self.prediction_strategy.predict_z0(z0_prev)
        
        # 1st-order prediction for z1.
        if len(decomposed_cache_list) >= 2 and order >= 1:
            dt2, _, z1_prev2, _ = decomposed_cache_list[1]
            z1_pred = self.prediction_strategy.predict_z1(z1_prev, z1_prev2, dt1, dt2)
        else:
            z1_pred = self.prediction_strategy.predict_z0(z1_prev)
        
        # 2nd-order prediction for z2.
        if len(decomposed_cache_list) >= 3 and order >= 2:
            dt2, _, _, z2_prev2 = decomposed_cache_list[1]
            dt3, _, _, z2_prev3 = decomposed_cache_list[2]
            if z2_forecast_method == "hermite":
                z2_pred = self.prediction_strategy.predict_z2_hermite(
                    z2_prev, z2_prev2, z2_prev3, dt1, dt2, dt3
                )
            else:
                z2_pred = self.prediction_strategy.predict_z2(
                    z2_prev, z2_prev2, z2_prev3, dt1, dt2, dt3
                )
        elif len(decomposed_cache_list) >= 2 and order >= 1:
            dt2, _, _, z2_prev2 = decomposed_cache_list[1]
            z2_pred = self.prediction_strategy.predict_z1(z2_prev, z2_prev2, dt1, dt2)
        else:
            z2_pred = self.prediction_strategy.predict_z0(z2_prev)
        
        # Concatenate the three partition predictions.
        z_pred = torch.cat([z0_pred, z1_pred, z2_pred], dim=-1)
        
        # A single inverse call reconstructs the feature.
        predicted = self.inverse(z_pred)
        
        return predicted
    
    def save_pretrained(self, save_path: str):
        """Save the model."""
        import os
        import json
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save the weights.
        torch.save(self.state_dict(), save_path)
        
        # Save the config.
        config = {
            'dim': self.dim,
            'num_blocks': self.num_blocks,
            'hidden_dim': self.hidden_dim,
            'split_dims': self.split_dims,
            'dropout': self.dropout,
        }
        config_path = save_path.replace('.pt', '_config.json')
        if not config_path.endswith('_config.json'):
            config_path = save_path + '_config.json'
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    @classmethod
    def from_pretrained(cls, checkpoint_path: str, device: str = 'cuda'):
        """Load the model from a checkpoint."""
        import json
        import os
        
        # Try to load the config.
        config_path = checkpoint_path.replace('.pt', '_config.json')
        if not os.path.exists(config_path):
            config_path = checkpoint_path + '_config.json'
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            # Fall back to the default config.
            config = {
                'dim': 3072,
                'num_blocks': 6,
                'hidden_dim': 512,
                'split_dims': [1024, 1024, 1024],
                'dropout': 0.1,
            }
        
        # Build the model.
        model = cls(
            dim=config['dim'],
            num_blocks=config['num_blocks'],
            hidden_dim=config['hidden_dim'],
            split_dims=config.get('split_dims', [1024, 1024, 1024]),
            dropout=config.get('dropout', 0.1),
        )
        
        # Load the weights.
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(device)
        model.eval()
        
        return model


# Self-test
if __name__ == "__main__":
    print("=" * 60)
    print("Hybrid architecture + configurable partitioning + fixed prediction (gate-free): self-test")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Build the model (with dropout, no gating).
    predictor = LearnedDecompositionPredictor(
        dim=3072,
        num_blocks=6,
        hidden_dim=512,
        split_dims=[1024, 1024, 1024],
        dropout=0.1,
    ).to(device)
    
    print(f"\nParameter count: {sum(p.numel() for p in predictor.parameters()):,}")
    print(f"Partitioning: {predictor.split_dims}")
    
    # Test data.
    x = torch.randn(2, 100, 3072, device=device)
    
    # Invertibility check.
    print("\n[1] Checking invertibility...")
    with torch.no_grad():
        z = predictor.forward(x)
        x_recon = predictor.inverse(z)
        error = (x - x_recon).abs().max().item()
        print(f"   Reconstruction error: {error:.2e}")
        
        if error < 1e-4:  # 1e-4 is comfortably accurate enough in practice
            print("   OK: invertibility check passed (sufficient precision)")
        else:
            print("   FAIL: invertibility check failed (error too large)")
    
    # Decompose / reconstruct check.
    print("\n[2] Checking decompose/reconstruct...")
    with torch.no_grad():
        z0, z1, z2 = predictor.decompose(x)
        x_recon = predictor.compose(z0, z1, z2)
        error = (x - x_recon).abs().max().item()
        print(f"   Partition shapes: z0={z0.shape}, z1={z1.shape}, z2={z2.shape}")
        print(f"   Reconstruction error: {error:.2e}")
    
    # Prediction check.
    print("\n[3] Checking prediction...")
    cache_list = [
        (1.0, torch.randn(100, 3072, device=device)),
        (5.0, torch.randn(100, 3072, device=device)),
        (10.0, torch.randn(100, 3072, device=device)),
    ]
    with torch.no_grad():
        pred = predictor.predict(cache_list, order=2)
        print(f"   Prediction shape: {pred.shape}")
        print(f"   Prediction range: [{pred.min():.4f}, {pred.max():.4f}]")
        print(f"   Prediction NaN: {torch.isnan(pred).any()}")
    
    # predict_from_decomposed check.
    print("\n[4] Checking predict_from_decomposed...")
    with torch.no_grad():
        decomposed_list = []
        for dt, feat in cache_list:
            z0_c, z1_c, z2_c = predictor.decompose(feat)
            decomposed_list.append((dt, z0_c, z1_c, z2_c))
        pred_decomp = predictor.predict_from_decomposed(decomposed_list, order=2)
        print(f"   Prediction shape: {pred_decomp.shape}")
        print(f"   Prediction range: [{pred_decomp.min():.4f}, {pred_decomp.max():.4f}]")
        # The two prediction paths must agree.
        diff = (pred - pred_decomp).abs().max().item()
        print(f"   Difference vs predict: {diff:.2e}")
        if diff < 1e-4:
            print("   OK: both prediction paths agree")
        else:
            print("   FAIL: prediction paths disagree (possible bug)")
    
    print("\n" + "=" * 60)
    print("All checks complete.")
    print("=" * 60)
