"""
Shared Components for Cross-Scale Weight Sharing
==================================================
Reduces parameter count by sharing common components across detection scales.

Weight sharing rationale:
  - Noise routing dynamics should be consistent across scales (same noise affects all)
  - Confidence estimation follows similar patterns regardless of scale
  - Scale-specific adaptation handled by lightweight per-scale projections

Savings (edge profile, channels=[64, 128, 256]):
  - SharedNoiseRouter:        ~17.5K params saved (69% router reduction)
  - SharedConfidenceRefiner:  ~6.7K params saved (31% refiner reduction)
  - Total:                    ~24.2K (6.2% of plugin overhead)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class SharedNoiseRouter(nn.Module):
    """Shared noise-conditioned router with per-scale channel adaptation.

    Instead of 3 independent routers (one per scale), uses a single shared
    noise projection followed by lightweight per-scale channel projections.

    Architecture:
        noise_level [B, noise_dim]
            → shared_noise_proj → [B, mid]
            → per_scale_proj[i] → [B, C_i]
            → gate_min + (1 - gate_min) * sigmoid → alpha [B, C_i, 1, 1]

    Args:
        channels_list: Channel count per scale, e.g. [64, 128, 256].
        noise_dim: Noise descriptor dimension.
        router_type: 'full' | 'simplified' | 'fixed'.
        reduction: Channel reduction ratio for shared projection.
        gate_min: Minimum gate value (floor).
        learnable_floor: Whether gate_min is learnable.
    """

    def __init__(
        self,
        channels_list: List[int],
        noise_dim: int = 32,
        router_type: str = "simplified",
        reduction: int = 4,
        gate_min: float = 0.05,
        learnable_floor: bool = False,
    ):
        super().__init__()
        self.router_type = router_type
        self.channels_list = channels_list
        self.learnable_floor = learnable_floor

        if router_type == "fixed":
            return

        # Gate floor
        if learnable_floor and gate_min > 0:
            init_raw = math.log(gate_min / (1.0 - gate_min + 1e-8))
            self._gate_min_raw = nn.Parameter(torch.tensor(init_raw))
            self.max_floor = 0.20
        else:
            self.register_buffer("_gate_min_fixed", torch.tensor(float(gate_min)))

        # Shared: noise projection to common mid dimension
        mid = max(max(channels_list) // reduction, 8)
        self.shared_noise_proj = nn.Sequential(
            nn.Linear(noise_dim, mid),
            nn.ReLU(inplace=True),
        )

        # Per-scale: mid → channels projection
        self.per_scale_proj = nn.ModuleList([
            nn.Linear(mid, ch) for ch in channels_list
        ])

        # Full router: also uses feature statistics (per-scale)
        if router_type == "full":
            self.feat_stats_list = nn.ModuleList([
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(ch, max(ch // reduction, 8)),
                    nn.ReLU(inplace=True),
                    nn.Linear(max(ch // reduction, 8), ch),
                )
                for ch in channels_list
            ])
            self.gate_norms = nn.ModuleList([
                nn.LayerNorm(ch) for ch in channels_list
            ])

    def forward(
        self,
        scale_idx: int,
        features: torch.Tensor,
        noise_level: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute routing weight alpha for a specific scale.

        Args:
            scale_idx: Scale index (0, 1, 2, ...).
            features: Feature map [B, C_i, H, W].
            noise_level: Noise descriptor [B, noise_dim] or None.

        Returns:
            alpha: Routing weight [B, C_i, 1, 1] in [gate_min, 1.0].
        """
        if self.router_type == "fixed":
            B = features.shape[0]
            return torch.full(
                (B, 1, 1, 1), 0.5,
                device=features.device, dtype=features.dtype,
            )

        if noise_level is None:
            B = features.shape[0]
            return torch.full(
                (B, 1, 1, 1), 0.5,
                device=features.device, dtype=features.dtype,
            )

        # Shared noise projection
        shared_feat = self.shared_noise_proj(noise_level)  # [B, mid]
        alpha = self.per_scale_proj[scale_idx](shared_feat)  # [B, C_i]

        if self.router_type == "full":
            feat_gate = self.feat_stats_list[scale_idx](features)  # [B, C_i]
            alpha = self.gate_norms[scale_idx](alpha + feat_gate)

        # Apply gate floor
        if self.learnable_floor and hasattr(self, '_gate_min_raw'):
            gate_min = torch.sigmoid(self._gate_min_raw).clamp(max=self.max_floor)
        else:
            gate_min = self._gate_min_fixed

        alpha = gate_min + (1.0 - gate_min) * torch.sigmoid(alpha)
        return alpha.unsqueeze(-1).unsqueeze(-1)  # [B, C_i, 1, 1]


class SharedConfidenceRefiner(nn.Module):
    """Shared confidence refiner with per-scale input projection.

    Uses per-scale projections to a common dimension, then a shared MLP
    to produce confidence scores.

    Architecture:
        features [B, C_i, H, W]
            → AdaptiveAvgPool2d(1) → [B, C_i]
            → input_projs[i] → [B, common_dim]
            → shared_core → [B, 1] (confidence score)

    Args:
        channels_list: Channel count per scale.
        common_dim: Shared intermediate dimension.
    """

    def __init__(self, channels_list: List[int], common_dim: int = 32):
        super().__init__()
        self.channels_list = channels_list
        self.common_dim = common_dim

        # Per-scale: channel → common_dim projection
        self.input_projs = nn.ModuleList([
            nn.Linear(ch, common_dim) for ch in channels_list
        ])

        # Shared: MLP core for confidence estimation
        self.shared_core = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(common_dim, common_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(common_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, scale_idx: int, features: torch.Tensor) -> torch.Tensor:
        """Compute confidence score for a specific scale.

        Args:
            scale_idx: Scale index.
            features: Feature map [B, C_i, H, W].

        Returns:
            confidence: [B, 1] confidence weight.
        """
        pooled = F.adaptive_avg_pool2d(features, 1).flatten(1)  # [B, C_i]
        projected = self.input_projs[scale_idx](pooled)  # [B, common_dim]
        return self.shared_core(projected)  # [B, 1]
