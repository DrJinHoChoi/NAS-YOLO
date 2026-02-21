"""
Hybrid Mamba-Attention Block
=============================
Combines Selective SSM (Mamba) with lightweight spatial attention for
robust feature refinement. Noise-conditioned routing dynamically
balances temporal smoothing (Mamba) vs spatial detail (Attention).

Supports 4 profiles for edge-to-cloud deployment:
  - Standard: MHSA 4 heads + expand=2 SSM (server)
  - Lite:     MHSA 2 heads + expand=1 SSM (desktop GPU)
  - Edge:     DWConv attention + expand=1 SSM (Jetson Orin) -- DEFAULT
  - Ultra-lite: SE-block + expand=1 d_state=4 SSM (Jetson Nano)

References:
  - Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces", 2023
  - Liu et al., "Swin Transformer: Hierarchical Vision Transformer", ICCV 2021
  - Hatamizadeh et al., "MambaVision: A Hybrid Mamba-Transformer Vision Backbone", 2024
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .nas_module import SpatialSSMAdapter


# =============================================================================
# Profile Presets
# =============================================================================

PROFILE_CONFIGS = {
    "standard": {
        "expand": 2, "d_state": 16, "attention_type": "mhsa",
        "num_heads": 4, "window_size": 4, "router_type": "full",
    },
    "lite": {
        "expand": 1, "d_state": 8, "attention_type": "mhsa",
        "num_heads": 2, "window_size": 4, "router_type": "full",
    },
    "edge": {
        "expand": 1, "d_state": 8, "attention_type": "dwconv",
        "num_heads": 1, "window_size": 0, "router_type": "simplified",
    },
    "ultra-lite": {
        "expand": 1, "d_state": 4, "attention_type": "se",
        "num_heads": 1, "window_size": 0, "router_type": "fixed",
    },
}


# =============================================================================
# Spatial Attention Variants
# =============================================================================

def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, int, int]:
    """Partition feature map into non-overlapping windows.

    Args:
        x: [B, C, H, W]
        window_size: window size

    Returns:
        windows: [B * num_windows, C, window_size, window_size]
        H, W: original spatial dims (for window_merge)
    """
    B, C, H, W = x.shape
    # Pad if not divisible
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h))

    Hp, Wp = x.shape[2], x.shape[3]
    nH, nW = Hp // window_size, Wp // window_size

    # [B, C, nH, ws, nW, ws] -> [B, nH, nW, C, ws, ws] -> [B*nH*nW, C, ws, ws]
    x = x.reshape(B, C, nH, window_size, nW, window_size)
    x = x.permute(0, 2, 4, 1, 3, 5).reshape(B * nH * nW, C, window_size, window_size)
    return x, H, W


def window_merge(windows: torch.Tensor, window_size: int,
                 H: int, W: int, B: int) -> torch.Tensor:
    """Merge windows back to feature map.

    Args:
        windows: [B * num_windows, C, window_size, window_size]
        window_size: window size
        H, W: original spatial dims
        B: batch size

    Returns:
        x: [B, C, H, W]
    """
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    Hp, Wp = H + pad_h, W + pad_w
    nH, nW = Hp // window_size, Wp // window_size
    C = windows.shape[1]

    # [B*nH*nW, C, ws, ws] -> [B, nH, nW, C, ws, ws] -> [B, C, Hp, Wp]
    x = windows.reshape(B, nH, nW, C, window_size, window_size)
    x = x.permute(0, 3, 1, 4, 2, 5).reshape(B, C, Hp, Wp)

    # Remove padding
    if pad_h > 0 or pad_w > 0:
        x = x[:, :, :H, :W]
    return x


class WindowMHSA(nn.Module):
    """Windowed Multi-Head Self-Attention (Standard/Lite profile).

    Partitions features into windows and applies MHSA within each window.
    No cross-window communication for speed.

    Args:
        dim: Channel dimension.
        num_heads: Number of attention heads.
        window_size: Window size (default 4).
        qkv_bias: Use bias in QKV projection.
    """

    def __init__(self, dim: int, num_heads: int = 4,
                 window_size: int = 4, qkv_bias: bool = True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            out: [B, C, H, W]
        """
        B, C, H, W = x.shape

        # Window partition
        windows, orig_H, orig_W = window_partition(x, self.window_size)
        # windows: [B*nW, C, ws, ws]
        nWindows = windows.shape[0]
        ws = self.window_size

        # Flatten spatial: [B*nW, C, ws, ws] -> [B*nW, ws*ws, C]
        tokens = windows.flatten(2).transpose(1, 2)
        tokens = self.norm(tokens)

        # QKV
        qkv = self.qkv(tokens).reshape(nWindows, ws * ws, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, nW, heads, tokens, head_dim]
        q, k, v = qkv.unbind(0)

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(nWindows, ws * ws, C)

        out = self.proj(out)

        # Reshape back: [B*nW, ws*ws, C] -> [B*nW, C, ws, ws]
        out = out.transpose(1, 2).reshape(nWindows, C, ws, ws)

        # Merge windows
        out = window_merge(out, self.window_size, orig_H, orig_W, B)

        # Residual
        return x + out


class DWConvAttention(nn.Module):
    """Depthwise Convolution Attention (Edge profile).

    Lightweight spatial refinement using depthwise separable convolution
    with sigmoid gating. Achieves similar spatial mixing as attention
    with 1/3 the parameters.

    Args:
        dim: Channel dimension.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dw_conv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.bn = nn.BatchNorm2d(dim)
        self.pw_conv = nn.Conv2d(dim, dim, 1, bias=False)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            out: [B, C, H, W]
        """
        feat = self.bn(self.dw_conv(x))
        feat = F.silu(feat)
        feat = self.pw_conv(feat)

        # Channel gating
        g = self.gate(x).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        return x + feat * g


class SEAttention(nn.Module):
    """Squeeze-and-Excitation Attention (Ultra-lite profile).

    Channel-wise attention with minimal parameters.

    Args:
        dim: Channel dimension.
        reduction: Reduction ratio for bottleneck.
    """

    def __init__(self, dim: int, reduction: int = 4):
        super().__init__()
        mid = max(dim // reduction, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dim, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            out: [B, C, H, W]
        """
        w = self.se(x).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        return x * w


def build_attention(attention_type: str, dim: int, num_heads: int = 4,
                    window_size: int = 4) -> nn.Module:
    """Factory function for attention variants."""
    if attention_type == "mhsa":
        return WindowMHSA(dim, num_heads=num_heads, window_size=window_size)
    elif attention_type == "dwconv":
        return DWConvAttention(dim)
    elif attention_type == "se":
        return SEAttention(dim)
    else:
        raise ValueError(f"Unknown attention_type: {attention_type}")


# =============================================================================
# Noise-Conditioned Router
# =============================================================================

class NoiseConditionedRouter(nn.Module):
    """Routes between Mamba and Attention branches based on noise level.

    Extends the NoiseAwareGate concept: instead of routing between
    temporal-state and current-frame, routes between Mamba-path
    (temporal smoothing) and Attention-path (spatial detail).

    High noise -> alpha -> 1 (prefer Mamba for temporal smoothing)
    Low noise  -> alpha -> 0 (prefer Attention for spatial detail)

    Args:
        channels: Feature channel dimension.
        noise_dim: Noise descriptor dimension.
        router_type: 'full' | 'simplified' | 'fixed'
        reduction: Channel reduction ratio.
    """

    def __init__(self, channels: int, noise_dim: int = 32,
                 router_type: str = "simplified", reduction: int = 4):
        super().__init__()
        self.router_type = router_type

        if router_type == "fixed":
            # Fixed 0.5 blending — no learnable params
            return

        mid = max(channels // reduction, 8)
        self.noise_proj = nn.Sequential(
            nn.Linear(noise_dim, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
        )

        if router_type == "full":
            self.feat_stats = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(channels, mid),
                nn.ReLU(inplace=True),
                nn.Linear(mid, channels),
            )
            self.gate_norm = nn.LayerNorm(channels)

    def forward(self, features: torch.Tensor,
                noise_level: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute routing weight alpha.

        Args:
            features: [B, C, H, W]
            noise_level: [B, noise_dim] or None

        Returns:
            alpha: [B, C, 1, 1] routing weight in [0, 1]
        """
        if self.router_type == "fixed":
            B = features.shape[0]
            return torch.full(
                (B, 1, 1, 1), 0.5,
                device=features.device, dtype=features.dtype,
            )

        if noise_level is None:
            # No noise info available: default to balanced
            B = features.shape[0]
            return torch.full(
                (B, 1, 1, 1), 0.5,
                device=features.device, dtype=features.dtype,
            )

        alpha = self.noise_proj(noise_level)  # [B, C]

        if self.router_type == "full":
            feat_gate = self.feat_stats(features)  # [B, C]
            alpha = self.gate_norm(alpha + feat_gate)

        alpha = torch.sigmoid(alpha)  # [B, C]
        return alpha.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]


# =============================================================================
# Hybrid Mamba-Attention Block
# =============================================================================

class MambaAttentionBlock(nn.Module):
    """Hybrid Mamba-Attention block for a single feature scale.

    Parallel dual-branch architecture:
      - Mamba branch: SpatialSSMAdapter for temporal state propagation
      - Attention branch: Spatial attention for local/global refinement
      - Noise-conditioned router: Dynamic alpha blending

    Output = alpha * F_mamba + (1-alpha) * F_attn

    Args:
        channels: Feature channel dimension.
        d_state: SSM hidden state dimension.
        d_conv: SSM local convolution width.
        expand: SSM expansion factor.
        pool_size: SSM spatial pooling target.
        attention_type: 'mhsa' | 'dwconv' | 'se'
        num_heads: MHSA attention heads.
        window_size: MHSA window size.
        noise_dim: Noise descriptor dimension.
        router_type: 'full' | 'simplified' | 'fixed'
        mode: 'hybrid' | 'mamba' | 'attention' (for ablation).
    """

    def __init__(
        self,
        channels: int,
        d_state: int = 8,
        d_conv: int = 4,
        expand: int = 1,
        pool_size: int = 8,
        attention_type: str = "dwconv",
        num_heads: int = 4,
        window_size: int = 4,
        noise_dim: int = 32,
        router_type: str = "simplified",
        mode: str = "hybrid",
    ):
        super().__init__()
        self.mode = mode
        self.channels = channels

        # Mamba branch (reuse existing SpatialSSMAdapter)
        if mode in ("hybrid", "mamba"):
            self.mamba_branch = SpatialSSMAdapter(
                channels=channels,
                d_state=d_state,
                d_conv=d_conv,
                pool_size=pool_size,
            )
            # Override expand factor by rebuilding SSM if needed
            if expand != 2:  # SpatialSSMAdapter defaults expand=2
                from .nas_module import SelectiveSSM
                self.mamba_branch.ssm = SelectiveSSM(
                    d_model=channels,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )

        # Attention branch
        if mode in ("hybrid", "attention"):
            self.attn_branch = build_attention(
                attention_type, channels, num_heads, window_size
            )

        # Noise-conditioned router (only for hybrid mode)
        if mode == "hybrid":
            self.router = NoiseConditionedRouter(
                channels, noise_dim, router_type
            )

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        noise_level: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass.

        Args:
            x: Feature map [B, C, H, W].
            state: Previous SSM state or None.
            noise_level: Noise descriptor [B, noise_dim] or None.

        Returns:
            out: Refined feature map [B, C, H, W].
            new_state: Updated SSM state (None if mode='attention').
        """
        if self.mode == "mamba":
            out, new_state = self.mamba_branch(x, state)
            return out, new_state

        elif self.mode == "attention":
            out = self.attn_branch(x)
            return out, state  # No temporal state update

        else:  # hybrid
            f_mamba, new_state = self.mamba_branch(x, state)
            f_attn = self.attn_branch(x)
            alpha = self.router(x, noise_level)
            out = alpha * f_mamba + (1.0 - alpha) * f_attn
            return out, new_state

    @torch.no_grad()
    def count_params(self) -> dict:
        """Count parameters by component."""
        info = {}
        if hasattr(self, "mamba_branch"):
            info["mamba"] = sum(p.numel() for p in self.mamba_branch.parameters())
        if hasattr(self, "attn_branch"):
            info["attention"] = sum(p.numel() for p in self.attn_branch.parameters())
        if hasattr(self, "router"):
            info["router"] = sum(p.numel() for p in self.router.parameters())
        info["total"] = sum(info.values())
        return info
