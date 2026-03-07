"""
Self-Attention augmented State Space Model (SA-SSM) / Hybrid Mamba-Attention Block
====================================================================================
Combines Selective SSM (Mamba) with lightweight spatial attention for
robust feature refinement. Noise-conditioned routing dynamically
balances temporal smoothing (Mamba) vs spatial detail (Attention).

This module realizes the SA-SSM architecture: SSM handles temporal/sequential
processing while Self-Attention handles spatial detail, with noise-conditioned
routing that dynamically specializes each branch based on input degradation.

The SA-SSM structure is domain-agnostic: the same combination of LTI-stable
SSM temporal processing and attention-based spatial/spectral detail has been
validated in both audio (keyword spotting) and vision (object detection)
domains, confirming that structural noise-robustness is a fundamental
property of the SA-SSM architecture, not a domain-specific artifact.

Supports 5 profiles for edge-to-cloud deployment:
  - Standard: MHSA 4 heads + expand=2 SSM (server)
  - Lite:     MHSA 2 heads + expand=1 SSM (desktop GPU)
  - Edge:     DWConv attention + expand=1 SSM (Jetson Orin) -- DEFAULT
  - Ultra-lite: SE-block + expand=1 d_state=4 SSM (Jetson Nano)
  - Tiny:     SE-block + expand=1 d_state=4 d_conv=2 SSM (Smart Glasses / XR2)

References:
  - Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces", 2023
  - Liu et al., "Swin Transformer: Hierarchical Vision Transformer", ICCV 2021
  - Hatamizadeh et al., "MambaVision: A Hybrid Mamba-Transformer Vision Backbone", 2024
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from typing import Optional, Tuple

from .nas_module import SpatialSSMAdapter


# =============================================================================
# Profile Presets
# =============================================================================

PROFILE_CONFIGS = {
    "standard": {
        "expand": 2, "d_state": 16, "attention_type": "mhsa",
        "num_heads": 4, "window_size": 4, "router_type": "full",
        "gate_min": 0.05, "learnable_floor": False, "weight_sharing": False,
    },
    "lite": {
        "expand": 1, "d_state": 8, "attention_type": "mhsa",
        "num_heads": 2, "window_size": 4, "router_type": "full",
        "gate_min": 0.05, "learnable_floor": False, "weight_sharing": False,
    },
    "edge": {
        "expand": 1, "d_state": 8, "attention_type": "dwconv",
        "num_heads": 1, "window_size": 0, "router_type": "simplified",
        "gate_min": 0.10, "learnable_floor": False, "weight_sharing": True,
    },
    "ultra-lite": {
        "expand": 1, "d_state": 4, "attention_type": "se",
        "num_heads": 1, "window_size": 0, "router_type": "fixed",
        "gate_min": 0.15, "learnable_floor": False, "weight_sharing": True,
    },
    "tiny": {
        "expand": 1, "d_state": 4, "attention_type": "se",
        "num_heads": 1, "window_size": 0, "router_type": "fixed",
        "gate_min": 0.20, "learnable_floor": False, "weight_sharing": True,
        "d_conv": 2,
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
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(B * nH * nW, C, window_size, window_size)
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
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous().reshape(B, C, Hp, Wp)

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
                 router_type: str = "simplified", reduction: int = 4,
                 gate_min: float = 0.05, learnable_floor: bool = False):
        super().__init__()
        self.router_type = router_type
        self.learnable_floor = learnable_floor

        if router_type == "fixed":
            # Fixed 0.5 blending — no learnable params, no floor needed
            return

        # Gate floor: ensures alpha >= gate_min
        if learnable_floor and gate_min > 0:
            init_raw = math.log(gate_min / (1.0 - gate_min + 1e-8))
            self._gate_min_raw = nn.Parameter(torch.tensor(init_raw))
            self.max_floor = 0.20
        else:
            self.register_buffer("_gate_min_fixed", torch.tensor(float(gate_min)))

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

        # Apply gate floor: alpha = gate_min + (1 - gate_min) * sigmoid(alpha)
        if self.learnable_floor and hasattr(self, '_gate_min_raw'):
            gate_min = torch.sigmoid(self._gate_min_raw).clamp(max=self.max_floor)
        else:
            gate_min = self._gate_min_fixed
        alpha = gate_min + (1.0 - gate_min) * torch.sigmoid(alpha)  # [B, C]
        return alpha.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]


# =============================================================================
# Hybrid Mamba-Attention Block
# =============================================================================

class MambaAttentionBlock(nn.Module):
    """Self-Attention augmented State Space Model (SA-SSM) block.

    Also known as: Hybrid Mamba-Attention block.

    Parallel dual-branch architecture:
      - Mamba branch (SSM): Temporal state propagation via LTI-stable SSM.
        The SSM A-matrix provides structural low-pass filtering with
        mathematically guaranteed BIBO stability (A = -exp(A_log) < 0).
      - Attention branch: Spatial detail capture via Q*K^T structure.
        Variants: MHSA (standard), DWConv (edge), SE (ultra-lite).
      - Noise-conditioned router: Dynamic alpha blending based on noise level.

    Output = alpha * F_mamba + (1-alpha) * F_attn

    Structural Properties (non-learned, domain-agnostic):
      1. LTI backbone: SSM A-matrix eigenvalues provide structural low-pass
         filtering independent of training data distribution
      2. Attention locality: Q*K^T structure captures spatial relationships
         by architectural design, not by learning
      3. Predetermined specialization: SSM=temporal axis, Attention=spatial
         axis by architecture (not by training convergence)
      4. Cross-domain: Same SA-SSM structure applies to audio (KWS: keyword
         spotting) and vision (detection), confirming domain-agnostic
         structural noise robustness

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

    # Non-learned structural properties of the SA-SSM architecture
    STRUCTURAL_PROPERTIES = {
        "lti_backbone": "A-matrix eigenvalues provide structural low-pass filtering",
        "attention_locality": "Q*K^T captures spatial relationships by architecture",
        "predetermined_specialization": "SSM=temporal, Attention=spatial by design",
        "cross_domain": "Same SA-SSM applies to audio (KWS) and vision (detection)",
    }

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
        gate_min: float = 0.05,
        learnable_floor: bool = False,
        use_external_router: bool = False,
        gradient_checkpointing: bool = False,
        use_parallel_scan: bool = False,
    ):
        super().__init__()
        self.mode = mode
        self.channels = channels
        self.use_external_router = use_external_router
        self.gradient_checkpointing = gradient_checkpointing

        # Mamba branch (reuse existing SpatialSSMAdapter)
        if mode in ("hybrid", "mamba"):
            self.mamba_branch = SpatialSSMAdapter(
                channels=channels,
                d_state=d_state,
                d_conv=d_conv,
                pool_size=pool_size,
                use_parallel_scan=use_parallel_scan,
            )
            # Override expand factor by rebuilding SSM if needed
            if expand != 2:  # SpatialSSMAdapter defaults expand=2
                from .nas_module import SelectiveSSM
                self.mamba_branch.ssm = SelectiveSSM(
                    d_model=channels,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    use_parallel_scan=use_parallel_scan,
                )

        # Attention branch
        if mode in ("hybrid", "attention"):
            self.attn_branch = build_attention(
                attention_type, channels, num_heads, window_size
            )

        # Noise-conditioned router (only for hybrid mode, skip if external router is used)
        if mode == "hybrid" and not use_external_router:
            self.router = NoiseConditionedRouter(
                channels, noise_dim, router_type,
                gate_min=gate_min, learnable_floor=learnable_floor,
            )

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        noise_level: Optional[torch.Tensor] = None,
        external_alpha: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass.

        Args:
            x: Feature map [B, C, H, W].
            state: Previous SSM state or None.
            noise_level: Noise descriptor [B, noise_dim] or None.
            external_alpha: Pre-computed routing weight from shared router [B, C, 1, 1] or None.

        Returns:
            out: Refined feature map [B, C, H, W].
            new_state: Updated SSM state (None if mode='attention').
        """
        if self.mode == "mamba":
            if self.gradient_checkpointing and self.training:
                out, new_state = cp.checkpoint(
                    self.mamba_branch, x, state, use_reentrant=False,
                )
            else:
                out, new_state = self.mamba_branch(x, state)
            return out, new_state

        elif self.mode == "attention":
            if self.gradient_checkpointing and self.training:
                out = cp.checkpoint(
                    self.attn_branch, x, use_reentrant=False,
                )
            else:
                out = self.attn_branch(x)
            return out, state  # No temporal state update

        else:  # hybrid
            if self.gradient_checkpointing and self.training:
                f_mamba, new_state = cp.checkpoint(
                    self.mamba_branch, x, state, use_reentrant=False,
                )
                f_attn = cp.checkpoint(
                    self.attn_branch, x, use_reentrant=False,
                )
            else:
                f_mamba, new_state = self.mamba_branch(x, state)
                f_attn = self.attn_branch(x)
            if external_alpha is not None:
                alpha = external_alpha
            else:
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


# =============================================================================
# SA-SSM Alias (Self-Attention augmented State Space Model)
# =============================================================================
# The MambaAttentionBlock IS the SA-SSM architecture. This alias formalizes
# the naming to connect with the broader SSM+Attention literature across
# domains (audio/KWS and vision/detection).

SA_SSM = MambaAttentionBlock
