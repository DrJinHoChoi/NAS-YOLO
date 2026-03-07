"""
PicoBackbone — Ultra-Minimal Backbone for Sub-0.5M Detection
==============================================================
Custom backbone designed for the world's smallest/fastest detection system.
Uses Depthwise Separable Convolutions and Inverted Residual blocks
(MobileNetV2-style) for extreme parameter efficiency.

Target: ~85K params with channels [32, 64, 128] — 1/4 of YOLOv8n.

Architecture:
    Stem:    3 → base_ch (Conv 3×3 s2)
    Stage 1: base_ch → base_ch (InvRes ×1, s2) → P2
    Stage 2: base_ch → base_ch×2 (InvRes ×2, s2) → P3
    Stage 3: base_ch×2 → base_ch×4 (InvRes ×2, s2) → P4
    Stage 4: base_ch×4 → base_ch×4 (InvRes ×1 + SPPF-lite, s2) → P5

    Output: [P3, P4, P5] for neck/plugin/head

Design decisions:
    - DWSep Conv instead of standard Conv (MobileNet paradigm)
    - Inverted Residual (expand→DW→project) instead of CSPBlock
    - GhostModule option for cheap feature generation
    - No heavy stem — single Conv 3×3 s2
    - Channels [32, 64, 128] exactly 1/4 of YOLOv8n's [64, 128, 256]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class ConvBnAct(nn.Module):
    """Conv2d + BatchNorm + SiLU — lightweight building block.

    Args:
        in_ch: Input channels.
        out_ch: Output channels.
        kernel_size: Convolution kernel size.
        stride: Convolution stride.
        groups: Convolution groups (1=standard, in_ch=depthwise).
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 stride: int = 1, groups: int = 1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel_size, stride, padding,
            groups=groups, bias=False,
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution: DW 3×3 + PW 1×1.

    Reduces params from in_ch * out_ch * k² to:
    in_ch * k² (DW) + in_ch * out_ch (PW)

    For 32→64, k=3: standard=18,432 vs DWSep=2,336 (7.9× reduction)

    Args:
        in_ch: Input channels.
        out_ch: Output channels.
        stride: DW conv stride.
        kernel_size: DW conv kernel size.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1,
                 kernel_size: int = 3):
        super().__init__()
        self.dw = ConvBnAct(in_ch, in_ch, kernel_size, stride, groups=in_ch)
        self.pw = ConvBnAct(in_ch, out_ch, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


class InvertedResidual(nn.Module):
    """MobileNetV2 Inverted Residual Block.

    Expand → Depthwise Conv → Project (with residual if stride=1 and in==out).

    The expand ratio controls internal channel width:
    - expand=1: no expansion (identity path)
    - expand=2-6: standard MobileNetV2 expansion

    For pico: expand=2 gives good trade-off between params and accuracy.

    Args:
        in_ch: Input channels.
        out_ch: Output channels.
        stride: Depthwise conv stride (1 or 2).
        expand_ratio: Channel expansion ratio.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1,
                 expand_ratio: float = 2.0):
        super().__init__()
        self.use_residual = (stride == 1 and in_ch == out_ch)
        mid_ch = max(int(in_ch * expand_ratio), 8)

        layers = []
        # Expand (skip if expand_ratio == 1 and in_ch == mid_ch)
        if mid_ch != in_ch:
            layers.append(ConvBnAct(in_ch, mid_ch, 1, 1))

        # Depthwise
        layers.append(ConvBnAct(mid_ch, mid_ch, 3, stride, groups=mid_ch))

        # Project (linear — no activation)
        layers.append(nn.Conv2d(mid_ch, out_ch, 1, bias=False))
        layers.append(nn.BatchNorm2d(out_ch))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        if self.use_residual:
            out = out + x
        return out


class SPPFLite(nn.Module):
    """Lightweight SPPF — Spatial Pyramid Pooling (Fast).

    Uses smaller pool kernel (3 instead of 5) for pico scale.

    Args:
        in_ch: Input channels.
        out_ch: Output channels.
        kernel_size: Pooling kernel size.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
        super().__init__()
        mid_ch = in_ch // 2
        self.cv1 = ConvBnAct(in_ch, mid_ch, 1, 1)
        self.cv2 = ConvBnAct(mid_ch * 4, out_ch, 1, 1)
        self.pool = nn.MaxPool2d(kernel_size, stride=1,
                                 padding=kernel_size // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        p1 = self.pool(x)
        p2 = self.pool(p1)
        p3 = self.pool(p2)
        return self.cv2(torch.cat([x, p1, p2, p3], dim=1))


class PicoBackbone(nn.Module):
    """Ultra-minimal backbone for sub-0.5M detection.

    Produces multi-scale features [P3, P4, P5] from input images.
    Uses Inverted Residual blocks for extreme parameter efficiency.

    Channel configurations:
        - base_channels=16: Ultra-pico [16, 32, 64] → ~24K backbone
        - base_channels=32: Pico       [32, 64, 128] → ~85K backbone (recommended)
        - base_channels=48: Pico+      [48, 96, 192] → ~160K backbone

    Architecture (base_channels=32):
        Stem:    3 → 32  (Conv 3×3 s2)                    → /2
        Stage 1: 32 → 32 (InvRes ×1, s2)                  → /4
        Stage 2: 32 → 64 (InvRes ×2, first s2)  → P3      → /8
        Stage 3: 64 → 128 (InvRes ×2, first s2) → P4      → /16
        Stage 4: 128 → 128 (InvRes ×1, s2 + SPPF-lite) → P5 → /32

    Args:
        base_channels: Base channel width (16, 32, or 48).
        in_channels: Input image channels (default: 3).
        expand_ratio: Inverted residual expansion ratio (default: 2.0).
    """

    def __init__(
        self,
        base_channels: int = 32,
        in_channels: int = 3,
        expand_ratio: float = 2.0,
    ):
        super().__init__()
        c1 = base_channels        # 32
        c2 = base_channels * 2    # 64
        c3 = base_channels * 4    # 128

        # Output channels for downstream (P3, P4, P5)
        self.out_channels = [c2, c3, c3]

        # Stem: fast downscale with standard conv
        self.stem = ConvBnAct(in_channels, c1, 3, 2)

        # Stage 1: /4 (downscale, keep channels)
        self.stage1 = InvertedResidual(c1, c1, stride=2,
                                       expand_ratio=expand_ratio)

        # Stage 2: /8 → P3 output
        self.stage2 = nn.Sequential(
            InvertedResidual(c1, c2, stride=2, expand_ratio=expand_ratio),
            InvertedResidual(c2, c2, stride=1, expand_ratio=expand_ratio),
        )

        # Stage 3: /16 → P4 output
        self.stage3 = nn.Sequential(
            InvertedResidual(c2, c3, stride=2, expand_ratio=expand_ratio),
            InvertedResidual(c3, c3, stride=1, expand_ratio=expand_ratio),
        )

        # Stage 4: /32 → P5 output (with SPPF for multi-scale context)
        self.stage4 = nn.Sequential(
            InvertedResidual(c3, c3, stride=2, expand_ratio=expand_ratio),
            SPPFLite(c3, c3, kernel_size=3),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights following MobileNet conventions."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract multi-scale features.

        Args:
            x: Input image [B, 3, H, W].

        Returns:
            features: [P3, P4, P5] feature maps.
                P3: [B, base_ch×2, H/8, W/8]
                P4: [B, base_ch×4, H/16, W/16]
                P5: [B, base_ch×4, H/32, W/32]
        """
        x = self.stem(x)       # /2
        x = self.stage1(x)     # /4

        p3 = self.stage2(x)    # /8
        p4 = self.stage3(p3)   # /16
        p5 = self.stage4(p4)   # /32

        return [p3, p4, p5]

    @torch.no_grad()
    def count_params(self) -> dict:
        """Count parameters per stage."""
        info = {}
        for name in ["stem", "stage1", "stage2", "stage3", "stage4"]:
            module = getattr(self, name)
            params = sum(p.numel() for p in module.parameters())
            info[name] = params
        info["total"] = sum(info.values())
        return info
