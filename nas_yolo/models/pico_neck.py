"""
PicoNeck — Lightweight Feature Pyramid for Sub-0.5M Detection
================================================================
FPN-only neck (no bottom-up PAN) using depthwise separable convolutions.

Standard PAFPN has both top-down FPN and bottom-up PAN paths, roughly
doubling parameters and latency. For pico-scale detection, we keep only
the top-down FPN path — sufficient for single-shot detection.

Architecture:
    P5 → 1×1 Conv → Upsample → Concat(P4) → DWSep → N4
    N4 → 1×1 Conv → Upsample → Concat(P3) → DWSep → N3

    Output: [N3, N4, P5] at original backbone channels

Compared to full PAFPN (~300K for YOLOv8n):
    PicoNeck: ~50K (6× reduction)

Design rationale:
    - SA-SSM plugin already performs cross-scale feature refinement
    - PAN bottom-up path is partially redundant with plugin's processing
    - DWSep convolutions for minimal fusion overhead
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from .pico_backbone import ConvBnAct, DepthwiseSeparableConv


class PicoNeck(nn.Module):
    """Lightweight FPN-only neck with depthwise separable convolutions.

    Takes [P3, P4, P5] from PicoBackbone and performs top-down feature
    fusion to produce [N3, N4, P5] for the plugin and detection head.

    Args:
        in_channels: List of input channel counts [C3, C4, C5].
                     Example: [64, 128, 128] for base_channels=32.
    """

    def __init__(self, in_channels: List[int]):
        super().__init__()
        assert len(in_channels) == 3, "PicoNeck expects exactly 3 scales"

        c3, c4, c5 = in_channels

        # Top-down path: P5 → P4
        self.lateral_p5 = ConvBnAct(c5, c4, 1, 1)  # Channel reduction
        self.td_fuse_p4 = DepthwiseSeparableConv(c4 * 2, c4, stride=1)

        # Top-down path: P4 → P3
        self.lateral_p4 = ConvBnAct(c4, c3, 1, 1)  # Channel reduction
        self.td_fuse_p3 = DepthwiseSeparableConv(c3 * 2, c3, stride=1)

        # Output channels (same as input — we fuse but don't change dims)
        self.out_channels = in_channels

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """Top-down feature fusion.

        Args:
            features: [P3, P4, P5] from backbone.

        Returns:
            fused: [N3, N4, P5] — fused features at three scales.
        """
        p3, p4, p5 = features

        # P5 → upsample → concat with P4 → fuse
        td_p5 = self.lateral_p5(p5)
        td_p5_up = F.interpolate(td_p5, size=p4.shape[-2:], mode="nearest")
        n4 = self.td_fuse_p4(torch.cat([td_p5_up, p4], dim=1))

        # N4 → upsample → concat with P3 → fuse
        td_n4 = self.lateral_p4(n4)
        td_n4_up = F.interpolate(td_n4, size=p3.shape[-2:], mode="nearest")
        n3 = self.td_fuse_p3(torch.cat([td_n4_up, p3], dim=1))

        # P5 passes through unchanged (already has global context from SPPF)
        return [n3, n4, p5]

    @torch.no_grad()
    def count_params(self) -> dict:
        """Count parameters per component."""
        return {
            "lateral_p5": sum(p.numel() for p in self.lateral_p5.parameters()),
            "td_fuse_p4": sum(p.numel() for p in self.td_fuse_p4.parameters()),
            "lateral_p4": sum(p.numel() for p in self.lateral_p4.parameters()),
            "td_fuse_p3": sum(p.numel() for p in self.td_fuse_p3.parameters()),
            "total": sum(p.numel() for p in self.parameters()),
        }
