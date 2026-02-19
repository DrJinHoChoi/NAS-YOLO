"""
Path Aggregation Feature Pyramid Network (PAFPN) for NAS-YOLO
===============================================================
Standard YOLO neck for multi-scale feature fusion.

Like the backbone, the neck is intentionally kept standard to isolate
the contribution of the NAS Module. Our temporal state modeling operates
on the features AFTER neck processing, before the detection head.

Architecture:
    P5 ──→ Upsample ──→ Concat(P4) ──→ CSP ──→ N4
    P4 ──→ Upsample ──→ Concat(P3) ──→ CSP ──→ N3
    N3 ──→ Downsample → Concat(N4)  ──→ CSP ──→ O4
    N4 ──→ Downsample → Concat(P5)  ──→ CSP ──→ O5

    Output: [O3, O4, O5] — three scales for detection head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from .backbone import ConvBlock, CSPBlock


class PAFPN(nn.Module):
    """Path Aggregation Feature Pyramid Network.

    Args:
        in_channels: List of input channel counts from backbone [C3, C4, C5].
        out_channels: List of output channel counts (typically same as in).
        depth_multiple: Block depth scaling factor.
        width_multiple: Channel width scaling factor.
    """

    def __init__(
        self,
        in_channels: List[int],
        depth_multiple: float = 1.0,
        width_multiple: float = 1.0,
    ):
        super().__init__()
        assert len(in_channels) == 3, "PAFPN expects exactly 3 input scales"

        c3, c4, c5 = in_channels
        n_blocks = max(round(3 * depth_multiple), 1)

        # === Top-down path (FPN) ===
        # P5 → lateral → upsample → concat with P4
        self.lateral_p5 = ConvBlock(c5, c4, 1, 1)
        self.td_csp_p4 = CSPBlock(c4 * 2, c4, n_blocks, shortcut=False)

        # P4 → lateral → upsample → concat with P3
        self.lateral_p4 = ConvBlock(c4, c3, 1, 1)
        self.td_csp_p3 = CSPBlock(c3 * 2, c3, n_blocks, shortcut=False)

        # === Bottom-up path (PAN) ===
        # N3 → downsample → concat with N4
        self.bu_down_p3 = ConvBlock(c3, c3, 3, 2)
        self.bu_csp_p4 = CSPBlock(c3 + c4, c4, n_blocks, shortcut=False)

        # N4 → downsample → concat with P5
        self.bu_down_p4 = ConvBlock(c4, c4, 3, 2)
        self.bu_csp_p5 = CSPBlock(c4 + c5, c5, n_blocks, shortcut=False)

        # Output channel list for downstream
        self.out_channels = [c3, c4, c5]

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """Multi-scale feature fusion.

        Args:
            features: [P3, P4, P5] from backbone.

        Returns:
            fused: [O3, O4, O5] — fused features at three scales.
        """
        p3, p4, p5 = features

        # === Top-down ===
        # P5 → upsample → concat with P4
        td_p5 = self.lateral_p5(p5)
        td_p5_up = F.interpolate(td_p5, size=p4.shape[-2:], mode="nearest")
        td_p4 = self.td_csp_p4(torch.cat([td_p5_up, p4], dim=1))

        # P4 → upsample → concat with P3
        td_p4_lat = self.lateral_p4(td_p4)
        td_p4_up = F.interpolate(td_p4_lat, size=p3.shape[-2:], mode="nearest")
        n3 = self.td_csp_p3(torch.cat([td_p4_up, p3], dim=1))

        # === Bottom-up ===
        # N3 → downsample → concat with N4
        bu_n3 = self.bu_down_p3(n3)
        n4 = self.bu_csp_p4(torch.cat([bu_n3, td_p4], dim=1))

        # N4 → downsample → concat with P5
        bu_n4 = self.bu_down_p4(n4)
        n5 = self.bu_csp_p5(torch.cat([bu_n4, p5], dim=1))

        return [n3, n4, n5]
