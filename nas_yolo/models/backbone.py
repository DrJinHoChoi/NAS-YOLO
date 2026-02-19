"""
CSPDarknet Backbone for NAS-YOLO
=================================
Standard YOLO backbone (CSPDarknet53 variant) providing multi-scale features.

We intentionally use a well-known backbone architecture to isolate the
contribution of the NAS Module. The backbone is NOT our contribution —
the novelty lies in what happens AFTER feature extraction.

Supports multiple model scales:
  - Nano  (n): width=0.25, depth=0.33 — for edge deployment
  - Small (s): width=0.50, depth=0.33 — balanced
  - Medium(m): width=0.75, depth=0.67 — accuracy-focused
  - Large (l): width=1.00, depth=1.00 — full capacity

Args:
    width_multiple: Channel width scaling factor.
    depth_multiple: Block depth scaling factor.
    in_channels: Input image channels (default: 3).
    out_indices: Which stages to output for FPN (default: [2, 3, 4] = P3/P4/P5).
"""

import torch
import torch.nn as nn
from typing import List, Tuple


class ConvBlock(nn.Module):
    """Standard Conv2d + BatchNorm + SiLU block.

    Args:
        in_ch: Input channels.
        out_ch: Output channels.
        kernel_size: Convolution kernel size.
        stride: Convolution stride.
        groups: Convolution groups (1 for standard, in_ch for depthwise).
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


class Bottleneck(nn.Module):
    """Standard bottleneck block with optional shortcut.

    Args:
        in_ch: Input channels.
        out_ch: Output channels.
        shortcut: Whether to use residual connection.
        expansion: Bottleneck expansion ratio.
    """

    def __init__(self, in_ch: int, out_ch: int, shortcut: bool = True,
                 expansion: float = 0.5):
        super().__init__()
        hidden_ch = int(out_ch * expansion)
        self.cv1 = ConvBlock(in_ch, hidden_ch, 1, 1)
        self.cv2 = ConvBlock(hidden_ch, out_ch, 3, 1)
        self.use_shortcut = shortcut and in_ch == out_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.cv2(self.cv1(x))
        return x + out if self.use_shortcut else out


class CSPBlock(nn.Module):
    """Cross Stage Partial block.

    Splits input channels, applies bottleneck stack to one branch,
    concatenates both branches, and fuses with 1x1 conv.

    Args:
        in_ch: Input channels.
        out_ch: Output channels.
        n_bottlenecks: Number of bottleneck blocks.
        shortcut: Whether bottlenecks use residual connections.
        expansion: Bottleneck expansion ratio.
    """

    def __init__(self, in_ch: int, out_ch: int, n_bottlenecks: int = 1,
                 shortcut: bool = True, expansion: float = 0.5):
        super().__init__()
        hidden_ch = int(out_ch * expansion)
        self.cv1 = ConvBlock(in_ch, hidden_ch, 1, 1)
        self.cv2 = ConvBlock(in_ch, hidden_ch, 1, 1)
        self.cv3 = ConvBlock(2 * hidden_ch, out_ch, 1, 1)
        self.bottlenecks = nn.Sequential(*[
            Bottleneck(hidden_ch, hidden_ch, shortcut=shortcut, expansion=1.0)
            for _ in range(n_bottlenecks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch1 = self.bottlenecks(self.cv1(x))
        branch2 = self.cv2(x)
        return self.cv3(torch.cat([branch1, branch2], dim=1))


class SPPFBlock(nn.Module):
    """Spatial Pyramid Pooling — Fast (SPPF).

    Uses sequential 5×5 max pooling (equivalent to multi-scale pooling
    but faster) to capture multi-scale context.

    Args:
        in_ch: Input channels.
        out_ch: Output channels.
        kernel_size: Pooling kernel size (default: 5).
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 5):
        super().__init__()
        hidden_ch = in_ch // 2
        self.cv1 = ConvBlock(in_ch, hidden_ch, 1, 1)
        self.cv2 = ConvBlock(hidden_ch * 4, out_ch, 1, 1)
        self.pool = nn.MaxPool2d(kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        p1 = self.pool(x)
        p2 = self.pool(p1)
        p3 = self.pool(p2)
        return self.cv2(torch.cat([x, p1, p2, p3], dim=1))


class CSPDarknet(nn.Module):
    """CSPDarknet backbone producing multi-scale features.

    Architecture:
        Stem → Stage1(P1) → Stage2(P2) → Stage3(P3) → Stage4(P4) → Stage5(P5/SPPF)

    Output features are from stages indexed by out_indices (default: P3, P4, P5).

    Args:
        width_multiple: Channel width scaling factor.
        depth_multiple: Block depth scaling factor.
        in_channels: Input image channels.
        out_indices: Stage indices to output (0-indexed from Stage1).
    """

    # Base channel/depth config (corresponds to width/depth_multiple = 1.0)
    BASE_CHANNELS = [64, 128, 256, 512, 1024]
    BASE_DEPTHS = [1, 2, 8, 8, 4]

    def __init__(
        self,
        width_multiple: float = 1.0,
        depth_multiple: float = 1.0,
        in_channels: int = 3,
        out_indices: Tuple[int, ...] = (2, 3, 4),
    ):
        super().__init__()
        self.out_indices = out_indices

        # Scale channels and depths
        channels = [max(int(c * width_multiple), 8) for c in self.BASE_CHANNELS]
        depths = [max(round(d * depth_multiple), 1) for d in self.BASE_DEPTHS]

        # Store output channel counts for downstream modules
        self.out_channels = [channels[i] for i in out_indices]

        # Stem
        self.stem = ConvBlock(in_channels, channels[0], 6, 2)

        # Stages
        self.stages = nn.ModuleList()
        in_ch = channels[0]
        for i in range(5):
            stage = nn.Sequential(
                ConvBlock(in_ch, channels[i], 3, 2),  # downsample
                CSPBlock(channels[i], channels[i], depths[i]),
            )
            if i == 4:
                # Add SPPF to last stage
                stage = nn.Sequential(
                    ConvBlock(in_ch, channels[i], 3, 2),
                    CSPBlock(channels[i], channels[i], depths[i]),
                    SPPFBlock(channels[i], channels[i]),
                )
            self.stages.append(stage)
            in_ch = channels[i]

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract multi-scale features.

        Args:
            x: Input image [B, 3, H, W].

        Returns:
            features: List of feature maps at selected scales.
        """
        x = self.stem(x)
        outputs = []

        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i in self.out_indices:
                outputs.append(x)

        return outputs
