"""
Noise-Aware Gating Module
==========================
Core contribution C2: Adaptive feature modulation conditioned on input noise.

Key idea: Instead of treating all frames equally, we estimate the noise
characteristics of each input frame (spectral energy distribution,
illumination level, blur magnitude) and use this to control how much
the temporal state should override the current frame's features.

The SpectralNoiseEstimator operates in the frequency domain (DCT-based,
not FFT, for efficiency on edge devices) to produce a compact noise
descriptor that drives the gating mechanism.

This enables:
  - Graceful degradation under increasing corruption severity
  - Automatic fallback to temporal memory under heavy noise
  - Preservation of spatial detail under clean conditions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional, Union


class SpectralNoiseEstimator(nn.Module):
    """Estimates input noise characteristics from the frequency domain.

    Uses a lightweight DCT-based analysis to extract noise descriptors without
    requiring explicit noise labels during training. The key insight is that
    natural images have characteristic spectral falloff (1/f), while noise
    tends to elevate high-frequency energy.

    Architecture:
        Input image → Spatial downscale → DCT approximation → MLP → Noise descriptor

    The noise descriptor encodes:
        - High-frequency energy ratio (noise indicator)
        - Illumination level (mean luminance)
        - Blur magnitude (high-freq attenuation)
        - Overall signal quality estimate

    Args:
        in_channels: Number of input image channels (3 for RGB).
        noise_dim: Dimension of the output noise descriptor.
        spatial_size: Size to which input is downscaled for analysis.
    """

    def __init__(self, in_channels: int = 3, noise_dim: int = 32, spatial_size: int = 16):
        super().__init__()
        self.noise_dim = noise_dim
        self.spatial_size = spatial_size

        # Lightweight feature extractor (operates on downscaled input)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(spatial_size),
        )

        # DCT basis functions (precomputed, non-learnable)
        # Using Type-II DCT for frequency analysis
        self.register_buffer("dct_basis", self._build_dct_basis(spatial_size))

        # Precompute frequency band masks (avoids Python loops in forward pass)
        low_mask, mid_mask, high_mask = self._build_frequency_masks(spatial_size)
        self.register_buffer("low_mask", low_mask)
        self.register_buffer("mid_mask", mid_mask)
        self.register_buffer("high_mask", high_mask)

        # Frequency band analysis: split spectrum into low/mid/high
        self.freq_analyzer = nn.Sequential(
            nn.Linear(32 * 3, 64),  # 3 frequency bands × 32 channels
            nn.ReLU(inplace=True),
            nn.Linear(64, noise_dim),
        )

        # Illumination & blur branch (spatial domain)
        self.spatial_analyzer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, noise_dim),
        )

        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(noise_dim * 2, noise_dim),
            nn.LayerNorm(noise_dim),
        )

    def _build_dct_basis(self, N: int) -> torch.Tensor:
        """Build 1D DCT-II basis matrix of size N×N."""
        basis = torch.zeros(N, N)
        for k in range(N):
            for n in range(N):
                if k == 0:
                    basis[k, n] = 1.0 / math.sqrt(N)
                else:
                    basis[k, n] = math.sqrt(2.0 / N) * math.cos(
                        math.pi * (2 * n + 1) * k / (2 * N)
                    )
        return basis  # [N, N]

    def _dct2d(self, x: torch.Tensor) -> torch.Tensor:
        """Apply 2D DCT via separable 1D transforms.

        Args:
            x: Input tensor [B, C, H, W] where H=W=spatial_size.

        Returns:
            X_dct: DCT coefficients [B, C, H, W].
        """
        # Row-wise DCT
        X = torch.matmul(self.dct_basis, x)
        # Column-wise DCT
        X = torch.matmul(X, self.dct_basis.t())
        return X

    @staticmethod
    def _build_frequency_masks(N: int):
        """Build low/mid/high frequency band masks for NxN DCT grid."""
        low_mask = torch.zeros(N, N)
        mid_mask = torch.zeros(N, N)
        high_mask = torch.zeros(N, N)
        max_dist = math.sqrt(N * N + N * N)

        for i in range(N):
            for j in range(N):
                dist = math.sqrt(i * i + j * j) / max_dist
                if dist < 0.33:
                    low_mask[i, j] = 1.0
                elif dist < 0.66:
                    mid_mask[i, j] = 1.0
                else:
                    high_mask[i, j] = 1.0

        return low_mask, mid_mask, high_mask

    def _extract_frequency_bands(self, dct_coeffs: torch.Tensor) -> torch.Tensor:
        """Extract energy from low/mid/high frequency bands.

        Uses precomputed masks registered as buffers (no per-forward computation).

        Args:
            dct_coeffs: DCT coefficients [B, C, H, W].

        Returns:
            band_energies: [B, C*3] concatenated band energies.
        """
        energy = dct_coeffs.abs().pow(2)

        # Use precomputed masks (registered buffers, auto device/dtype)
        low_energy = (energy * self.low_mask).sum(dim=(-2, -1)) / (self.low_mask.sum() + 1e-8)
        mid_energy = (energy * self.mid_mask).sum(dim=(-2, -1)) / (self.mid_mask.sum() + 1e-8)
        high_energy = (energy * self.high_mask).sum(dim=(-2, -1)) / (self.high_mask.sum() + 1e-8)

        return torch.cat([low_energy, mid_energy, high_energy], dim=-1)  # [B, C*3]

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Estimate noise descriptor from input image.

        Args:
            image: Input image [B, 3, H, W] (any resolution).

        Returns:
            noise_desc: Noise descriptor [B, noise_dim].
        """
        # Encode to fixed spatial size
        feat = self.encoder(image)  # [B, 32, spatial_size, spatial_size]

        # Frequency domain analysis
        dct_coeffs = self._dct2d(feat)
        freq_bands = self._extract_frequency_bands(dct_coeffs)  # [B, 32*3]
        freq_desc = self.freq_analyzer(freq_bands)  # [B, noise_dim]

        # Spatial domain analysis (illumination, global statistics)
        spatial_desc = self.spatial_analyzer(feat)  # [B, noise_dim]

        # Fuse descriptors
        noise_desc = self.fusion(torch.cat([freq_desc, spatial_desc], dim=-1))

        return noise_desc


class NoiseAwareGate(nn.Module):
    """Noise-conditioned gating for adaptive temporal-spatial blending.

    Produces a per-channel, spatially-varying gate weight that controls
    how much the SSM temporal output should override the current frame features.

    The gate is conditioned on:
      1. The noise descriptor from SpectralNoiseEstimator
      2. The current feature statistics (channel-wise mean/var)

    Under high noise: gate → 1 (trust temporal state)
    Under low noise:  gate → 0 (trust current frame)

    This is fundamentally different from standard attention or SE blocks
    because it is explicitly conditioned on estimated noise, not just
    feature statistics.

    Args:
        channels: Number of feature channels at this scale.
        noise_dim: Dimension of the noise descriptor (default: 32).
        reduction: Channel reduction ratio for efficiency.
    """

    def __init__(self, channels: int, noise_dim: int = 32, reduction: int = 4):
        super().__init__()
        self.channels = channels

        # Noise condition projection
        self.noise_proj = nn.Sequential(
            nn.Linear(noise_dim, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
        )

        # Feature statistics branch
        self.feat_stats = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
        )

        # Spatial gate (lightweight depthwise conv)
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
        )

        # Final gate activation
        self.gate_norm = nn.LayerNorm(channels)

    def forward(
        self,
        features: torch.Tensor,
        noise_level: torch.Tensor,
    ) -> torch.Tensor:
        """Compute noise-aware gate weights.

        Args:
            features: Current frame features [B, C, H, W].
            noise_level: Noise descriptor [B, noise_dim].

        Returns:
            gate: Per-channel gate weight [B, C, 1, 1] in [0, 1].
        """
        B, C, H, W = features.shape

        # Channel gate from noise condition
        noise_gate = self.noise_proj(noise_level)  # [B, C]

        # Channel gate from feature statistics
        feat_gate = self.feat_stats(features)  # [B, C]

        # Combine: noise condition modulates feature-based gate
        combined = self.gate_norm(noise_gate + feat_gate)  # [B, C]
        gate = torch.sigmoid(combined)  # [B, C]

        # Reshape for broadcasting: [B, C] → [B, C, 1, 1]
        gate = gate.unsqueeze(-1).unsqueeze(-1)

        return gate


class MultiScaleNoiseEstimator(nn.Module):
    """Estimates noise at multiple feature scales for hierarchical gating.

    Produces noise descriptors at each detection scale by combining the
    global noise estimate with scale-specific local statistics. This allows
    different scales to respond differently to noise — e.g., small object
    features (high-res scale) may be more noise-sensitive than large object
    features (low-res scale).

    Args:
        in_channels: Input image channels (3 for RGB).
        channels_list: List of channel counts per feature scale.
        noise_dim: Noise descriptor dimension.
    """

    def __init__(
        self,
        in_channels: int = 3,
        channels_list: list = None,
        noise_dim: int = 32,
    ):
        super().__init__()
        self.global_estimator = SpectralNoiseEstimator(
            in_channels=in_channels,
            noise_dim=noise_dim,
        )

        if channels_list is not None:
            self.scale_adapters = nn.ModuleList([
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(ch, noise_dim),
                )
                for ch in channels_list
            ])
            self.scale_fusion = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(noise_dim * 2, noise_dim),
                    nn.LayerNorm(noise_dim),
                )
                for _ in channels_list
            ])
        else:
            self.scale_adapters = None
            self.scale_fusion = None

    def forward(
        self, image: torch.Tensor, features: list = None
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Estimate noise descriptors.

        Args:
            image: Input image [B, 3, H, W].
            features: Optional list of multi-scale features for scale-specific estimation.

        Returns:
            If features is None: global noise descriptor [B, noise_dim].
            If features provided: list of scale-specific descriptors.
        """
        global_noise = self.global_estimator(image)

        if features is None or self.scale_adapters is None:
            return global_noise

        scale_descriptors = []
        for i, feat in enumerate(features):
            local = self.scale_adapters[i](feat)
            fused = self.scale_fusion[i](torch.cat([global_noise, local], dim=-1))
            scale_descriptors.append(fused)

        return scale_descriptors
