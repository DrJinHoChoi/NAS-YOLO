"""
NAS Plugin — Standalone Plug-in Module for Any YOLO Detector
==============================================================
Drop-in module between any YOLO's neck and detection head.

Usage:
    # After loading any YOLO model
    backbone_features = yolo.backbone(images)
    neck_features = yolo.neck(backbone_features)

    # Insert NASPlugin
    enhanced_features, new_states, conf_weights = plugin(
        neck_features, images=images
    )

    # Continue to detection head
    outputs = yolo.head(enhanced_features)

The plugin is self-contained: it includes noise estimation, hybrid
Mamba-Attention blocks, confidence refinement, and temporal state
management. No modifications to the host YOLO model are required.

Profiles (default: edge):
    - standard: 0.92M overhead (26%, server)
    - lite:     0.60M overhead (17%, desktop)
    - edge:     0.39M overhead (11%, Jetson Orin) ← DEFAULT
    - ultra-lite: 0.22M overhead (6%, Jetson Nano)
"""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple

from .mamba_attention import MambaAttentionBlock, PROFILE_CONFIGS
from .noise_gate import SpectralNoiseEstimator
from .temporal_buffer import TemporalFeatureBuffer


class NASPlugin(nn.Module):
    """Standalone Noise-Aware State plugin for any YOLO detector.

    Processes multi-scale features from any detector's neck through
    hybrid Mamba-Attention blocks with noise-conditioned routing.

    Args:
        channels_list: Channel count per feature scale, e.g. [64, 128, 256].
        profile: Deployment profile ('standard', 'lite', 'edge', 'ultra-lite').
        mode: Block mode ('hybrid', 'mamba', 'attention') for ablation.
        use_noise_gate: Enable noise estimation and routing.
        noise_dim: Noise descriptor dimension.
        in_channels: Input image channels (for noise estimator).
        ema_momentum: Temporal confidence smoothing momentum.
        pool_size: SSM spatial pooling target.
    """

    def __init__(
        self,
        channels_list: List[int],
        profile: str = "edge",
        mode: str = "hybrid",
        use_noise_gate: bool = True,
        noise_dim: int = 32,
        in_channels: int = 3,
        ema_momentum: float = 0.9,
        pool_size: int = 8,
    ):
        super().__init__()
        self.channels_list = channels_list
        self.num_scales = len(channels_list)
        self.use_noise_gate = use_noise_gate
        self.mode = mode
        self.profile = profile

        # Get profile config
        if profile not in PROFILE_CONFIGS:
            raise ValueError(
                f"Unknown profile '{profile}'. "
                f"Available: {list(PROFILE_CONFIGS.keys())}"
            )
        pcfg = PROFILE_CONFIGS[profile]

        # Noise estimator (global, shared across scales)
        if use_noise_gate and mode != "attention":
            self.noise_estimator = SpectralNoiseEstimator(
                in_channels=in_channels,
                noise_dim=noise_dim,
            )
        else:
            self.noise_estimator = None

        # Per-scale hybrid blocks
        self.blocks = nn.ModuleList()
        for ch in channels_list:
            block = MambaAttentionBlock(
                channels=ch,
                d_state=pcfg["d_state"],
                d_conv=min(pcfg.get("d_conv", 4), 4),
                expand=pcfg["expand"],
                pool_size=pool_size,
                attention_type=pcfg["attention_type"],
                num_heads=pcfg["num_heads"],
                window_size=pcfg["window_size"],
                noise_dim=noise_dim,
                router_type=pcfg["router_type"],
                mode=mode,
            )
            self.blocks.append(block)

        # Confidence refiners (lightweight per-scale heads)
        self.conf_refiners = nn.ModuleList()
        for ch in channels_list:
            mid = max(ch // 4, 8)
            self.conf_refiners.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(ch, mid),
                nn.ReLU(inplace=True),
                nn.Linear(mid, 1),
                nn.Sigmoid(),
            ))

        # Temporal buffer for state management
        self.temporal_buffer = TemporalFeatureBuffer(
            num_scales=self.num_scales,
            ema_momentum=ema_momentum,
        )

    def forward(
        self,
        features: List[torch.Tensor],
        images: Optional[torch.Tensor] = None,
        states: Optional[List[Optional[torch.Tensor]]] = None,
    ) -> Tuple[List[torch.Tensor], List[Optional[torch.Tensor]], List[torch.Tensor]]:
        """Process multi-scale features through hybrid Mamba-Attention blocks.

        Args:
            features: List of feature maps [B, C_i, H_i, W_i] from neck.
            images: Raw input images [B, 3, H, W] for noise estimation (optional).
            states: Previous SSM states per scale, or None.

        Returns:
            refined: List of enhanced feature maps (same shapes as input).
            new_states: Updated SSM states for next frame.
            conf_weights: Per-scale confidence weights [B, 1].
        """
        # 1. Estimate noise (global descriptor)
        noise_level = None
        if self.noise_estimator is not None and images is not None:
            noise_level = self.noise_estimator(images)  # [B, noise_dim]

        # 2. Get temporal states
        if states is None and self.temporal_buffer.is_initialized:
            states = self.temporal_buffer.get_states()
        if states is None:
            states = [None] * self.num_scales

        # 3. Scene change detection
        self.temporal_buffer.detect_scene_change(features)

        # 4. Process each scale
        refined = []
        new_states = []
        raw_conf = []

        for i, (feat, state) in enumerate(zip(features, states)):
            out, ns = self.blocks[i](feat, state, noise_level)
            refined.append(out)
            new_states.append(ns)
            raw_conf.append(self.conf_refiners[i](out))

        # 5. Update temporal buffer and smooth confidence
        self.temporal_buffer.update_states(new_states)
        conf_weights = self.temporal_buffer.smooth_confidence(raw_conf)

        return refined, new_states, conf_weights

    def reset_state(self):
        """Reset temporal state (call at video/sequence boundaries)."""
        self.temporal_buffer.reset()

    @torch.no_grad()
    def get_plugin_info(self) -> dict:
        """Get plugin statistics for paper reporting."""
        total = sum(p.numel() for p in self.parameters())
        per_block = []
        for i, block in enumerate(self.blocks):
            per_block.append({
                "scale": i,
                "channels": self.channels_list[i],
                "params": sum(p.numel() for p in block.parameters()),
                "breakdown": block.count_params(),
            })

        noise_params = 0
        if self.noise_estimator is not None:
            noise_params = sum(p.numel() for p in self.noise_estimator.parameters())

        conf_params = sum(
            sum(p.numel() for p in r.parameters())
            for r in self.conf_refiners
        )

        return {
            "profile": self.profile,
            "mode": self.mode,
            "total_params": total,
            "total_params_M": total / 1e6,
            "noise_estimator_params": noise_params,
            "conf_refiners_params": conf_params,
            "per_block": per_block,
        }
