"""
Temporal Feature Buffer
========================
Manages the frame-to-frame state propagation for the NAS Module.

The buffer stores SSM hidden states across frames, enabling the detector to
maintain temporal continuity without storing raw feature maps (memory efficient).

For video/streaming inference:
  - States are carried forward frame-by-frame
  - Optional exponential moving average (EMA) smoothing
  - Automatic state reset on scene changes (detected via feature divergence)

For image-based training with synthetic temporal augmentation:
  - Generates pseudo-temporal sequences via corruption progression
  - Enables temporal training without video data
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict


class TemporalFeatureBuffer(nn.Module):
    """Manages temporal state for multi-scale feature propagation.

    Stores per-scale SSM hidden states and provides utilities for:
      - State initialization, update, and retrieval
      - Scene change detection and state reset
      - EMA-based confidence smoothing across frames

    Args:
        num_scales: Number of feature pyramid scales.
        ema_momentum: Momentum for exponential moving average smoothing.
        scene_change_threshold: Cosine similarity threshold for scene change detection.
        max_history: Maximum number of confidence values to track for smoothing.
    """

    def __init__(
        self,
        num_scales: int = 3,
        ema_momentum: float = 0.9,
        scene_change_threshold: float = 0.5,
        max_history: int = 10,
    ):
        super().__init__()
        self.num_scales = num_scales
        self.ema_momentum = ema_momentum
        self.scene_change_threshold = scene_change_threshold
        self.max_history = max_history

        # Internal state (not nn.Parameters — managed manually)
        self._states: Optional[List[Optional[torch.Tensor]]] = None
        self._prev_features: Optional[List[torch.Tensor]] = None
        self._confidence_history: List[List[torch.Tensor]] = [[] for _ in range(num_scales)]
        self._frame_count: int = 0

    def reset(self):
        """Reset all temporal state (e.g., at start of new video/sequence)."""
        self._states = None
        self._prev_features = None
        self._confidence_history = [[] for _ in range(self.num_scales)]
        self._frame_count = 0

    def get_states(self) -> Optional[List[Optional[torch.Tensor]]]:
        """Retrieve current SSM states for all scales."""
        return self._states

    def update_states(self, new_states: List[Optional[torch.Tensor]]):
        """Store updated SSM states after NAS Module processing."""
        self._states = new_states
        self._frame_count += 1

    def detect_scene_change(self, current_features: List[torch.Tensor]) -> bool:
        """Detect scene changes by comparing feature statistics.

        Uses cosine similarity of global average pooled features between
        consecutive frames. A sharp drop indicates a scene change, triggering
        state reset.

        Args:
            current_features: List of current frame features per scale.

        Returns:
            True if a scene change is detected.
        """
        if self._prev_features is None:
            self._prev_features = [
                f.detach().mean(dim=(-2, -1)) for f in current_features
            ]
            return False

        similarities = []
        for i, feat in enumerate(current_features):
            curr_global = feat.detach().mean(dim=(-2, -1))  # [B, C]
            prev_global = self._prev_features[i]

            # Cosine similarity
            sim = torch.nn.functional.cosine_similarity(
                curr_global, prev_global, dim=-1
            ).mean()
            similarities.append(sim.item())

        # Update stored features
        self._prev_features = [
            f.detach().mean(dim=(-2, -1)) for f in current_features
        ]

        # Scene change if average similarity drops below threshold
        avg_sim = sum(similarities) / len(similarities)
        if avg_sim < self.scene_change_threshold:
            self.reset()
            self._prev_features = [
                f.detach().mean(dim=(-2, -1)) for f in current_features
            ]
            return True

        return False

    def smooth_confidence(
        self, confidence_weights: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """Apply EMA smoothing to confidence weights across frames.

        This reduces confidence jitter — a key contributor to bounding box
        instability in noisy conditions.

        Args:
            confidence_weights: List of per-scale confidence weights [B, 1].

        Returns:
            smoothed: EMA-smoothed confidence weights.
        """
        smoothed = []
        for i, conf in enumerate(confidence_weights):
            if len(self._confidence_history[i]) == 0:
                smoothed.append(conf)
            else:
                prev = self._confidence_history[i][-1]
                # Ensure device consistency between stored history and current conf
                if prev.device != conf.device:
                    prev = prev.to(conf.device)
                ema = self.ema_momentum * prev + (1 - self.ema_momentum) * conf
                smoothed.append(ema)

            # Store detached: gradients flow through current conf only (by design).
            # Inter-frame gradient flow is handled by TCL loss, not EMA chain.
            self._confidence_history[i].append(conf.detach())
            if len(self._confidence_history[i]) > self.max_history:
                self._confidence_history[i].pop(0)

        return smoothed

    @property
    def frame_count(self) -> int:
        """Number of frames processed since last reset."""
        return self._frame_count

    @property
    def is_initialized(self) -> bool:
        """Whether temporal states have been initialized."""
        return self._states is not None


class PseudoTemporalGenerator(nn.Module):
    """Generates pseudo-temporal sequences for training without video data.

    Creates synthetic frame sequences by applying progressive corruption
    to a single image, simulating temporal noise dynamics. This enables
    training the temporal state module on static image datasets (COCO).

    Strategies:
      - Progressive noise: gradually increase noise level across "frames"
      - Random walk: random corruption severity per frame
      - Burst noise: clean-clean-noisy-clean pattern
      - Illumination ramp: simulate lighting changes

    Args:
        sequence_length: Number of pseudo-frames to generate.
        strategy: Generation strategy ('progressive', 'random_walk', 'burst').
    """

    def __init__(self, sequence_length: int = 5, strategy: str = "progressive"):
        super().__init__()
        self.sequence_length = sequence_length
        self.strategy = strategy

    def forward(self, image: torch.Tensor) -> List[torch.Tensor]:
        """Generate a pseudo-temporal sequence from a single image.

        Args:
            image: Single image [B, C, H, W].

        Returns:
            sequence: List of T images [B, C, H, W], each with different noise.
        """
        B, C, H, W = image.shape
        sequence = []

        if self.strategy == "progressive":
            for t in range(self.sequence_length):
                severity = t / max(self.sequence_length - 1, 1)
                noise_std = severity * 0.15  # max noise std = 0.15
                noise = torch.randn_like(image) * noise_std
                sequence.append(torch.clamp(image + noise, 0, 1))

        elif self.strategy == "random_walk":
            severity = 0.0
            for t in range(self.sequence_length):
                severity += torch.randn(1).item() * 0.05
                severity = max(0.0, min(severity, 0.2))
                noise = torch.randn_like(image) * severity
                sequence.append(torch.clamp(image + noise, 0, 1))

        elif self.strategy == "burst":
            # Clean-clean-noisy-clean-clean pattern
            burst_frame = self.sequence_length // 2
            for t in range(self.sequence_length):
                if t == burst_frame:
                    noise = torch.randn_like(image) * 0.2
                    sequence.append(torch.clamp(image + noise, 0, 1))
                else:
                    noise = torch.randn_like(image) * 0.02
                    sequence.append(torch.clamp(image + noise, 0, 1))

        elif self.strategy == "illumination":
            for t in range(self.sequence_length):
                factor = 0.5 + t / max(self.sequence_length - 1, 1) * 1.0
                darkened = image * factor
                sequence.append(torch.clamp(darkened, 0, 1))

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        return sequence
