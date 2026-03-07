"""
PicoKD — Knowledge Distillation for PicoYOLO
===============================================
Transfers knowledge from YOLOv8n teacher (3.2M) to PicoYOLO student (~0.3M).

Handles channel mismatch via 1×1 projection layers:
    Student [32, 64, 128] → Teacher [64, 128, 256]

KD strategy:
    1. Feature-level: Project student features to teacher dimension, L2 loss
    2. Output-level: Temperature-scaled KL divergence on class logits
    3. Noise-conditioned: Reduce KD weight when input is noisy
       (reuses NoiseConditionedKDWrapper from existing KD framework)
    4. Progressive: Start with strong KD (α=1.0), decay over epochs

This module is designed for the two-stage training pipeline:
    - Stage 1 (20 epochs): Plugin warmup, KD from teacher neck features
    - Stage 2 (280 epochs): Full fine-tune, progressive KD decay
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict

from .knowledge_distillation import (
    FeatureDistillationLoss,
    OutputDistillationLoss,
    NoiseConditionedKDWrapper,
)


class ChannelProjector(nn.Module):
    """1×1 Conv projection from student to teacher channel dimension.

    Minimal overhead: just in_ch × out_ch parameters per scale.
    For [32→64, 64→128, 128→256]: total = 2K + 8K + 32K = 42K params.

    These projection layers are NOT part of the deployed model —
    they are only used during KD training and discarded at export.

    Args:
        student_channels: Student channel counts per scale.
        teacher_channels: Teacher channel counts per scale.
    """

    def __init__(
        self,
        student_channels: List[int],
        teacher_channels: List[int],
    ):
        super().__init__()
        assert len(student_channels) == len(teacher_channels)

        self.projectors = nn.ModuleList([
            nn.Conv2d(s_ch, t_ch, 1, bias=False)
            for s_ch, t_ch in zip(student_channels, teacher_channels)
        ])

    def forward(
        self, student_features: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """Project student features to teacher dimension.

        Args:
            student_features: List of [B, S_ch, H, W] per scale.

        Returns:
            projected: List of [B, T_ch, H, W] per scale.
        """
        return [
            proj(feat)
            for proj, feat in zip(self.projectors, student_features)
        ]


class PicoKDWrapper(nn.Module):
    """Knowledge Distillation wrapper for PicoYOLO training.

    Wraps the student PicoYOLO model and computes KD losses against
    a frozen teacher model during training.

    Args:
        student: PicoYOLO model (trainable).
        teacher_channels: Teacher model channel dimensions.
        noise_dim: Noise descriptor dimension.
        alpha_feature: Feature distillation loss weight.
        alpha_output: Output distillation loss weight.
        temperature: KL divergence temperature.
    """

    def __init__(
        self,
        student: nn.Module,
        teacher_channels: List[int] = None,
        noise_dim: int = 32,
        alpha_feature: float = 0.5,
        alpha_output: float = 0.5,
        temperature: float = 4.0,
    ):
        super().__init__()
        self.student = student

        if teacher_channels is None:
            teacher_channels = [64, 128, 256]  # YOLOv8n default

        # Student channel dimensions from backbone
        student_channels = student.backbone.out_channels

        # Channel projectors (training only, discarded at export)
        self.channel_projector = ChannelProjector(
            student_channels=student_channels,
            teacher_channels=teacher_channels,
        )

        # Feature distillation loss
        self.feature_loss = FeatureDistillationLoss(loss_type="combined")

        # Output distillation loss
        self.output_loss = OutputDistillationLoss(temperature=temperature)

        # Noise-conditioned reliability
        self.noise_kd = NoiseConditionedKDWrapper(
            noise_dim=noise_dim,
            feature_loss_type="combined",
        )

        self.alpha_feature = alpha_feature
        self.alpha_output = alpha_output

    def compute_kd_loss(
        self,
        student_features: List[torch.Tensor],
        teacher_features: List[torch.Tensor],
        student_outputs: Optional[Dict] = None,
        teacher_outputs: Optional[Dict] = None,
        noise_level: Optional[torch.Tensor] = None,
        epoch: int = 0,
        total_epochs: int = 300,
    ) -> Dict[str, torch.Tensor]:
        """Compute knowledge distillation losses.

        Args:
            student_features: Student neck output features.
            teacher_features: Teacher neck output features.
            student_outputs: Student detection outputs (optional).
            teacher_outputs: Teacher detection outputs (optional).
            noise_level: Noise descriptor for reliability weighting.
            epoch: Current epoch (for progressive decay).
            total_epochs: Total training epochs.

        Returns:
            losses: Dict with 'kd_feature', 'kd_output', 'kd_total'.
        """
        # Progressive KD decay: strong early, weak late
        progress = epoch / max(total_epochs, 1)
        kd_weight = max(0.1, 1.0 - 0.9 * progress)  # 1.0 → 0.1

        losses = {}

        # Feature distillation
        projected = self.channel_projector(student_features)
        feature_loss = torch.tensor(0.0, device=projected[0].device)
        for s_feat, t_feat in zip(projected, teacher_features):
            # Resize if spatial dims don't match
            if s_feat.shape[-2:] != t_feat.shape[-2:]:
                t_feat = F.interpolate(
                    t_feat, size=s_feat.shape[-2:], mode="nearest"
                )
            feature_loss = feature_loss + self.feature_loss(s_feat, t_feat)
        feature_loss = feature_loss / len(projected)
        losses["kd_feature"] = feature_loss * self.alpha_feature * kd_weight

        # Output distillation (if available)
        if student_outputs is not None and teacher_outputs is not None:
            output_loss = torch.tensor(0.0, device=projected[0].device)
            for s_cls, t_cls in zip(
                student_outputs.get("cls_preds", []),
                teacher_outputs.get("cls_preds", []),
            ):
                if s_cls.shape != t_cls.shape:
                    t_cls = F.interpolate(
                        t_cls, size=s_cls.shape[-2:], mode="nearest"
                    )
                output_loss = output_loss + self.output_loss(s_cls, t_cls)
            losses["kd_output"] = (
                output_loss * self.alpha_output * kd_weight
            )
        else:
            losses["kd_output"] = torch.tensor(0.0, device=projected[0].device)

        # Noise-conditioned reliability (reduce KD when noisy)
        if noise_level is not None:
            reliability = self.noise_kd.reliability_net(noise_level).mean()
            losses["kd_feature"] = losses["kd_feature"] * reliability
            losses["kd_output"] = losses["kd_output"] * reliability
            losses["kd_reliability"] = reliability

        losses["kd_total"] = losses["kd_feature"] + losses["kd_output"]
        losses["kd_weight"] = torch.tensor(kd_weight)

        return losses
