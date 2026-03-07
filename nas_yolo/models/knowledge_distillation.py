"""
Knowledge Distillation for NAS-YOLO
=====================================
Transfers CNN baseline's spatial filtering knowledge to the
hybrid Mamba-Attention student model.

Key novelty: Noise-conditioned KD weighting.
  - Teacher trained on clean data → reliable on clean inputs
  - When input is noisy → teacher output is less reliable → reduce KD weight
  - The SpectralNoiseEstimator's output drives this modulation

KD strategy:
  - Feature-level: Match student NASPlugin output to teacher neck features (L2/cosine)
  - Output-level: Temperature-scaled KL divergence on detection logits
  - Noise-conditioned: Dynamically weight KD loss by input noise level

References:
  - Hinton et al., "Distilling the Knowledge in a Neural Network", 2015
  - Chen et al., "Learning Efficient Object Detection Models with Knowledge Distillation", NeurIPS 2017
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional


class FeatureDistillationLoss(nn.Module):
    """Feature-level knowledge distillation loss.

    Matches student's refined features (after NASPlugin) to teacher's
    neck features (without NASPlugin). This encourages the student to
    preserve the CNN's spatial filtering behavior while adding temporal
    robustness through the Mamba branch.

    Args:
        loss_type: 'l2' | 'cosine' | 'combined'.
        normalize: Normalize features before computing loss.
    """

    def __init__(self, loss_type: str = "combined", normalize: bool = True):
        super().__init__()
        self.loss_type = loss_type
        self.normalize = normalize

    def forward(
        self,
        student_features: List[torch.Tensor],
        teacher_features: List[torch.Tensor],
    ) -> torch.Tensor:
        """Compute feature distillation loss.

        Args:
            student_features: List of student feature maps [B, C_i, H_i, W_i].
            teacher_features: List of teacher feature maps (same shapes).

        Returns:
            Scalar loss value.
        """
        assert len(student_features) == len(teacher_features), \
            f"Scale mismatch: student has {len(student_features)}, teacher has {len(teacher_features)}"

        total_loss = torch.tensor(0.0, device=student_features[0].device)

        for s_feat, t_feat in zip(student_features, teacher_features):
            # Flatten spatial dims for comparison
            B = s_feat.shape[0]
            s_flat = s_feat.reshape(B, -1)
            t_flat = t_feat.reshape(B, -1).detach()

            if self.normalize:
                s_flat = F.normalize(s_flat, dim=-1)
                t_flat = F.normalize(t_flat, dim=-1)

            if self.loss_type == "l2":
                loss = F.mse_loss(s_flat, t_flat)
            elif self.loss_type == "cosine":
                loss = 1.0 - F.cosine_similarity(s_flat, t_flat, dim=-1).mean()
            elif self.loss_type == "combined":
                l2_loss = F.mse_loss(s_flat, t_flat)
                cos_loss = 1.0 - F.cosine_similarity(s_flat, t_flat, dim=-1).mean()
                loss = 0.5 * l2_loss + 0.5 * cos_loss
            else:
                raise ValueError(f"Unknown loss_type: {self.loss_type}")

            total_loss = total_loss + loss

        return total_loss / len(student_features)


class OutputDistillationLoss(nn.Module):
    """Output-level knowledge distillation via temperature-scaled KL divergence.

    Softens the teacher's detection outputs (classification, objectness)
    and minimizes KL divergence between softened student/teacher distributions.

    Args:
        temperature: Softmax temperature (higher = softer distributions).
    """

    def __init__(self, temperature: float = 4.0):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        student_outputs: Dict[str, List[torch.Tensor]],
        teacher_outputs: Dict[str, List[torch.Tensor]],
    ) -> torch.Tensor:
        """Compute output distillation loss.

        Args:
            student_outputs: Dict with 'cls_preds' and 'obj_preds' lists.
            teacher_outputs: Dict with same structure (detached).

        Returns:
            Scalar loss value.
        """
        total_loss = torch.tensor(0.0, device=next(iter(
            student_outputs.get("cls_preds", student_outputs.get("features", [torch.zeros(1)]))
        )).device)

        T = self.temperature
        count = 0

        # Classification logits KD
        if "cls_preds" in student_outputs and "cls_preds" in teacher_outputs:
            for s_cls, t_cls in zip(student_outputs["cls_preds"],
                                     teacher_outputs["cls_preds"]):
                t_cls = t_cls.detach()
                s_soft = F.log_softmax(s_cls / T, dim=1)
                t_soft = F.softmax(t_cls / T, dim=1)
                kl_loss = F.kl_div(s_soft, t_soft, reduction='batchmean') * (T * T)
                total_loss = total_loss + kl_loss
                count += 1

        # Objectness logits KD (binary KL)
        if "obj_preds" in student_outputs and "obj_preds" in teacher_outputs:
            for s_obj, t_obj in zip(student_outputs["obj_preds"],
                                     teacher_outputs["obj_preds"]):
                t_obj = t_obj.detach()
                s_prob = torch.sigmoid(s_obj / T)
                t_prob = torch.sigmoid(t_obj / T)
                # Binary KL: p*log(p/q) + (1-p)*log((1-p)/(1-q))
                eps = 1e-7
                binary_kl = (
                    t_prob * torch.log((t_prob + eps) / (s_prob + eps))
                    + (1 - t_prob) * torch.log((1 - t_prob + eps) / (1 - s_prob + eps))
                )
                total_loss = total_loss + binary_kl.mean() * (T * T)
                count += 1

        return total_loss / max(count, 1)


class NoiseConditionedKDWrapper(nn.Module):
    """Noise-conditioned Knowledge Distillation wrapper.

    Modulates KD loss weight based on input noise level. The key insight:
    the CNN teacher was trained on clean data, so its outputs are most
    reliable when the student's input is also clean. Under heavy noise,
    the teacher's knowledge becomes less relevant.

    reliability = sigmoid(MLP(noise_level))
    kd_loss = reliability * (alpha_feat * L_feature + alpha_out * L_output)

    Args:
        noise_dim: Noise descriptor dimension.
        feature_loss_type: Feature KD loss type ('l2', 'cosine', 'combined').
        temperature: Output KD temperature.
        alpha_feature: Weight for feature-level KD.
        alpha_output: Weight for output-level KD.
    """

    def __init__(
        self,
        noise_dim: int = 32,
        feature_loss_type: str = "combined",
        temperature: float = 4.0,
        alpha_feature: float = 1.0,
        alpha_output: float = 1.0,
    ):
        super().__init__()
        self.alpha_feature = alpha_feature
        self.alpha_output = alpha_output

        # Noise → reliability mapping
        self.reliability_net = nn.Sequential(
            nn.Linear(noise_dim, noise_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(noise_dim // 2, 1),
        )
        # Initialize bias to start with high reliability (~0.73)
        with torch.no_grad():
            self.reliability_net[-1].bias.fill_(1.0)

        # Sub-losses
        self.feature_loss = FeatureDistillationLoss(loss_type=feature_loss_type)
        self.output_loss = OutputDistillationLoss(temperature=temperature)

    def forward(
        self,
        student_features: List[torch.Tensor],
        teacher_features: List[torch.Tensor],
        student_outputs: Optional[Dict[str, List[torch.Tensor]]] = None,
        teacher_outputs: Optional[Dict[str, List[torch.Tensor]]] = None,
        noise_level: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute noise-conditioned KD loss.

        Args:
            student_features: Student's refined feature maps.
            teacher_features: Teacher's neck feature maps.
            student_outputs: Student's detection outputs (optional).
            teacher_outputs: Teacher's detection outputs (optional).
            noise_level: Noise descriptor [B, noise_dim] (optional).

        Returns:
            Dict with 'kd_feature_loss', 'kd_output_loss', 'kd_total_loss',
            'teacher_reliability'.
        """
        device = student_features[0].device

        # Compute reliability from noise level
        if noise_level is not None:
            reliability = torch.sigmoid(self.reliability_net(noise_level))  # [B, 1]
            reliability = reliability.mean()  # Scalar for batch
        else:
            reliability = torch.tensor(1.0, device=device)

        # Feature-level KD
        feat_loss = self.feature_loss(student_features, teacher_features)

        # Output-level KD (optional)
        if student_outputs is not None and teacher_outputs is not None:
            out_loss = self.output_loss(student_outputs, teacher_outputs)
        else:
            out_loss = torch.tensor(0.0, device=device)

        # Weighted total
        total_kd = reliability * (
            self.alpha_feature * feat_loss + self.alpha_output * out_loss
        )

        return {
            "kd_feature_loss": feat_loss.detach(),
            "kd_output_loss": out_loss.detach(),
            "kd_total_loss": total_kd,
            "teacher_reliability": reliability.detach(),
        }
