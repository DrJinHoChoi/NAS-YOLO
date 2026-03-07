"""
PicoHead — Minimal Detection Head for Sub-0.5M Detection
==========================================================
Anchor-free decoupled detection head optimized for extreme efficiency.

Key differences from standard DetectionHead:
    1. Single conv per branch (stacked_convs=1 vs standard 2)
    2. Uses depthwise separable convs instead of standard convs
    3. Shared classification weights across scales (optional)
    4. Supports both 30-class (smart glasses) and 80-class (COCO) configs

Target: ~65K params for 30-class, ~75K for 80-class.

Context-Aware Extension:
    PicoHead supports an optional context-aware classification mode
    where scene context (from the SA-SSM plugin's noise estimator or
    a separate context module) modulates class predictions. This enables
    situation-aware object detection — e.g., in "kitchen" context,
    boost food/utensil classes; in "road" context, boost vehicle/pedestrian.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict

from .pico_backbone import ConvBnAct, DepthwiseSeparableConv


class PicoDecoupledHead(nn.Module):
    """Single-scale decoupled detection head for pico models.

    Uses depthwise separable convolutions and single-layer branches
    for extreme parameter efficiency.

    Args:
        in_channels: Input feature channels.
        num_classes: Number of object classes.
        use_dwsep: Use depthwise separable conv (default: True).
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        use_dwsep: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes

        # Stem: channel normalization
        self.stem = ConvBnAct(in_channels, in_channels, 1, 1)

        # Classification branch (single DWSep conv)
        if use_dwsep:
            self.cls_conv = DepthwiseSeparableConv(in_channels, in_channels)
        else:
            self.cls_conv = ConvBnAct(in_channels, in_channels, 3, 1)
        self.cls_pred = nn.Conv2d(in_channels, num_classes, 1, bias=True)

        # Regression branch (single DWSep conv)
        if use_dwsep:
            self.reg_conv = DepthwiseSeparableConv(in_channels, in_channels)
        else:
            self.reg_conv = ConvBnAct(in_channels, in_channels, 3, 1)
        self.reg_pred = nn.Conv2d(in_channels, 4, 1, bias=True)
        self.obj_pred = nn.Conv2d(in_channels, 1, 1, bias=True)

        self._init_bias()

    def _init_bias(self):
        """Initialize biases for stable early training."""
        prior_prob = 0.01
        bias_value = -torch.log(
            torch.tensor((1.0 - prior_prob) / prior_prob)
        )
        nn.init.constant_(self.cls_pred.bias, bias_value)
        nn.init.constant_(self.obj_pred.bias, bias_value)

    def forward(
        self,
        x: torch.Tensor,
        conf_weight: Optional[torch.Tensor] = None,
        context_weight: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Feature map [B, C, H, W].
            conf_weight: NAS plugin confidence [B, 1].
            context_weight: Context-aware class modulation [B, num_classes].

        Returns:
            cls_output: [B, num_classes, H, W]
            reg_output: [B, 4, H, W]
            obj_output: [B, 1, H, W]
        """
        x = self.stem(x)

        # Classification
        cls_feat = self.cls_conv(x)
        cls_output = self.cls_pred(cls_feat)

        # Context-aware modulation: boost relevant classes based on scene
        if context_weight is not None:
            # context_weight: [B, num_classes] → [B, num_classes, 1, 1]
            cls_output = cls_output + context_weight.unsqueeze(-1).unsqueeze(-1)

        # Regression + Objectness
        reg_feat = self.reg_conv(x)
        reg_output = self.reg_pred(reg_feat)
        obj_output = self.obj_pred(reg_feat)

        # NAS confidence modulation
        if conf_weight is not None:
            obj_output = obj_output * conf_weight.unsqueeze(-1).unsqueeze(-1)

        return cls_output, reg_output, obj_output


class ContextClassifier(nn.Module):
    """Scene context classifier for situation-aware detection.

    Takes the global noise descriptor from SA-SSM's SpectralNoiseEstimator
    and produces context-aware class weights. This enables the model to
    boost relevant classes based on detected scene type.

    Example contexts:
        - Road/driving: boost vehicle, pedestrian, traffic_light
        - Kitchen/indoor: boost cup, bowl, bottle, knife
        - Outdoor/nature: boost bird, cat, dog, horse

    This is a lightweight (~2K params) module that adds zero latency
    on hardware with parallel execution units.

    Args:
        noise_dim: Dimension of noise descriptor from SpectralNoiseEstimator.
        num_classes: Number of detection classes.
        num_contexts: Number of scene contexts (soft clustering).
    """

    def __init__(
        self,
        noise_dim: int = 32,
        num_classes: int = 30,
        num_contexts: int = 8,
    ):
        super().__init__()
        self.context_proj = nn.Sequential(
            nn.Linear(noise_dim, num_contexts),
            nn.ReLU(inplace=True),
        )
        self.context_to_class = nn.Linear(num_contexts, num_classes, bias=False)

        # Initialize with small weights (subtle modulation, not override)
        nn.init.normal_(self.context_to_class.weight, std=0.01)

    def forward(self, noise_descriptor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            noise_descriptor: [B, noise_dim] from SpectralNoiseEstimator.

        Returns:
            context_weight: [B, num_classes] — additive bias for class logits.
        """
        context = self.context_proj(noise_descriptor)  # [B, num_contexts]
        return self.context_to_class(context)           # [B, num_classes]


class PicoDetectionHead(nn.Module):
    """Multi-scale pico detection head.

    Applies PicoDecoupledHead at each feature pyramid scale.
    Optionally includes context-aware classification.

    Args:
        in_channels_list: Channel counts per scale.
        num_classes: Number of object classes.
        strides: Feature map strides.
        use_context: Enable context-aware detection.
        noise_dim: Noise descriptor dimension (for context module).
    """

    def __init__(
        self,
        in_channels_list: List[int],
        num_classes: int,
        strides: List[int] = None,
        use_context: bool = False,
        noise_dim: int = 32,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_scales = len(in_channels_list)
        self.use_context = use_context

        if strides is None:
            strides = [8, 16, 32]
        self.strides = strides

        self.heads = nn.ModuleList([
            PicoDecoupledHead(ch, num_classes)
            for ch in in_channels_list
        ])

        # Optional context-aware classification
        if use_context:
            self.context_classifier = ContextClassifier(
                noise_dim=noise_dim,
                num_classes=num_classes,
            )
        else:
            self.context_classifier = None

    def forward(
        self,
        features: List[torch.Tensor],
        conf_weights: Optional[List[torch.Tensor]] = None,
        noise_descriptor: Optional[torch.Tensor] = None,
    ) -> Dict[str, object]:
        """Process multi-scale features through detection heads.

        Args:
            features: Feature maps per scale from plugin.
            conf_weights: Confidence weights from NAS plugin.
            noise_descriptor: Noise descriptor for context-aware detection.

        Returns:
            outputs: Dict with cls_preds, reg_preds, obj_preds, strides.
        """
        # Get context weights if enabled
        context_weight = None
        if self.use_context and self.context_classifier is not None:
            if noise_descriptor is not None:
                context_weight = self.context_classifier(noise_descriptor)

        cls_preds = []
        reg_preds = []
        obj_preds = []

        for i, (feat, head) in enumerate(zip(features, self.heads)):
            conf_w = conf_weights[i] if conf_weights is not None else None
            cls_out, reg_out, obj_out = head(feat, conf_w, context_weight)
            cls_preds.append(cls_out)
            reg_preds.append(reg_out)
            obj_preds.append(obj_out)

        return {
            "cls_preds": cls_preds,
            "reg_preds": reg_preds,
            "obj_preds": obj_preds,
            "strides": self.strides,
        }

    def decode_outputs(
        self,
        outputs: dict,
        img_size: Tuple[int, int],
        conf_threshold: float = 0.25,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode raw predictions into boxes, scores, class indices.

        Args:
            outputs: Raw outputs from forward().
            img_size: (H, W) of input image.
            conf_threshold: Minimum confidence.

        Returns:
            boxes: [B, N, 4] in xyxy format.
            scores: [B, N] confidence scores.
            class_ids: [B, N] predicted class indices.
        """
        all_boxes = []
        all_scores = []
        all_cls_ids = []

        for scale_idx in range(self.num_scales):
            cls_pred = outputs["cls_preds"][scale_idx]
            reg_pred = outputs["reg_preds"][scale_idx]
            obj_pred = outputs["obj_preds"][scale_idx]
            stride = self.strides[scale_idx]

            B, _, H, W = cls_pred.shape

            grid_y, grid_x = torch.meshgrid(
                torch.arange(H, device=cls_pred.device, dtype=cls_pred.dtype),
                torch.arange(W, device=cls_pred.device, dtype=cls_pred.dtype),
                indexing="ij",
            )
            grid = torch.stack([grid_x, grid_y], dim=0)

            xy = (reg_pred[:, :2] + grid.unsqueeze(0)) * stride
            wh = torch.exp(reg_pred[:, 2:4].clamp(max=10.0)) * stride

            x1 = xy[:, 0] - wh[:, 0] / 2
            y1 = xy[:, 1] - wh[:, 1] / 2
            x2 = xy[:, 0] + wh[:, 0] / 2
            y2 = xy[:, 1] + wh[:, 1] / 2

            boxes = torch.stack([x1, y1, x2, y2], dim=1)
            boxes = boxes.flatten(2).permute(0, 2, 1)

            obj_score = torch.sigmoid(obj_pred)
            cls_score = torch.sigmoid(cls_pred)
            scores = obj_score * cls_score
            scores = scores.flatten(2).permute(0, 2, 1)
            max_scores, cls_ids = scores.max(dim=-1)

            all_boxes.append(boxes)
            all_scores.append(max_scores)
            all_cls_ids.append(cls_ids)

        all_boxes = torch.cat(all_boxes, dim=1)
        all_scores = torch.cat(all_scores, dim=1)
        all_cls_ids = torch.cat(all_cls_ids, dim=1)

        img_h, img_w = img_size
        all_boxes[..., 0].clamp_(min=0, max=img_w)
        all_boxes[..., 1].clamp_(min=0, max=img_h)
        all_boxes[..., 2].clamp_(min=0, max=img_w)
        all_boxes[..., 3].clamp_(min=0, max=img_h)

        mask = all_scores > conf_threshold
        all_scores = all_scores * mask.float()

        return all_boxes, all_scores, all_cls_ids

    @torch.no_grad()
    def count_params(self) -> dict:
        """Count parameters per component."""
        info = {}
        for i, head in enumerate(self.heads):
            info[f"head_scale_{i}"] = sum(p.numel() for p in head.parameters())
        if self.context_classifier is not None:
            info["context_classifier"] = sum(
                p.numel() for p in self.context_classifier.parameters()
            )
        info["total"] = sum(p.numel() for p in self.parameters())
        return info
