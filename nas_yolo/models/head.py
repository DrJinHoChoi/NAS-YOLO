"""
Detection Head for NAS-YOLO
============================
Anchor-free decoupled detection head with confidence refinement.

The head receives noise-aware, temporally-smoothed features from the NAS Module
and produces final detections. The key integration point is the confidence
modulation: per-scale confidence weights from the NAS Module are applied to
objectness scores, enabling the temporal state to suppress false positives.

Architecture per scale:
    Feature → Stem Conv
               ├── Classification branch → [B, num_classes, H, W]
               └── Regression branch     → [B, 4+1, H, W]  (bbox + objectness)

Confidence refinement:
    obj_score_final = obj_score * conf_weight_from_NAS

This is where temporal state directly improves detection quality.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
from .backbone import ConvBlock


class DecoupledHead(nn.Module):
    """Single-scale decoupled detection head.

    Separates classification and regression into independent branches,
    following YOLOX design. Each branch has its own feature processing
    to avoid task interference.

    Args:
        in_channels: Input feature channels.
        num_classes: Number of object classes.
        num_anchors: Number of anchors per location (1 for anchor-free).
        stacked_convs: Number of stacked convolution layers per branch.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        num_anchors: int = 1,
        stacked_convs: int = 2,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        # Stem
        self.stem = ConvBlock(in_channels, in_channels, 1, 1)

        # Classification branch
        cls_convs = []
        for _ in range(stacked_convs):
            cls_convs.append(ConvBlock(in_channels, in_channels, 3, 1))
        self.cls_convs = nn.Sequential(*cls_convs)
        self.cls_pred = nn.Conv2d(
            in_channels, num_classes * num_anchors, 1, bias=True
        )

        # Regression branch (4 bbox coords + 1 objectness)
        reg_convs = []
        for _ in range(stacked_convs):
            reg_convs.append(ConvBlock(in_channels, in_channels, 3, 1))
        self.reg_convs = nn.Sequential(*reg_convs)
        self.reg_pred = nn.Conv2d(
            in_channels, 4 * num_anchors, 1, bias=True
        )
        self.obj_pred = nn.Conv2d(
            in_channels, 1 * num_anchors, 1, bias=True
        )

        self._init_bias()

    def _init_bias(self):
        """Initialize biases for stable early training."""
        # Classification bias: log(prior_prob) for focal loss stability
        prior_prob = 0.01
        bias_value = -torch.log(torch.tensor((1.0 - prior_prob) / prior_prob))
        nn.init.constant_(self.cls_pred.bias, bias_value)
        nn.init.constant_(self.obj_pred.bias, bias_value)

    def forward(
        self, x: torch.Tensor, conf_weight: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Feature map [B, C, H, W].
            conf_weight: Confidence modulation from NAS Module [B, 1].

        Returns:
            cls_output: Class predictions [B, num_classes, H, W].
            reg_output: Bbox predictions [B, 4, H, W].
            obj_output: Objectness predictions [B, 1, H, W].
        """
        x = self.stem(x)

        # Classification branch
        cls_feat = self.cls_convs(x)
        cls_output = self.cls_pred(cls_feat)

        # Regression + Objectness branch
        reg_feat = self.reg_convs(x)
        reg_output = self.reg_pred(reg_feat)
        obj_output = self.obj_pred(reg_feat)

        # Apply NAS confidence modulation to objectness
        if conf_weight is not None:
            # conf_weight: [B, 1] → [B, 1, 1, 1]
            obj_output = obj_output * conf_weight.unsqueeze(-1).unsqueeze(-1)

        return cls_output, reg_output, obj_output


class DetectionHead(nn.Module):
    """Multi-scale detection head.

    Applies DecoupledHead at each feature pyramid scale and produces
    unified detection outputs.

    Args:
        in_channels_list: List of channel counts per scale.
        num_classes: Number of object classes.
        strides: Feature map strides relative to input image.
    """

    def __init__(
        self,
        in_channels_list: List[int],
        num_classes: int,
        strides: List[int] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_scales = len(in_channels_list)

        if strides is None:
            strides = [8, 16, 32]
        self.strides = strides

        self.heads = nn.ModuleList([
            DecoupledHead(ch, num_classes)
            for ch in in_channels_list
        ])

    def forward(
        self,
        features: List[torch.Tensor],
        conf_weights: Optional[List[torch.Tensor]] = None,
    ) -> dict:
        """Process multi-scale features through detection heads.

        Args:
            features: List of feature maps per scale.
            conf_weights: Optional confidence weights from NAS Module.

        Returns:
            outputs: Dict with keys:
                'cls_preds': List of [B, num_classes, H_i, W_i] per scale
                'reg_preds': List of [B, 4, H_i, W_i] per scale
                'obj_preds': List of [B, 1, H_i, W_i] per scale
                'strides': List of stride values
        """
        cls_preds = []
        reg_preds = []
        obj_preds = []

        for i, (feat, head) in enumerate(zip(features, self.heads)):
            conf_w = conf_weights[i] if conf_weights is not None else None
            cls_out, reg_out, obj_out = head(feat, conf_w)
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
        """Decode raw predictions into boxes, scores, and class indices.

        Args:
            outputs: Raw outputs from forward().
            img_size: (H, W) of the input image.
            conf_threshold: Minimum confidence threshold for filtering.

        Returns:
            boxes: [B, N, 4] in xyxy format.
            scores: [B, N] confidence scores.
            class_ids: [B, N] predicted class indices.
        """
        all_boxes = []
        all_scores = []
        all_cls_ids = []

        for scale_idx in range(self.num_scales):
            cls_pred = outputs["cls_preds"][scale_idx]  # [B, num_cls, H, W]
            reg_pred = outputs["reg_preds"][scale_idx]  # [B, 4, H, W]
            obj_pred = outputs["obj_preds"][scale_idx]  # [B, 1, H, W]
            stride = self.strides[scale_idx]

            B, _, H, W = cls_pred.shape

            # Generate grid
            grid_y, grid_x = torch.meshgrid(
                torch.arange(H, device=cls_pred.device, dtype=cls_pred.dtype),
                torch.arange(W, device=cls_pred.device, dtype=cls_pred.dtype),
                indexing="ij",
            )
            grid = torch.stack([grid_x, grid_y], dim=0)  # [2, H, W]

            # Decode boxes: center-offset format → xyxy
            xy = (reg_pred[:, :2] + grid.unsqueeze(0)) * stride
            wh = torch.exp(reg_pred[:, 2:4].clamp(max=10.0)) * stride

            x1 = xy[:, 0] - wh[:, 0] / 2
            y1 = xy[:, 1] - wh[:, 1] / 2
            x2 = xy[:, 0] + wh[:, 0] / 2
            y2 = xy[:, 1] + wh[:, 1] / 2

            boxes = torch.stack([x1, y1, x2, y2], dim=1)  # [B, 4, H, W]
            boxes = boxes.flatten(2).permute(0, 2, 1)  # [B, H*W, 4]

            # Decode scores: objectness × class probability
            obj_score = torch.sigmoid(obj_pred)  # [B, 1, H, W]
            cls_score = torch.sigmoid(cls_pred)  # [B, num_cls, H, W]
            scores = obj_score * cls_score  # [B, num_cls, H, W]

            # Flatten and find max class per location
            scores = scores.flatten(2).permute(0, 2, 1)  # [B, H*W, num_cls]
            max_scores, cls_ids = scores.max(dim=-1)  # [B, H*W]

            all_boxes.append(boxes)
            all_scores.append(max_scores)
            all_cls_ids.append(cls_ids)

        # Concatenate across scales
        all_boxes = torch.cat(all_boxes, dim=1)
        all_scores = torch.cat(all_scores, dim=1)
        all_cls_ids = torch.cat(all_cls_ids, dim=1)

        # Clip boxes to image boundaries
        img_h, img_w = img_size
        all_boxes[..., 0].clamp_(min=0, max=img_w)
        all_boxes[..., 1].clamp_(min=0, max=img_h)
        all_boxes[..., 2].clamp_(min=0, max=img_w)
        all_boxes[..., 3].clamp_(min=0, max=img_h)

        # Filter by confidence threshold
        mask = all_scores > conf_threshold
        # Keep all entries but zero out low-confidence ones for batched NMS
        all_scores = all_scores * mask.float()

        return all_boxes, all_scores, all_cls_ids
