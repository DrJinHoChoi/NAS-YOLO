"""
NAS-YOLO: Full Model Assembly
===============================
Noise-Aware State YOLO — complete model integrating all components.

Pipeline:
    Image → Backbone → Neck → NAS Module → Detection Head → Outputs
              ↑                    ↑
         (standard)        (our contribution)

The NAS Module sits between neck and head, processing multi-scale features
through temporal state spaces conditioned on noise estimation. This
architectural placement is deliberate:
  - After neck: features are already multi-scale fused
  - Before head: temporal refinement improves both cls and reg

Model Variants:
    NAS-YOLO-n: ~3.5M params, ~4ms latency (Jetson Nano)
    NAS-YOLO-s: ~8.5M params, ~6ms latency
    NAS-YOLO-m: ~18M params, ~10ms latency
    NAS-YOLO-l: ~35M params, ~16ms latency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict

from .backbone import CSPDarknet
from .neck import PAFPN
from .nas_module import NoiseAwareStateModule
from .noise_gate import MultiScaleNoiseEstimator
from .temporal_buffer import TemporalFeatureBuffer
from .head import DetectionHead


# ============================================================================
# Model Configuration Presets
# ============================================================================

MODEL_CONFIGS = {
    "nano": {
        "width_multiple": 0.25,
        "depth_multiple": 0.33,
        "d_state": 8,
        "d_conv": 3,
    },
    "small": {
        "width_multiple": 0.50,
        "depth_multiple": 0.33,
        "d_state": 16,
        "d_conv": 4,
    },
    "medium": {
        "width_multiple": 0.75,
        "depth_multiple": 0.67,
        "d_state": 16,
        "d_conv": 4,
    },
    "large": {
        "width_multiple": 1.00,
        "depth_multiple": 1.00,
        "d_state": 32,
        "d_conv": 4,
    },
}


class NASYOLO(nn.Module):
    """Noise-Aware State YOLO detection model.

    "We turn object detection from frame-wise perception into state-aware perception."

    Args:
        num_classes: Number of object categories.
        model_scale: Model variant ('nano', 'small', 'medium', 'large').
        in_channels: Input image channels.
        use_noise_gate: Enable noise-aware gating (True for full model, False for ablation).
        use_temporal: Enable temporal state module (True for full model, False for ablation).
        noise_dim: Noise descriptor dimension.
        ema_momentum: Temporal confidence smoothing momentum.
        pretrained_backbone: Path to pretrained backbone weights (optional).
    """

    def __init__(
        self,
        num_classes: int = 80,
        model_scale: str = "small",
        in_channels: int = 3,
        use_noise_gate: bool = True,
        use_temporal: bool = True,
        noise_dim: int = 32,
        ema_momentum: float = 0.9,
        pretrained_backbone: str = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.model_scale = model_scale
        self.use_noise_gate = use_noise_gate
        self.use_temporal = use_temporal

        # Get config for model scale
        config = MODEL_CONFIGS[model_scale]
        width = config["width_multiple"]
        depth = config["depth_multiple"]
        d_state = config["d_state"]
        d_conv = config["d_conv"]

        # === Backbone ===
        self.backbone = CSPDarknet(
            width_multiple=width,
            depth_multiple=depth,
            in_channels=in_channels,
            out_indices=(2, 3, 4),
        )

        if pretrained_backbone:
            state_dict = torch.load(pretrained_backbone, map_location="cpu")
            self.backbone.load_state_dict(state_dict, strict=False)

        # === Neck ===
        self.neck = PAFPN(
            in_channels=self.backbone.out_channels,
            depth_multiple=depth,
            width_multiple=width,
        )

        # === Noise-Aware State Module (our contribution) ===
        neck_channels = self.neck.out_channels

        if use_temporal:
            # Noise estimator
            if use_noise_gate:
                self.noise_estimator = MultiScaleNoiseEstimator(
                    in_channels=in_channels,
                    channels_list=neck_channels,
                    noise_dim=noise_dim,
                )
            else:
                self.noise_estimator = None

            # Core NAS Module
            self.nas_module = NoiseAwareStateModule(
                channels_list=neck_channels,
                d_state=d_state,
                d_conv=d_conv,
                use_noise_gate=use_noise_gate,
                noise_dim=noise_dim,
            )

            # Temporal buffer for state management
            self.temporal_buffer = TemporalFeatureBuffer(
                num_scales=len(neck_channels),
                ema_momentum=ema_momentum,
            )
        else:
            self.noise_estimator = None
            self.nas_module = None
            self.temporal_buffer = None

        # === Detection Head ===
        self.head = DetectionHead(
            in_channels_list=neck_channels,
            num_classes=num_classes,
            strides=[8, 16, 32],
        )

    def forward(
        self,
        images: torch.Tensor,
        targets: Optional[List[Dict]] = None,
    ) -> Dict:
        """Forward pass for single frame.

        For temporal inference, call forward_temporal() instead.

        Args:
            images: Input images [B, 3, H, W].
            targets: Optional ground truth for loss computation.

        Returns:
            outputs: Detection outputs dict.
        """
        # Backbone
        backbone_features = self.backbone(images)

        # Neck
        neck_features = self.neck(backbone_features)

        # NAS Module
        conf_weights = None
        if self.use_temporal and self.nas_module is not None:
            # Estimate noise (per-scale descriptors)
            noise_level = None
            if self.noise_estimator is not None:
                noise_level = self.noise_estimator(images, neck_features)

            # Get temporal states
            states = None
            if self.temporal_buffer is not None:
                # Detect scene changes
                self.temporal_buffer.detect_scene_change(neck_features)
                states = self.temporal_buffer.get_states()

            # Process through NAS Module
            refined_features, new_states, raw_conf_weights = self.nas_module(
                neck_features, states, noise_level
            )

            # Update temporal buffer
            if self.temporal_buffer is not None:
                self.temporal_buffer.update_states(new_states)
                conf_weights = self.temporal_buffer.smooth_confidence(raw_conf_weights)
            else:
                conf_weights = raw_conf_weights

            neck_features = refined_features

        # Detection Head
        outputs = self.head(neck_features, conf_weights)

        # Compute loss if training
        if targets is not None and self.training:
            losses = self.compute_loss(outputs, targets, images.shape[-2:])
            outputs["losses"] = losses

        return outputs

    def forward_temporal(
        self,
        image_sequence: List[torch.Tensor],
        targets_sequence: Optional[List[List[Dict]]] = None,
    ) -> List[Dict]:
        """Forward pass for temporal (video) sequence.

        Processes frames sequentially, propagating state across frames.
        This is the primary inference mode for NAS-YOLO.

        Args:
            image_sequence: List of T images [B, 3, H, W].
            targets_sequence: Optional list of T target dicts.

        Returns:
            outputs_sequence: List of T output dicts.
        """
        # Reset temporal state at start of sequence
        if self.temporal_buffer is not None:
            self.temporal_buffer.reset()

        outputs_sequence = []
        for t, images in enumerate(image_sequence):
            targets = targets_sequence[t] if targets_sequence is not None else None
            outputs = self.forward(images, targets)
            outputs_sequence.append(outputs)

        return outputs_sequence

    def compute_loss(
        self,
        outputs: Dict,
        targets: List[Dict],
        img_size: Tuple[int, int],
    ) -> Dict[str, torch.Tensor]:
        """Compute detection losses.

        Uses:
          - Binary Cross-Entropy with Focal modulation for classification
          - GIoU loss for bounding box regression
          - Binary Cross-Entropy for objectness
          - Temporal consistency loss (NAS-YOLO specific)

        Args:
            outputs: Detection head outputs.
            targets: List of target dicts with 'boxes' [N, 4] and 'labels' [N].
            img_size: (H, W) of input image.

        Returns:
            loss_dict: Dict of individual losses and total loss.
        """
        device = outputs["cls_preds"][0].device
        num_scales = len(outputs["cls_preds"])

        total_cls_loss = torch.tensor(0.0, device=device)
        total_reg_loss = torch.tensor(0.0, device=device)
        total_obj_loss = torch.tensor(0.0, device=device)

        for scale_idx in range(num_scales):
            cls_pred = outputs["cls_preds"][scale_idx]
            reg_pred = outputs["reg_preds"][scale_idx]
            obj_pred = outputs["obj_preds"][scale_idx]
            stride = outputs["strides"][scale_idx]

            B, num_cls, H, W = cls_pred.shape

            # Build targets for this scale
            cls_target = torch.zeros_like(cls_pred)
            obj_target = torch.zeros_like(obj_pred)
            reg_target = torch.zeros_like(reg_pred)

            for b in range(B):
                if targets[b] is None or len(targets[b].get("boxes", [])) == 0:
                    continue

                gt_boxes = targets[b]["boxes"]  # [N, 4] xyxy
                gt_labels = targets[b]["labels"]  # [N]

                # Compute GT centers in grid coordinates
                cx_float = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2 / stride
                cy_float = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2 / stride
                cx = cx_float.long().clamp(0, W - 1)
                cy = cy_float.long().clamp(0, H - 1)

                for n in range(len(gt_boxes)):
                    gx, gy = cx[n], cy[n]
                    label = gt_labels[n].clamp(0, num_cls - 1)

                    box = gt_boxes[n]
                    box_cx = (box[0] + box[2]) / 2 / stride
                    box_cy = (box[1] + box[3]) / 2 / stride
                    w = (box[2] - box[0]) / stride
                    h = (box[3] - box[1]) / stride
                    log_w = torch.log(w.clamp(min=1e-6))
                    log_h = torch.log(h.clamp(min=1e-6))

                    # Multi-positive assignment: center + 3×3 neighborhood
                    for dy in range(-1, 2):
                        for dx in range(-1, 2):
                            ny = (gy + dy).clamp(0, H - 1)
                            nx = (gx + dx).clamp(0, W - 1)

                            obj_target[b, 0, ny, nx] = 1.0
                            cls_target[b, label, ny, nx] = 1.0

                            reg_target[b, 0, ny, nx] = box_cx - nx.float()
                            reg_target[b, 1, ny, nx] = box_cy - ny.float()
                            reg_target[b, 2, ny, nx] = log_w
                            reg_target[b, 3, ny, nx] = log_h

            # Focal BCE for classification
            cls_loss = self._focal_bce(cls_pred, cls_target)
            total_cls_loss = total_cls_loss + cls_loss

            # BCE for objectness
            obj_loss = F.binary_cross_entropy_with_logits(obj_pred, obj_target)
            total_obj_loss = total_obj_loss + obj_loss

            # L1 loss for regression (only on positive samples)
            pos_mask = obj_target > 0
            if pos_mask.sum() > 0:
                reg_loss = F.l1_loss(
                    reg_pred[pos_mask.expand_as(reg_pred)],
                    reg_target[pos_mask.expand_as(reg_pred)],
                )
                total_reg_loss = total_reg_loss + reg_loss

        # Loss weights
        total_loss = (
            1.0 * total_cls_loss
            + 5.0 * total_reg_loss
            + 1.0 * total_obj_loss
        )

        return {
            "loss": total_loss,
            "cls_loss": total_cls_loss,
            "reg_loss": total_reg_loss,
            "obj_loss": total_obj_loss,
        }

    @staticmethod
    def _focal_bce(
        pred: torch.Tensor,
        target: torch.Tensor,
        alpha: float = 0.25,
        gamma: float = 2.0,
    ) -> torch.Tensor:
        """Focal Binary Cross-Entropy Loss."""
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        p = torch.sigmoid(pred)
        p_t = p * target + (1 - p) * (1 - target)
        focal_weight = (1 - p_t) ** gamma
        alpha_t = alpha * target + (1 - alpha) * (1 - target)
        loss = alpha_t * focal_weight * bce
        return loss.mean()

    def reset_temporal_state(self):
        """Reset temporal buffer (call at video boundary or scene change)."""
        if self.temporal_buffer is not None:
            self.temporal_buffer.reset()

    @torch.no_grad()
    def get_model_info(self) -> Dict:
        """Get model statistics for paper reporting."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # Component-wise breakdown
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        neck_params = sum(p.numel() for p in self.neck.parameters())
        head_params = sum(p.numel() for p in self.head.parameters())
        nas_params = 0
        if self.nas_module is not None:
            nas_params = sum(p.numel() for p in self.nas_module.parameters())
        noise_params = 0
        if self.noise_estimator is not None:
            noise_params = sum(p.numel() for p in self.noise_estimator.parameters())

        return {
            "model_scale": self.model_scale,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "total_params_M": total_params / 1e6,
            "backbone_params_M": backbone_params / 1e6,
            "neck_params_M": neck_params / 1e6,
            "head_params_M": head_params / 1e6,
            "nas_module_params_M": nas_params / 1e6,
            "noise_estimator_params_M": noise_params / 1e6,
            "nas_overhead_pct": (nas_params + noise_params) / total_params * 100 if total_params > 0 else 0,
        }


class NASYOLOWithTCL(NASYOLO):
    """NAS-YOLO with Temporal Consistency Loss (TCL).

    Extends the base NAS-YOLO with a loss term that encourages temporally
    consistent predictions across frames. This is used during training
    with pseudo-temporal sequences.

    TCL = Σ_t ||D_t - EMA(D_{<t})||² × confidence_t

    where D_t are decoded detections at frame t.

    Args:
        tcl_weight: Weight for temporal consistency loss.
        tcl_ema: EMA momentum for temporal reference.
        **kwargs: Arguments for base NASYOLO.
    """

    def __init__(self, tcl_weight: float = 0.5, tcl_ema: float = 0.95, **kwargs):
        super().__init__(**kwargs)
        self.tcl_weight = tcl_weight
        self.tcl_ema = tcl_ema
        self._prev_pred_ema = None

    def forward_temporal_with_tcl(
        self,
        image_sequence: List[torch.Tensor],
        targets_sequence: Optional[List[List[Dict]]] = None,
    ) -> Tuple[List[Dict], torch.Tensor]:
        """Forward pass with temporal consistency loss computation.

        Args:
            image_sequence: List of T images.
            targets_sequence: Optional list of T target dicts.

        Returns:
            outputs_sequence: List of output dicts per frame.
            tcl_loss: Temporal consistency loss scalar.
        """
        if self.temporal_buffer is not None:
            self.temporal_buffer.reset()
        self._prev_pred_ema = None

        outputs_sequence = []
        tcl_losses = []

        for t, images in enumerate(image_sequence):
            targets = targets_sequence[t] if targets_sequence is not None else None
            outputs = self.forward(images, targets)

            # Compute TCL against EMA of previous predictions
            if self._prev_pred_ema is not None and self.training:
                tcl = self._compute_tcl(outputs)
                tcl_losses.append(tcl)

            # Update EMA prediction reference
            self._update_pred_ema(outputs)
            outputs_sequence.append(outputs)

        device = image_sequence[0].device if image_sequence else torch.device('cpu')
        tcl_loss = torch.stack(tcl_losses).mean() if tcl_losses else torch.tensor(0.0, device=device)
        return outputs_sequence, tcl_loss * self.tcl_weight

    def _compute_tcl(self, outputs: Dict) -> torch.Tensor:
        """Compute temporal consistency loss against EMA reference."""
        tcl = torch.tensor(0.0, device=outputs["cls_preds"][0].device)

        for i in range(len(outputs["cls_preds"])):
            curr_obj = torch.sigmoid(outputs["obj_preds"][i])
            prev_obj = self._prev_pred_ema["obj_preds"][i]

            curr_cls = torch.sigmoid(outputs["cls_preds"][i])
            prev_cls = self._prev_pred_ema["cls_preds"][i]

            # Objectness consistency
            tcl = tcl + F.mse_loss(curr_obj, prev_obj)
            # Classification consistency (weighted by objectness)
            weight = (curr_obj + prev_obj) / 2
            tcl = tcl + (weight * F.mse_loss(curr_cls, prev_cls, reduction="none").mean(dim=1, keepdim=True)).mean()

        return tcl / len(outputs["cls_preds"])

    @torch.no_grad()
    def _update_pred_ema(self, outputs: Dict):
        """Update EMA of predictions for TCL computation."""
        if self._prev_pred_ema is None:
            self._prev_pred_ema = {
                "obj_preds": [torch.sigmoid(o).detach() for o in outputs["obj_preds"]],
                "cls_preds": [torch.sigmoid(c).detach() for c in outputs["cls_preds"]],
            }
        else:
            for i in range(len(outputs["obj_preds"])):
                self._prev_pred_ema["obj_preds"][i] = (
                    self.tcl_ema * self._prev_pred_ema["obj_preds"][i]
                    + (1 - self.tcl_ema) * torch.sigmoid(outputs["obj_preds"][i]).detach()
                )
                self._prev_pred_ema["cls_preds"][i] = (
                    self.tcl_ema * self._prev_pred_ema["cls_preds"][i]
                    + (1 - self.tcl_ema) * torch.sigmoid(outputs["cls_preds"][i]).detach()
                )
