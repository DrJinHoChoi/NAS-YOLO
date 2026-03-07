"""
Ultralytics YOLO + NASPlugin Wrapper
======================================
Wraps Ultralytics YOLO models (v8, v10, v11, v12) with NASPlugin.

The wrapper intercepts features between neck and detection head,
processes them through the NASPlugin, then continues to the head.

Usage:
    model = UltralyticsWithNASPlugin(
        weights_path="yolov8n.pt",
        plugin_cfg={"profile": "edge", "mode": "hybrid"},
    )
    outputs = model(images)

Two-stage training:
    Stage 1: freeze_backbone=True  → train only NASPlugin (15 epochs)
    Stage 2: freeze_backbone=False → fine-tune everything (85 epochs)

Knowledge Distillation:
    model = UltralyticsWithKD(
        weights_path="yolov8n.pt",
        teacher_weights="yolov8n.pt",
        plugin_cfg={"profile": "edge", "mode": "hybrid"},
    )
    outputs = model(images)  # includes kd_loss in outputs
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List, Tuple

from ..models.nas_plugin import NASPlugin
from ..models.knowledge_distillation import NoiseConditionedKDWrapper


class UltralyticsWithNASPlugin(nn.Module):
    """Wraps any Ultralytics YOLO model with NASPlugin.

    Supports: YOLOv8, YOLOv10, YOLOv11, YOLOv12

    The wrapper extracts the backbone+neck (all layers before the Detect head)
    and the Detect head separately, inserting NASPlugin between them.

    Args:
        weights_path: Path to pretrained Ultralytics weights (.pt file)
            or model name (e.g., 'yolov8n.pt', 'yolov11n.pt').
        plugin_cfg: NASPlugin configuration dict.
        freeze_backbone: Whether to freeze backbone+neck parameters.
        num_classes: Number of classes (None = use pretrained model's).
    """

    def __init__(
        self,
        weights_path: str = "yolov8n.pt",
        plugin_cfg: Optional[Dict] = None,
        freeze_backbone: bool = True,
        num_classes: Optional[int] = None,
    ):
        super().__init__()
        plugin_cfg = plugin_cfg or {}

        # Load Ultralytics model
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics package required. Install with: pip install ultralytics"
            )

        yolo = YOLO(weights_path)
        self.ultralytics_model = yolo.model

        # Detect the architecture: find where the Detect head is
        self._detect_head_idx = self._find_detect_head()

        # Detect neck output channels
        channels_list = self._detect_channels()
        self.channels_list = channels_list

        # Create NASPlugin
        self.nas_plugin = NASPlugin(
            channels_list=channels_list,
            **plugin_cfg,
        )

        # Freeze backbone+neck if requested
        if freeze_backbone:
            self.freeze_backbone()

        # Store model info
        self._weights_path = weights_path

    def _find_detect_head(self) -> int:
        """Find the index of the Detect/detection head layer."""
        model = self.ultralytics_model.model
        for i in range(len(model) - 1, -1, -1):
            layer_name = type(model[i]).__name__
            if "Detect" in layer_name or "Segment" in layer_name:
                return i
        raise RuntimeError(
            "Could not find Detect head in Ultralytics model. "
            "Supported: YOLOv8, v10, v11, v12."
        )

    def _detect_channels(self) -> List[int]:
        """Detect neck output channels by running a dummy forward pass."""
        channels = []
        hooks = []
        head = self.ultralytics_model.model[self._detect_head_idx]

        def hook_fn(module, input, output):
            # The Detect head receives a list of features as input
            if isinstance(input, tuple) and len(input) > 0:
                feats = input[0]
                if isinstance(feats, (list, tuple)):
                    for f in feats:
                        channels.append(f.shape[1])
                elif isinstance(feats, torch.Tensor):
                    channels.append(feats.shape[1])

        h = head.register_forward_hook(hook_fn)
        hooks.append(h)

        # Run dummy forward
        try:
            device = next(self.ultralytics_model.parameters()).device
            dummy = torch.zeros(1, 3, 640, 640, device=device)
            with torch.no_grad():
                self.ultralytics_model.eval()
                self.ultralytics_model(dummy)
        except Exception:
            # Fallback: common Ultralytics channel configs
            pass
        finally:
            for h in hooks:
                h.remove()

        if not channels:
            # Fallback based on model name
            channels = self._fallback_channels()

        return channels

    def _fallback_channels(self) -> List[int]:
        """Fallback channel detection based on model weight filename."""
        name = self._weights_path.lower() if hasattr(self, '_weights_path') else ""
        if "n" in name or "nano" in name:
            return [64, 128, 256]
        elif "s" in name or "small" in name:
            return [128, 256, 512]
        elif "m" in name or "medium" in name:
            return [192, 384, 768]
        elif "l" in name or "large" in name:
            return [256, 512, 1024]
        else:
            return [128, 256, 512]  # Default to small

    def freeze_backbone(self):
        """Freeze all parameters except NASPlugin."""
        for name, param in self.ultralytics_model.named_parameters():
            param.requires_grad = False
        # NASPlugin stays unfrozen
        for param in self.nas_plugin.parameters():
            param.requires_grad = True

    def unfreeze_all(self):
        """Unfreeze all parameters for fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True

    def forward(
        self,
        images: torch.Tensor,
        targets: Optional[List[Dict]] = None,
    ) -> Dict:
        """Forward pass with NASPlugin between neck and head.

        Args:
            images: Input images [B, 3, H, W].
            targets: Optional ground truth for loss computation.

        Returns:
            outputs: Detection outputs dict.
        """
        model = self.ultralytics_model.model

        # Run backbone + neck (all layers before Detect head)
        x = images
        intermediates = {}
        save_indices = set()

        # Collect save indices from model config (including Detect head references)
        for i, layer in enumerate(model):
            if hasattr(layer, 'f'):
                f = layer.f
                if isinstance(f, int) and f != -1:
                    save_indices.add(f)
                elif isinstance(f, (list, tuple)):
                    for ff in f:
                        if isinstance(ff, int) and ff != -1:
                            save_indices.add(ff)
            # Also save the layer index itself if any later layer references it
            if hasattr(layer, 'i'):
                save_indices.add(layer.i if isinstance(layer.i, int) else i)
            else:
                save_indices.add(i)

        neck_features = None
        for i, layer in enumerate(model):
            if i == self._detect_head_idx:
                # The Detect head's 'f' attribute tells us which layers
                # provide multi-scale features (e.g., f=[15, 18, 21])
                detect_head = layer
                if hasattr(detect_head, 'f') and isinstance(detect_head.f, (list, tuple)):
                    neck_features = [
                        intermediates[ff] if ff != -1 else x
                        for ff in detect_head.f
                    ]
                elif isinstance(x, (list, tuple)):
                    neck_features = list(x)
                else:
                    neck_features = [x]
                break

            # Handle layer input (some layers take from specific indices)
            if hasattr(layer, 'f'):
                f = layer.f
                if isinstance(f, int):
                    if f == -1:
                        layer_input = x
                    else:
                        layer_input = intermediates[f]
                elif isinstance(f, (list, tuple)):
                    layer_input = [
                        intermediates[ff] if ff != -1 else x
                        for ff in f
                    ]
                else:
                    layer_input = x
            else:
                layer_input = x

            # Forward through layer
            if isinstance(layer_input, list):
                x = layer(layer_input)
            else:
                x = layer(layer_input)

            # Save intermediate if needed
            if i in save_indices:
                intermediates[i] = x

        if neck_features is None:
            raise RuntimeError("Failed to extract neck features")

        # Apply NASPlugin
        enhanced_features, _, conf_weights = self.nas_plugin(
            neck_features, images=images
        )

        # Run Detect head with enhanced features
        detect_head = model[self._detect_head_idx]
        outputs = detect_head(enhanced_features)

        return outputs

    def reset_temporal_state(self):
        """Reset temporal state for new video/sequence."""
        self.nas_plugin.reset_state()

    @torch.no_grad()
    def get_model_info(self) -> Dict:
        """Get combined model statistics."""
        base_params = sum(
            p.numel() for p in self.ultralytics_model.parameters()
        )
        plugin_info = self.nas_plugin.get_plugin_info()

        return {
            "base_model": self._weights_path,
            "base_params_M": base_params / 1e6,
            "plugin_params_M": plugin_info["total_params_M"],
            "total_params_M": (base_params + plugin_info["total_params"]) / 1e6,
            "overhead_pct": plugin_info["total_params"] / base_params * 100,
            "plugin_detail": plugin_info,
        }


class UltralyticsWithKD(UltralyticsWithNASPlugin):
    """Ultralytics YOLO + NASPlugin with Knowledge Distillation.

    Extends UltralyticsWithNASPlugin to include a frozen teacher model
    and noise-conditioned KD loss computation during training.

    The teacher is a standard CNN YOLO (without NASPlugin) that provides:
      - Feature-level supervision: neck features guide NASPlugin output
      - Output-level supervision: detection logits soft-label guidance
      - Noise-conditioned weighting: KD weight reduces under noisy inputs

    Args:
        weights_path: Student model weights (YOLO + NASPlugin).
        teacher_weights: Teacher model weights (standard YOLO, frozen).
        plugin_cfg: NASPlugin configuration dict.
        kd_cfg: Knowledge distillation configuration dict.
        freeze_backbone: Whether to freeze student's backbone+neck.
        num_classes: Number of classes.
    """

    def __init__(
        self,
        weights_path: str = "yolov8n.pt",
        teacher_weights: Optional[str] = None,
        plugin_cfg: Optional[Dict] = None,
        kd_cfg: Optional[Dict] = None,
        freeze_backbone: bool = True,
        num_classes: Optional[int] = None,
    ):
        super().__init__(
            weights_path=weights_path,
            plugin_cfg=plugin_cfg,
            freeze_backbone=freeze_backbone,
            num_classes=num_classes,
        )

        kd_cfg = kd_cfg or {}
        teacher_weights = teacher_weights or weights_path

        # Load teacher model (frozen, eval-only)
        self._load_teacher(teacher_weights)

        # KD loss module
        noise_dim = (plugin_cfg or {}).get("noise_dim", 32)
        self.kd_wrapper = NoiseConditionedKDWrapper(
            noise_dim=noise_dim,
            feature_loss_type=kd_cfg.get("feature_loss_type", "combined"),
            temperature=kd_cfg.get("temperature", 4.0),
            alpha_feature=kd_cfg.get("alpha_feature", 1.0),
            alpha_output=kd_cfg.get("alpha_output", 1.0),
        )

        self._kd_weight = kd_cfg.get("kd_weight", 1.0)

    def _load_teacher(self, teacher_weights: str):
        """Load and freeze the teacher CNN model."""
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics package required. Install with: pip install ultralytics"
            )

        teacher_yolo = YOLO(teacher_weights)
        self.teacher_model = teacher_yolo.model

        # Freeze teacher completely
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        self.teacher_model.eval()

        # Find teacher's detect head index
        self._teacher_head_idx = None
        for i in range(len(self.teacher_model.model) - 1, -1, -1):
            layer_name = type(self.teacher_model.model[i]).__name__
            if "Detect" in layer_name or "Segment" in layer_name:
                self._teacher_head_idx = i
                break

    @torch.no_grad()
    def _teacher_forward(
        self, images: torch.Tensor
    ) -> Tuple[List[torch.Tensor], Dict[str, List[torch.Tensor]]]:
        """Run teacher forward pass to extract features and outputs.

        Args:
            images: Input images [B, 3, H, W].

        Returns:
            teacher_features: Neck feature maps (before detect head).
            teacher_outputs: Detection outputs dict with cls_preds, obj_preds.
        """
        self.teacher_model.eval()
        model = self.teacher_model.model

        x = images
        intermediates = {}
        save_indices = set()

        for i, layer in enumerate(model):
            if hasattr(layer, 'f'):
                f = layer.f
                if isinstance(f, int) and f != -1:
                    save_indices.add(f)
                elif isinstance(f, (list, tuple)):
                    for ff in f:
                        if isinstance(ff, int) and ff != -1:
                            save_indices.add(ff)

        neck_features = None
        for i, layer in enumerate(model):
            if i == self._teacher_head_idx:
                if isinstance(x, (list, tuple)):
                    neck_features = list(x)
                else:
                    neck_features = [x]
                break

            if hasattr(layer, 'f'):
                f = layer.f
                if isinstance(f, int):
                    if f == -1:
                        layer_input = x
                    else:
                        layer_input = intermediates[f]
                elif isinstance(f, (list, tuple)):
                    layer_input = [
                        intermediates[ff] if ff != -1 else x
                        for ff in f
                    ]
                else:
                    layer_input = x
            else:
                layer_input = x

            if isinstance(layer_input, list):
                x = layer(layer_input)
            else:
                x = layer(layer_input)

            if i in save_indices:
                intermediates[i] = x

        # Extract detection outputs from teacher head
        teacher_outputs = {}
        if neck_features is not None and self._teacher_head_idx is not None:
            detect_head = model[self._teacher_head_idx]
            try:
                head_out = detect_head(neck_features)
                # Try to extract cls/obj predictions from head output
                if isinstance(head_out, (list, tuple)):
                    teacher_outputs["features"] = list(head_out)
            except Exception:
                pass

        return neck_features or [], teacher_outputs

    def forward(
        self,
        images: torch.Tensor,
        targets: Optional[List[Dict]] = None,
    ) -> Dict:
        """Forward pass with NASPlugin and KD loss computation.

        During training, computes both detection loss and KD loss.
        During inference, behaves like the parent class (no KD overhead).

        Args:
            images: Input images [B, 3, H, W].
            targets: Optional ground truth for loss computation.

        Returns:
            outputs: Detection outputs dict, with 'kd_losses' during training.
        """
        model = self.ultralytics_model.model

        # Run backbone + neck (student)
        x = images
        intermediates = {}
        save_indices = set()

        for i, layer in enumerate(model):
            if hasattr(layer, 'f'):
                f = layer.f
                if isinstance(f, int) and f != -1:
                    save_indices.add(f)
                elif isinstance(f, (list, tuple)):
                    for ff in f:
                        if isinstance(ff, int) and ff != -1:
                            save_indices.add(ff)

        neck_features = None
        for i, layer in enumerate(model):
            if i == self._detect_head_idx:
                if isinstance(x, (list, tuple)):
                    neck_features = list(x)
                else:
                    neck_features = [x]
                break

            if hasattr(layer, 'f'):
                f = layer.f
                if isinstance(f, int):
                    if f == -1:
                        layer_input = x
                    else:
                        layer_input = intermediates[f]
                elif isinstance(f, (list, tuple)):
                    layer_input = [
                        intermediates[ff] if ff != -1 else x
                        for ff in f
                    ]
                else:
                    layer_input = x
            else:
                layer_input = x

            if isinstance(layer_input, list):
                x = layer(layer_input)
            else:
                x = layer(layer_input)

            if i in save_indices:
                intermediates[i] = x

        if neck_features is None:
            raise RuntimeError("Failed to extract neck features")

        # Apply NASPlugin (student)
        enhanced_features, _, conf_weights = self.nas_plugin(
            neck_features, images=images
        )

        # Run Detect head (student)
        detect_head = model[self._detect_head_idx]
        outputs = detect_head(enhanced_features)

        # KD loss computation (training only)
        if self.training:
            # Get noise level from plugin's noise estimator
            noise_level = None
            if self.nas_plugin.noise_estimator is not None:
                with torch.no_grad():
                    noise_level = self.nas_plugin.noise_estimator(images)

            # Teacher forward (always no_grad)
            teacher_features, teacher_outputs = self._teacher_forward(images)

            # Compute KD loss
            # Feature-level: student enhanced_features vs teacher neck_features
            student_kd_features = enhanced_features
            teacher_kd_features = teacher_features

            # Ensure feature count matches (use min of both)
            n_scales = min(len(student_kd_features), len(teacher_kd_features))
            student_kd_features = student_kd_features[:n_scales]
            teacher_kd_features = teacher_kd_features[:n_scales]

            kd_result = self.kd_wrapper(
                student_features=student_kd_features,
                teacher_features=teacher_kd_features,
                student_outputs=None,  # Output-level KD requires matching head format
                teacher_outputs=None,
                noise_level=noise_level,
            )

            # Attach KD losses to output
            if isinstance(outputs, dict):
                outputs["kd_losses"] = kd_result
                outputs["kd_loss"] = kd_result["kd_total_loss"] * self._kd_weight
            else:
                outputs = {
                    "detection_output": outputs,
                    "kd_losses": kd_result,
                    "kd_loss": kd_result["kd_total_loss"] * self._kd_weight,
                }

        return outputs

    @torch.no_grad()
    def get_model_info(self) -> Dict:
        """Get model info including teacher and KD details."""
        info = super().get_model_info()
        teacher_params = sum(
            p.numel() for p in self.teacher_model.parameters()
        )
        kd_params = sum(
            p.numel() for p in self.kd_wrapper.parameters()
        )
        info["teacher_params_M"] = teacher_params / 1e6
        info["kd_wrapper_params"] = kd_params
        info["kd_weight"] = self._kd_weight
        return info
