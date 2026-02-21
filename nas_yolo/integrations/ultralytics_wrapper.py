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
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List, Tuple

from ..models.nas_plugin import NASPlugin


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

        # Collect save indices from model config
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
                # Intercept: x should be the neck features (list)
                if isinstance(x, (list, tuple)):
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

        # Run Detect head
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
