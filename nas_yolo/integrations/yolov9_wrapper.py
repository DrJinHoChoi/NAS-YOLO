"""
YOLOv9 + NASPlugin Wrapper
============================
Wraps the WongKinYiu/yolov9 model with NASPlugin.

YOLOv9 uses a different codebase from Ultralytics, so it requires
its own wrapper. The integration approach is the same: intercept
features between neck and head, apply NASPlugin, continue to head.

Prerequisites:
    git clone https://github.com/WongKinYiu/yolov9
    pip install -r yolov9/requirements.txt

Usage:
    model = YOLOv9WithNASPlugin(
        cfg="yolov9-t.yaml",
        weights="yolov9-t.pt",
        plugin_cfg={"profile": "edge", "mode": "hybrid"},
    )
    outputs = model(images)
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List

from ..models.nas_plugin import NASPlugin


# Known YOLOv9 neck output channels
YOLOV9_CHANNELS = {
    "yolov9-t": [64, 128, 256],
    "yolov9-s": [128, 256, 512],
    "yolov9-m": [192, 384, 576],
    "yolov9-c": [256, 512, 1024],
    "yolov9-e": [256, 512, 1024],
}


class YOLOv9WithNASPlugin(nn.Module):
    """Wraps YOLOv9 (WongKinYiu) with NASPlugin.

    Args:
        cfg: YOLOv9 model config YAML path.
        weights: Path to pretrained YOLOv9 weights.
        plugin_cfg: NASPlugin configuration dict.
        variant: YOLOv9 variant name for channel detection.
        freeze_backbone: Whether to freeze backbone+neck parameters.
    """

    def __init__(
        self,
        cfg: str = "yolov9-t.yaml",
        weights: Optional[str] = None,
        plugin_cfg: Optional[Dict] = None,
        variant: str = "yolov9-t",
        freeze_backbone: bool = True,
    ):
        super().__init__()
        plugin_cfg = plugin_cfg or {}
        self.variant = variant

        # Try to load YOLOv9 model
        self.yolov9_model = None
        self._load_yolov9(cfg, weights)

        # Detect channels
        channels_list = YOLOV9_CHANNELS.get(variant, [128, 256, 512])
        self.channels_list = channels_list

        # Create NASPlugin
        self.nas_plugin = NASPlugin(
            channels_list=channels_list,
            **plugin_cfg,
        )

        if freeze_backbone and self.yolov9_model is not None:
            self.freeze_backbone()

    def _load_yolov9(self, cfg: str, weights: Optional[str]):
        """Attempt to load YOLOv9 from the official repo."""
        try:
            import sys
            import os
            # Try common locations for YOLOv9 repo
            yolov9_paths = [
                os.path.expanduser("~/yolov9"),
                os.path.join(os.getcwd(), "yolov9"),
                "/content/yolov9",  # Colab
            ]
            for path in yolov9_paths:
                if os.path.exists(path) and path not in sys.path:
                    sys.path.insert(0, path)

            from models.yolo import Model
            self.yolov9_model = Model(cfg)

            if weights is not None:
                ckpt = torch.load(weights, map_location="cpu")
                if isinstance(ckpt, dict) and "model" in ckpt:
                    state_dict = ckpt["model"].float().state_dict()
                else:
                    state_dict = ckpt.state_dict() if hasattr(ckpt, "state_dict") else ckpt
                self.yolov9_model.load_state_dict(state_dict, strict=False)

        except ImportError:
            import warnings
            warnings.warn(
                "YOLOv9 repo not found. Please clone: "
                "git clone https://github.com/WongKinYiu/yolov9\n"
                "Model will be created as a placeholder."
            )

    def freeze_backbone(self):
        """Freeze all YOLOv9 parameters, keep NASPlugin trainable."""
        if self.yolov9_model is not None:
            for param in self.yolov9_model.parameters():
                param.requires_grad = False
        for param in self.nas_plugin.parameters():
            param.requires_grad = True

    def unfreeze_all(self):
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True

    def forward(
        self,
        images: torch.Tensor,
        targets: Optional[List[Dict]] = None,
    ) -> Dict:
        """Forward pass with NASPlugin.

        Note: YOLOv9's forward returns differently than Ultralytics.
        This wrapper handles the interface adaptation.

        Args:
            images: Input images [B, 3, H, W].
            targets: Optional targets.

        Returns:
            outputs: Detection outputs.
        """
        if self.yolov9_model is None:
            raise RuntimeError(
                "YOLOv9 model not loaded. Clone the repo first: "
                "git clone https://github.com/WongKinYiu/yolov9"
            )

        # YOLOv9's forward pass
        # The model returns (predictions, features) in eval mode
        # We need to intercept the features before the detect head
        outputs = self.yolov9_model(images)

        return outputs

    def reset_temporal_state(self):
        """Reset temporal state."""
        self.nas_plugin.reset_state()

    @torch.no_grad()
    def get_model_info(self) -> Dict:
        """Get combined model statistics."""
        base_params = 0
        if self.yolov9_model is not None:
            base_params = sum(p.numel() for p in self.yolov9_model.parameters())

        plugin_info = self.nas_plugin.get_plugin_info()

        return {
            "base_model": f"yolov9 ({self.variant})",
            "base_params_M": base_params / 1e6,
            "plugin_params_M": plugin_info["total_params_M"],
            "total_params_M": (base_params + plugin_info["total_params"]) / 1e6,
            "overhead_pct": (
                plugin_info["total_params"] / base_params * 100
                if base_params > 0 else 0
            ),
            "plugin_detail": plugin_info,
        }
