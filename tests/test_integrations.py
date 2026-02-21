"""Tests for YOLO integration wrappers.

Note: These tests require the ultralytics package.
Run with: pytest tests/test_integrations.py -v
"""

import pytest
import torch


@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def ultralytics_available():
    try:
        import ultralytics
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not ultralytics_available(), reason="ultralytics not installed")
class TestUltralyticsWrapper:
    def test_yolov8n_forward(self, device):
        from nas_yolo.integrations import UltralyticsWithNASPlugin
        model = UltralyticsWithNASPlugin(
            weights_path="yolov8n.pt",
            plugin_cfg={"profile": "edge", "mode": "hybrid"},
            freeze_backbone=True,
        ).to(device)

        images = torch.randn(1, 3, 640, 640, device=device)
        with torch.no_grad():
            outputs = model(images)
        assert outputs is not None

    def test_model_info(self, device):
        from nas_yolo.integrations import UltralyticsWithNASPlugin
        model = UltralyticsWithNASPlugin(
            weights_path="yolov8n.pt",
            plugin_cfg={"profile": "edge"},
        ).to(device)

        info = model.get_model_info()
        assert "base_params_M" in info
        assert "plugin_params_M" in info
        assert "overhead_pct" in info
        assert info["overhead_pct"] > 0

    def test_freeze_unfreeze(self, device):
        from nas_yolo.integrations import UltralyticsWithNASPlugin
        model = UltralyticsWithNASPlugin(
            weights_path="yolov8n.pt",
            plugin_cfg={"profile": "edge"},
            freeze_backbone=True,
        ).to(device)

        # After freeze: only plugin params should be trainable
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        plugin_params = sum(p.numel() for p in model.nas_plugin.parameters())
        assert trainable == plugin_params

        # After unfreeze: all params should be trainable
        model.unfreeze_all()
        trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        assert trainable_after == total

    def test_temporal_reset(self, device):
        from nas_yolo.integrations import UltralyticsWithNASPlugin
        model = UltralyticsWithNASPlugin(
            weights_path="yolov8n.pt",
            plugin_cfg={"profile": "edge"},
        ).to(device)

        model.reset_temporal_state()
        assert model.nas_plugin.temporal_buffer.frame_count == 0


class TestYOLOv9Wrapper:
    def test_initialization(self, device):
        """Test that YOLOv9 wrapper can be created (without loading model)."""
        from nas_yolo.integrations import YOLOv9WithNASPlugin
        # This will warn about missing YOLOv9 repo but shouldn't crash
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = YOLOv9WithNASPlugin(
                variant="yolov9-t",
                plugin_cfg={"profile": "edge"},
            )

        info = model.get_model_info()
        assert info["plugin_params_M"] > 0

    def test_plugin_params(self, device):
        from nas_yolo.integrations import YOLOv9WithNASPlugin
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = YOLOv9WithNASPlugin(
                variant="yolov9-t",
                plugin_cfg={"profile": "edge", "mode": "hybrid"},
            )

        # Verify plugin has expected structure
        assert hasattr(model, "nas_plugin")
        assert len(model.nas_plugin.blocks) == 3
