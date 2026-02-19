"""
Integration Tests for Full NAS-YOLO Model
============================================
End-to-end tests for model construction, forward pass, and loss computation.
"""

import torch
import pytest
from nas_yolo.models.nas_yolo import NASYOLO, NASYOLOWithTCL, MODEL_CONFIGS


class TestNASYOLO:
    """Integration tests for the full model."""

    @pytest.mark.parametrize("scale", ["nano", "small", "medium"])
    def test_forward_pass(self, scale):
        """Forward pass should produce valid detection outputs."""
        model = NASYOLO(num_classes=80, model_scale=scale)
        model.eval()

        x = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            outputs = model(x)

        assert "cls_preds" in outputs
        assert "reg_preds" in outputs
        assert "obj_preds" in outputs
        assert len(outputs["cls_preds"]) == 3  # 3 scales

    def test_forward_with_targets(self):
        """Forward with targets should produce losses."""
        model = NASYOLO(num_classes=10, model_scale="nano")
        model.train()

        x = torch.randn(2, 3, 320, 320)
        targets = [
            {"boxes": torch.tensor([[50, 50, 150, 150.0]]),
             "labels": torch.tensor([3])},
            {"boxes": torch.tensor([[20, 20, 100, 100.0]]),
             "labels": torch.tensor([1])},
        ]

        outputs = model(x, targets)

        assert "losses" in outputs
        assert "loss" in outputs["losses"]
        assert outputs["losses"]["loss"].item() > 0

    def test_temporal_forward(self):
        """Temporal forward should process sequence and propagate state."""
        model = NASYOLO(num_classes=10, model_scale="nano")
        model.eval()

        sequence = [torch.randn(1, 3, 320, 320) for _ in range(5)]

        with torch.no_grad():
            outputs_seq = model.forward_temporal(sequence)

        assert len(outputs_seq) == 5
        for out in outputs_seq:
            assert "cls_preds" in out

    def test_ablation_no_temporal(self):
        """Model without temporal should still work."""
        model = NASYOLO(
            num_classes=10, model_scale="nano",
            use_temporal=False, use_noise_gate=False,
        )
        model.eval()

        x = torch.randn(1, 3, 320, 320)
        with torch.no_grad():
            outputs = model(x)

        assert "cls_preds" in outputs

    def test_ablation_no_noise_gate(self):
        """Model with temporal but no noise gate should work."""
        model = NASYOLO(
            num_classes=10, model_scale="nano",
            use_temporal=True, use_noise_gate=False,
        )
        model.eval()

        x = torch.randn(1, 3, 320, 320)
        with torch.no_grad():
            outputs = model(x)

        assert "cls_preds" in outputs

    def test_model_info(self):
        """Model info should report correct statistics."""
        model = NASYOLO(num_classes=80, model_scale="small")
        info = model.get_model_info()

        assert info["model_scale"] == "small"
        assert info["total_params_M"] > 0
        assert info["nas_overhead_pct"] > 0
        assert info["backbone_params_M"] > 0

    def test_reset_temporal_state(self):
        """Reset should clear temporal buffer."""
        model = NASYOLO(num_classes=10, model_scale="nano")

        x = torch.randn(1, 3, 320, 320)
        model.eval()
        with torch.no_grad():
            model(x)  # Populate state

        assert model.temporal_buffer.is_initialized

        model.reset_temporal_state()
        assert not model.temporal_buffer.is_initialized


class TestNASYOLOWithTCL:
    """Tests for the TCL-enhanced model."""

    def test_tcl_forward(self):
        """TCL forward should return outputs + TCL loss."""
        model = NASYOLOWithTCL(
            num_classes=10, model_scale="nano",
            tcl_weight=0.5,
        )
        model.train()

        sequence = [torch.randn(2, 3, 320, 320) for _ in range(3)]
        targets_seq = [
            [{"boxes": torch.tensor([[50, 50, 150, 150.0]]),
              "labels": torch.tensor([3])},
             {"boxes": torch.tensor([[20, 20, 100, 100.0]]),
              "labels": torch.tensor([1])}]
            for _ in range(3)
        ]

        outputs_seq, tcl_loss = model.forward_temporal_with_tcl(
            sequence, targets_seq
        )

        assert len(outputs_seq) == 3
        assert isinstance(tcl_loss, torch.Tensor)


class TestModelScaling:
    """Tests for model scaling (params should increase with scale)."""

    def test_param_ordering(self):
        """Larger scales should have more parameters."""
        param_counts = {}
        for scale in ["nano", "small", "medium"]:
            model = NASYOLO(num_classes=80, model_scale=scale)
            params = sum(p.numel() for p in model.parameters())
            param_counts[scale] = params

        assert param_counts["nano"] < param_counts["small"]
        assert param_counts["small"] < param_counts["medium"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
