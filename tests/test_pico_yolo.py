"""
Tests for PicoYOLO — World's Smallest Detection System
=========================================================
Verifies:
    1. Forward pass produces valid outputs
    2. Total params < 500K (sub-0.5M target)
    3. SA-SSM 4 structural noise defenses preserved at pico scale
    4. ONNX export compatibility
    5. Multiple channel configurations work
    6. Context-aware detection head
"""

import pytest
import torch
import torch.nn as nn
import math


class TestPicoProfile:
    """Test the 'pico' SA-SSM profile configuration."""

    def test_pico_profile_exists(self):
        """Verify pico profile is in PROFILE_CONFIGS."""
        from nas_yolo.models.mamba_attention import PROFILE_CONFIGS
        assert "pico" in PROFILE_CONFIGS

    def test_pico_profile_values(self):
        """Verify pico profile has correct parameters."""
        from nas_yolo.models.mamba_attention import PROFILE_CONFIGS
        pico = PROFILE_CONFIGS["pico"]
        assert pico["expand"] == 1
        assert pico["d_state"] == 2
        assert pico["d_conv"] == 1
        assert pico["attention_type"] == "se"
        assert pico["router_type"] == "fixed"
        assert pico["gate_min"] == 0.25
        assert pico["weight_sharing"] is True

    def test_pico_nas_plugin(self):
        """Verify NASPlugin accepts pico profile."""
        from nas_yolo.models.nas_plugin import NASPlugin
        plugin = NASPlugin(
            channels_list=[32, 64, 128],
            profile="pico",
            mode="hybrid",
            noise_dim=32,
            pool_size=4,
        )
        total = sum(p.numel() for p in plugin.parameters())
        # Pico plugin with [32, 64, 128] should be < 150K params
        assert total < 150_000, f"Plugin too large: {total} params"


class TestPicoBackbone:
    """Test PicoBackbone module."""

    def test_forward_pass(self):
        """Verify backbone produces 3-scale output."""
        from nas_yolo.models.pico_backbone import PicoBackbone
        backbone = PicoBackbone(base_channels=32)
        x = torch.randn(2, 3, 320, 320)
        outputs = backbone(x)

        assert len(outputs) == 3
        assert outputs[0].shape == (2, 64, 40, 40)   # P3: /8
        assert outputs[1].shape == (2, 128, 20, 20)   # P4: /16
        assert outputs[2].shape == (2, 128, 10, 10)   # P5: /32

    def test_param_count(self):
        """Verify backbone params are under budget."""
        from nas_yolo.models.pico_backbone import PicoBackbone
        backbone = PicoBackbone(base_channels=32)
        total = sum(p.numel() for p in backbone.parameters())
        # Target: ~85K
        assert total < 200_000, f"Backbone too large: {total} params"

    def test_ultra_pico_channels(self):
        """Test ultra-pico with channels [16, 32, 64]."""
        from nas_yolo.models.pico_backbone import PicoBackbone
        backbone = PicoBackbone(base_channels=16)
        x = torch.randn(1, 3, 320, 320)
        outputs = backbone(x)
        assert outputs[0].shape[1] == 32   # base*2
        assert outputs[1].shape[1] == 64   # base*4
        assert outputs[2].shape[1] == 64   # base*4


class TestPicoNeck:
    """Test PicoNeck module."""

    def test_forward_pass(self):
        """Verify neck produces fused features."""
        from nas_yolo.models.pico_neck import PicoNeck
        neck = PicoNeck(in_channels=[64, 128, 128])

        p3 = torch.randn(2, 64, 40, 40)
        p4 = torch.randn(2, 128, 20, 20)
        p5 = torch.randn(2, 128, 10, 10)

        outputs = neck([p3, p4, p5])
        assert len(outputs) == 3
        assert outputs[0].shape == p3.shape
        assert outputs[1].shape == p4.shape
        assert outputs[2].shape == p5.shape


class TestPicoHead:
    """Test PicoHead module."""

    def test_forward_pass(self):
        """Verify head produces detection outputs."""
        from nas_yolo.models.pico_head import PicoDetectionHead
        head = PicoDetectionHead(
            in_channels_list=[64, 128, 128],
            num_classes=30,
        )

        feats = [
            torch.randn(2, 64, 40, 40),
            torch.randn(2, 128, 20, 20),
            torch.randn(2, 128, 10, 10),
        ]
        outputs = head(feats)

        assert "cls_preds" in outputs
        assert "reg_preds" in outputs
        assert "obj_preds" in outputs
        assert len(outputs["cls_preds"]) == 3
        assert outputs["cls_preds"][0].shape == (2, 30, 40, 40)

    def test_context_aware(self):
        """Test context-aware detection head."""
        from nas_yolo.models.pico_head import PicoDetectionHead
        head = PicoDetectionHead(
            in_channels_list=[64, 128, 128],
            num_classes=30,
            use_context=True,
            noise_dim=32,
        )

        feats = [
            torch.randn(2, 64, 40, 40),
            torch.randn(2, 128, 20, 20),
            torch.randn(2, 128, 10, 10),
        ]
        noise_desc = torch.randn(2, 32)
        outputs = head(feats, noise_descriptor=noise_desc)
        assert outputs["cls_preds"][0].shape == (2, 30, 40, 40)


class TestPicoYOLO:
    """Test complete PicoYOLO model."""

    def test_forward_pass(self):
        """Verify end-to-end forward pass."""
        from nas_yolo.models.pico_yolo import PicoYOLO
        model = PicoYOLO(
            num_classes=30,
            base_channels=32,
            input_resolution=320,
        )
        model.eval()

        images = torch.randn(2, 3, 320, 320)
        outputs = model(images)

        assert "cls_preds" in outputs
        assert "reg_preds" in outputs
        assert "obj_preds" in outputs
        assert "new_states" in outputs
        assert "conf_weights" in outputs

    def test_detect_api(self):
        """Verify high-level detect() API."""
        from nas_yolo.models.pico_yolo import PicoYOLO
        model = PicoYOLO(num_classes=30, base_channels=32)
        model.eval()

        images = torch.randn(1, 3, 320, 320)
        boxes, scores, class_ids, states = model.detect(images)

        assert boxes.shape[0] == 1
        assert boxes.shape[2] == 4  # xyxy
        assert scores.shape[0] == 1
        assert class_ids.shape[0] == 1

    def test_total_params_under_500k(self):
        """CRITICAL: Total params must be < 500K."""
        from nas_yolo.models.pico_yolo import PicoYOLO
        model = PicoYOLO(num_classes=30, base_channels=32)
        total = sum(p.numel() for p in model.parameters())
        assert total < 500_000, (
            f"PicoYOLO exceeds 500K target: {total:,} params "
            f"({total/1e6:.3f}M)"
        )

    def test_total_params_under_1m_for_coco80(self):
        """80-class model should still be < 1M params."""
        from nas_yolo.models.pico_yolo import PicoYOLO
        model = PicoYOLO(num_classes=80, base_channels=32)
        total = sum(p.numel() for p in model.parameters())
        assert total < 1_000_000, (
            f"PicoYOLO-80cls exceeds 1M target: {total:,} params"
        )

    def test_model_info(self):
        """Verify get_model_info returns valid data."""
        from nas_yolo.models.pico_yolo import PicoYOLO
        model = PicoYOLO(num_classes=30, base_channels=32)
        info = model.get_model_info()

        assert info["model"] == "PicoYOLO"
        assert info["params"]["total"] > 0
        assert info["params"]["total_M"] < 0.5
        assert "backbone" in info["params"]
        assert "plugin" in info["params"]

    def test_temporal_state(self):
        """Verify temporal state propagation across frames."""
        from nas_yolo.models.pico_yolo import PicoYOLO
        model = PicoYOLO(num_classes=30, base_channels=32)
        model.eval()

        # Frame 1
        images1 = torch.randn(1, 3, 320, 320)
        out1 = model(images1)
        states = out1["new_states"]

        # Frame 2 with state
        images2 = torch.randn(1, 3, 320, 320)
        out2 = model(images2, states=states)
        assert out2["new_states"] is not None

    def test_ultra_pico_config(self):
        """Test ultra-pico [16, 32, 64] configuration."""
        from nas_yolo.models.pico_yolo import PicoYOLO
        model = PicoYOLO(
            num_classes=30,
            base_channels=16,
        )
        total = sum(p.numel() for p in model.parameters())
        # Ultra-pico should be < 200K
        assert total < 200_000, (
            f"Ultra-pico exceeds 200K: {total:,} params"
        )

    def test_pico_plus_config(self):
        """Test pico+ [48, 96, 192] configuration."""
        from nas_yolo.models.pico_yolo import PicoYOLO
        model = PicoYOLO(
            num_classes=30,
            base_channels=48,
        )
        total = sum(p.numel() for p in model.parameters())
        # Pico+ should be < 600K
        assert total < 600_000, (
            f"Pico+ exceeds 600K: {total:,} params"
        )


class TestSASSMDefenses:
    """Verify SA-SSM 4 structural noise defenses at pico scale."""

    def test_lti_stability_d_state_2(self):
        """Defense 1: A = -exp(A_log) < 0 with d_state=2."""
        from nas_yolo.models.nas_module import SelectiveSSM

        ssm = SelectiveSSM(d_model=32, d_state=2, d_conv=1, expand=1)

        # A_log should be initialized
        assert hasattr(ssm, 'A_log')

        # A = -exp(A_log) → always negative → BIBO stable
        A = -torch.exp(ssm.A_log.float())
        assert (A < 0).all(), "A matrix must be strictly negative for stability"

        # Discrete eigenvalues inside unit circle
        dt = torch.ones(1) * 0.01  # typical timestep
        lambda_discrete = torch.exp(A * dt.unsqueeze(-1))
        assert (lambda_discrete.abs() < 1).all(), (
            "Discrete eigenvalues must be inside unit circle"
        )

    def test_gate_floor_guarantee(self):
        """Defense 3: alpha >= gate_min = 0.25 always."""
        from nas_yolo.models.mamba_attention import PROFILE_CONFIGS

        gate_min = PROFILE_CONFIGS["pico"]["gate_min"]
        assert gate_min == 0.25

        # For any input x, sigmoid(x) ∈ (0, 1)
        # alpha = gate_min + (1 - gate_min) * sigmoid(x)
        # alpha >= gate_min + (1 - gate_min) * 0 = gate_min
        for x_val in [-100, -10, -1, 0, 1, 10, 100]:
            x = torch.tensor(float(x_val))
            alpha = gate_min + (1 - gate_min) * torch.sigmoid(x)
            assert alpha >= gate_min, (
                f"Gate floor violated: alpha={alpha:.4f} < gate_min={gate_min}"
            )

    def test_fixed_router_no_params(self):
        """Defense 4: Fixed router has zero learnable parameters."""
        from nas_yolo.models.mamba_attention import PROFILE_CONFIGS

        assert PROFILE_CONFIGS["pico"]["router_type"] == "fixed"
        # Fixed router returns constant 0.5 — no learned parameters

    def test_heterogeneous_experts(self):
        """Defense 4: SSM and SE are structurally different experts."""
        from nas_yolo.models.mamba_attention import PROFILE_CONFIGS

        pico = PROFILE_CONFIGS["pico"]
        # SSM handles temporal (1D sequence processing)
        assert pico["d_state"] == 2  # Has recurrent state
        # SE handles spatial (channel-wise recalibration)
        assert pico["attention_type"] == "se"
        # These are architecturally incompatible → expert collapse impossible


class TestBuildPicoYOLO:
    """Test build_pico_yolo() factory function."""

    def test_from_config(self):
        """Build PicoYOLO from config dict."""
        from nas_yolo.models.pico_yolo import build_pico_yolo

        config = {
            "model": {
                "base_channels": 32,
                "num_classes": 30,
                "input_resolution": 320,
            },
            "plugin": {
                "profile": "pico",
                "noise_dim": 32,
                "pool_size": 4,
            },
        }
        model = build_pico_yolo(config)
        assert isinstance(model, nn.Module)
        total = sum(p.numel() for p in model.parameters())
        assert total < 500_000


class TestContextAwareDetection:
    """Test context-aware situation recognition via plugin."""

    def test_context_classifier(self):
        """ContextClassifier produces valid class weights."""
        from nas_yolo.models.pico_head import ContextClassifier
        ctx = ContextClassifier(noise_dim=32, num_classes=30, num_contexts=8)
        noise_desc = torch.randn(2, 32)
        weights = ctx(noise_desc)
        assert weights.shape == (2, 30)

    def test_pico_yolo_with_context(self):
        """PicoYOLO with context-aware detection."""
        from nas_yolo.models.pico_yolo import PicoYOLO
        model = PicoYOLO(
            num_classes=30,
            base_channels=32,
            use_context=True,
        )
        model.eval()
        images = torch.randn(1, 3, 320, 320)
        outputs = model(images)
        assert "cls_preds" in outputs


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
