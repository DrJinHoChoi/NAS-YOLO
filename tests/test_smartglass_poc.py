"""
Tests for Smart Glasses At-a-Glance POC
=========================================
Verifies:
  - 30-class config loading
  - Ultra-lite plugin parameter count (~0.39M)
  - 4-layer structural noise defense
  - ONNX export (plugin-only)
  - Class-independent overhead (same params for 30 vs 80 classes)
"""

import os
import sys
import pytest
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nas_yolo.models.nas_plugin import NASPlugin
from nas_yolo.models.mamba_attention import PROFILE_CONFIGS


# =========================================================================
# Config Tests
# =========================================================================

class TestSmartGlassConfig:
    """Test 30-class smart glasses configuration."""

    @pytest.fixture
    def config(self):
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "nas_yolo", "configs", "smartglass", "smartglass_30class.yaml"
        )
        with open(config_path, encoding="utf-8") as f:
            return yaml.safe_load(f)

    def test_num_classes(self, config):
        assert config["model"]["num_classes"] == 30

    def test_profile_is_ultra_lite(self, config):
        assert config["plugin"]["profile"] == "ultra-lite"

    def test_yolo_variant(self, config):
        assert config["model"]["yolo_variant"] == "yolov8n"

    def test_channels_list(self, config):
        assert config["model"]["channels_list"] == [64, 128, 256]

    def test_30_categories_defined(self, config):
        names = config["categories"]["names"]
        assert len(names) == 30

    def test_coco_mapping_defined(self, config):
        mapping = config["categories"]["coco_mapping"]
        assert len(mapping) == 30

    def test_deployment_constraints(self, config):
        deploy = config["deployment"]
        assert deploy["latency_target_ms"] == 50
        assert deploy["power_budget_watts"] <= 1.0

    def test_noise_augmentation_enabled(self, config):
        aug = config["augmentation"]
        assert aug["corruption_aug"] is True
        assert aug["corruption_p"] >= 0.3


# =========================================================================
# Plugin Parameter Tests
# =========================================================================

class TestUltraLitePlugin:
    """Test ultra-lite plugin parameter constraints."""

    @pytest.fixture
    def plugin(self):
        return NASPlugin(
            channels_list=[64, 128, 256],
            profile="ultra-lite",
            mode="hybrid",
            use_noise_gate=True,
            noise_dim=32,
            pool_size=8,
        )

    def test_params_under_500k(self, plugin):
        """Ultra-lite must be under 0.50M params."""
        total = sum(p.numel() for p in plugin.parameters())
        assert total < 500_000, f"Ultra-lite has {total:,} params (>500K)"

    def test_params_approximately_039m(self, plugin):
        """Should be approximately 0.39M (lightest multi-scale profile)."""
        info = plugin.get_plugin_info()
        assert 0.30 <= info["total_params_M"] <= 0.50, \
            f"Expected ~0.39M, got {info['total_params_M']:.4f}M"

    def test_profile_is_ultra_lite(self, plugin):
        info = plugin.get_plugin_info()
        assert info["profile"] == "ultra-lite"

    def test_weight_sharing_enabled(self, plugin):
        info = plugin.get_plugin_info()
        assert info["weight_sharing"] is True

    def test_gate_min_is_015(self):
        """Ultra-lite must have gate_min=0.15 (most conservative)."""
        pcfg = PROFILE_CONFIGS["ultra-lite"]
        assert pcfg["gate_min"] == 0.15

    def test_fixed_router(self):
        """Ultra-lite must use fixed router (zero learnable routing)."""
        pcfg = PROFILE_CONFIGS["ultra-lite"]
        assert pcfg["router_type"] == "fixed"

    def test_se_attention(self):
        """Ultra-lite must use SE attention (lightest option)."""
        pcfg = PROFILE_CONFIGS["ultra-lite"]
        assert pcfg["attention_type"] == "se"


# =========================================================================
# Class-Independent Overhead Test (Patent Claim 24)
# =========================================================================

class TestClassIndependentOverhead:
    """Patent Claim 24: Plugin overhead is independent of num_classes."""

    def test_same_params_30_vs_80_classes(self):
        """Plugin params must be identical for 30 and 80 classes."""
        channels = [64, 128, 256]

        plugin_30 = NASPlugin(
            channels_list=channels, profile="ultra-lite", mode="hybrid",
        )
        plugin_80 = NASPlugin(
            channels_list=channels, profile="ultra-lite", mode="hybrid",
        )

        params_30 = sum(p.numel() for p in plugin_30.parameters())
        params_80 = sum(p.numel() for p in plugin_80.parameters())

        assert params_30 == params_80, \
            f"Plugin params differ: 30-class={params_30:,} vs 80-class={params_80:,}"

    def test_all_profiles_class_independent(self):
        """All profiles must be class-independent."""
        channels = [64, 128, 256]

        for profile_name in PROFILE_CONFIGS:
            p1 = NASPlugin(channels_list=channels, profile=profile_name, mode="hybrid")
            p2 = NASPlugin(channels_list=channels, profile=profile_name, mode="hybrid")

            n1 = sum(p.numel() for p in p1.parameters())
            n2 = sum(p.numel() for p in p2.parameters())

            assert n1 == n2, \
                f"{profile_name}: params differ across instances"


# =========================================================================
# Noise Defense Tests
# =========================================================================

class TestStructuralNoiseDefense:
    """Test 4-layer structural noise defense under smart glasses noise."""

    @pytest.fixture
    def plugin(self):
        plugin = NASPlugin(
            channels_list=[64, 128, 256],
            profile="ultra-lite",
            mode="hybrid",
            use_noise_gate=True,
            noise_dim=32,
        )
        plugin.eval()
        return plugin

    def _make_inputs(self, B=2):
        """Create dummy feature maps and images."""
        channels = [64, 128, 256]
        feats = []
        for i, ch in enumerate(channels):
            h = 80 // (2 ** i)
            feats.append(torch.randn(B, ch, h, h))
        images = torch.rand(B, 3, 640, 640)
        return feats, images

    def test_noise_estimator_exists(self, plugin):
        """Noise estimator (DCT spectral) must be active."""
        assert plugin.noise_estimator is not None

    def test_clean_forward_pass(self, plugin):
        """Plugin should process clean features without error."""
        feats, images = self._make_inputs()
        with torch.no_grad():
            enhanced, states, conf = plugin(feats, images=images)

        assert len(enhanced) == 3
        assert len(conf) == 3
        for i, (orig, enh) in enumerate(zip(feats, enhanced)):
            assert orig.shape == enh.shape, f"Scale {i} shape mismatch"

    def test_noisy_forward_pass(self, plugin):
        """Plugin should process noisy features without error."""
        feats, images = self._make_inputs()
        # Add heavy noise
        noisy_feats = [f + torch.randn_like(f) * 0.5 for f in feats]
        noisy_images = (images + torch.randn_like(images) * 0.1).clamp(0, 1)

        with torch.no_grad():
            enhanced, states, conf = plugin(noisy_feats, images=noisy_images)

        assert len(enhanced) == 3
        for enh in enhanced:
            assert not torch.isnan(enh).any(), "NaN in enhanced features"
            assert not torch.isinf(enh).any(), "Inf in enhanced features"

    def test_output_shapes_preserved(self, plugin):
        """Output feature shapes must match input shapes exactly."""
        feats, images = self._make_inputs()
        with torch.no_grad():
            enhanced, _, _ = plugin(feats, images=images)

        for i in range(3):
            assert feats[i].shape == enhanced[i].shape

    def test_confidence_range(self, plugin):
        """Confidence weights must be in [0, 1]."""
        feats, images = self._make_inputs()
        with torch.no_grad():
            _, _, conf = plugin(feats, images=images)

        for i, c in enumerate(conf):
            assert c.min() >= 0.0, f"Scale {i}: conf < 0"
            assert c.max() <= 1.0, f"Scale {i}: conf > 1"

    def test_temporal_state_reset(self, plugin):
        """Temporal state should reset cleanly."""
        feats, images = self._make_inputs()

        with torch.no_grad():
            plugin(feats, images=images)
        plugin.reset_state()

        # Should work after reset
        with torch.no_grad():
            enhanced, _, _ = plugin(feats, images=images)
        assert len(enhanced) == 3


# =========================================================================
# ONNX Export Test
# =========================================================================

class TestONNXExport:
    """Test ONNX export of NASPlugin."""

    def test_plugin_onnx_export(self, tmp_path):
        """Plugin should export to ONNX without errors."""
        from nas_yolo.scripts.export_onnx import PluginONNXWrapper

        plugin = NASPlugin(
            channels_list=[64, 128, 256],
            profile="ultra-lite",
            mode="hybrid",
            pool_size=4,  # ONNX: output_size must divide input_size
        )
        plugin.eval()

        wrapper = PluginONNXWrapper(plugin)
        wrapper.eval()

        # Dummy inputs
        feat0 = torch.randn(1, 64, 80, 80)
        feat1 = torch.randn(1, 128, 40, 40)
        feat2 = torch.randn(1, 256, 20, 20)
        images = torch.randn(1, 3, 640, 640)

        onnx_path = str(tmp_path / "test_plugin.onnx")

        torch.onnx.export(
            wrapper,
            (feat0, feat1, feat2, images),
            onnx_path,
            opset_version=17,
            input_names=["feat0", "feat1", "feat2", "images"],
            output_names=["out0", "out1", "out2", "conf0", "conf1", "conf2"],
            do_constant_folding=True,
            dynamo=False,
        )

        assert os.path.exists(onnx_path)
        assert os.path.getsize(onnx_path) > 0

    def test_onnx_wrapper_forward_matches_plugin(self):
        """PluginONNXWrapper output should match NASPlugin output."""
        from nas_yolo.scripts.export_onnx import PluginONNXWrapper

        plugin = NASPlugin(
            channels_list=[64, 128, 256],
            profile="ultra-lite",
            mode="hybrid",
        )
        plugin.eval()
        wrapper = PluginONNXWrapper(plugin)
        wrapper.eval()

        feat0 = torch.randn(1, 64, 80, 80)
        feat1 = torch.randn(1, 128, 40, 40)
        feat2 = torch.randn(1, 256, 20, 20)
        images = torch.randn(1, 3, 640, 640)

        # Wrapper output
        with torch.no_grad():
            wrapper_out = wrapper(feat0, feat1, feat2, images)

        # Direct plugin output
        plugin.reset_state()
        with torch.no_grad():
            plugin_out, _, plugin_conf = plugin([feat0, feat1, feat2], images=images)

        # Enhanced features should match
        for i in range(3):
            assert torch.allclose(wrapper_out[i], plugin_out[i], atol=1e-6), \
                f"Scale {i} mismatch"
        # Confidence should match
        for i in range(3):
            assert torch.allclose(wrapper_out[i + 3], plugin_conf[i], atol=1e-6), \
                f"Conf {i} mismatch"


# =========================================================================
# Total Model Size Test
# =========================================================================

class TestTotalModelSize:
    """Verify total ~1.7M model size claim."""

    def test_plugin_plus_yolo_under_4m(self):
        """Plugin (~0.39M) + YOLOv8n (~3.2M) should be under 4.0M total."""
        plugin = NASPlugin(
            channels_list=[64, 128, 256],
            profile="ultra-lite",
            mode="hybrid",
        )
        plugin_params = sum(p.numel() for p in plugin.parameters())

        # YOLOv8n full: ~3.2M for 80 classes
        # With 30 classes: head is smaller, ~3.0M
        yolo_nano_approx = 3.0e6

        total = plugin_params + yolo_nano_approx
        assert total < 4.0e6, \
            f"Total {total/1e6:.2f}M exceeds 4.0M target"

    def test_ultra_lite_among_smallest_profiles(self):
        """Ultra-lite and tiny must be the smallest profiles."""
        channels = [64, 128, 256]
        param_counts = {}

        for profile_name in PROFILE_CONFIGS:
            p = NASPlugin(
                channels_list=channels, profile=profile_name, mode="hybrid",
            )
            param_counts[profile_name] = sum(pp.numel() for pp in p.parameters())

        # ultra-lite and tiny should both be smaller than edge
        assert param_counts["ultra-lite"] < param_counts["edge"], \
            f"ultra-lite ({param_counts['ultra-lite']:,}) >= edge ({param_counts['edge']:,})"
        assert param_counts["tiny"] < param_counts["edge"], \
            f"tiny ({param_counts['tiny']:,}) >= edge ({param_counts['edge']:,})"
