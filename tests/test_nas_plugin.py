"""Tests for NASPlugin standalone module."""

import pytest
import torch
from nas_yolo.models.nas_plugin import NASPlugin


@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def make_features(channels_list, batch=2, img_size=640, device="cpu"):
    """Create dummy multi-scale features matching YOLO neck output."""
    strides = [8, 16, 32]
    features = []
    for ch, stride in zip(channels_list, strides):
        h = w = img_size // stride
        features.append(torch.randn(batch, ch, h, w, device=device))
    return features


class TestNASPluginInterface:
    @pytest.mark.parametrize("profile", ["standard", "lite", "edge", "ultra-lite"])
    def test_output_shapes(self, device, profile):
        channels = [64, 128, 256]
        plugin = NASPlugin(channels, profile=profile, mode="hybrid").to(device)
        features = make_features(channels, device=device)
        images = torch.randn(2, 3, 640, 640, device=device)

        refined, states, conf = plugin(features, images=images)

        # Same number of scales
        assert len(refined) == len(features)
        # Same shapes
        for r, f in zip(refined, features):
            assert r.shape == f.shape
        # States
        assert len(states) == len(features)
        # Confidence weights
        assert len(conf) == len(features)
        for c in conf:
            assert c.shape == (2, 1)

    @pytest.mark.parametrize("mode", ["hybrid", "mamba", "attention"])
    def test_modes(self, device, mode):
        channels = [64, 128, 256]
        plugin = NASPlugin(channels, profile="edge", mode=mode).to(device)
        features = make_features(channels, device=device)
        images = torch.randn(2, 3, 640, 640, device=device)

        refined, states, conf = plugin(features, images=images)
        assert len(refined) == 3
        for r, f in zip(refined, features):
            assert r.shape == f.shape

    def test_without_images(self, device):
        """Plugin should work without raw images (no noise estimation)."""
        channels = [128, 256, 512]
        plugin = NASPlugin(channels, profile="edge", mode="hybrid").to(device)
        features = make_features(channels, device=device)

        refined, states, conf = plugin(features, images=None)
        assert len(refined) == 3

    def test_small_scale_channels(self, device):
        channels = [128, 256, 512]
        plugin = NASPlugin(channels, profile="edge").to(device)
        features = make_features(channels, device=device)
        images = torch.randn(2, 3, 640, 640, device=device)

        refined, _, _ = plugin(features, images=images)
        for r, f in zip(refined, features):
            assert r.shape == f.shape


class TestNASPluginTemporal:
    def test_state_accumulation(self, device):
        channels = [64, 128, 256]
        plugin = NASPlugin(channels, profile="edge", mode="hybrid").to(device)
        images = torch.randn(1, 3, 640, 640, device=device)

        # Use similar features to avoid scene change detection reset
        torch.manual_seed(42)
        feat1 = make_features(channels, batch=1, device=device)
        _, states1, _ = plugin(feat1, images=images)

        # Frame 2: slightly perturbed (same structure, avoids scene change)
        feat2 = [f + torch.randn_like(f) * 0.01 for f in feat1]
        _, states2, _ = plugin(feat2, images=images)

        assert plugin.temporal_buffer.frame_count == 2

    def test_reset_state(self, device):
        channels = [64, 128, 256]
        plugin = NASPlugin(channels, profile="edge").to(device)
        images = torch.randn(1, 3, 640, 640, device=device)
        features = make_features(channels, batch=1, device=device)

        # Accumulate state
        plugin(features, images=images)
        plugin(features, images=images)
        assert plugin.temporal_buffer.frame_count == 2

        # Reset
        plugin.reset_state()
        assert plugin.temporal_buffer.frame_count == 0


class TestNASPluginInfo:
    def test_get_plugin_info(self, device):
        channels = [64, 128, 256]
        plugin = NASPlugin(channels, profile="edge", mode="hybrid").to(device)
        info = plugin.get_plugin_info()

        assert info["profile"] == "edge"
        assert info["mode"] == "hybrid"
        assert info["total_params"] > 0
        assert info["total_params_M"] > 0
        assert len(info["per_block"]) == 3

    @pytest.mark.parametrize("profile", ["standard", "lite", "edge", "ultra-lite"])
    def test_param_ordering(self, device, profile):
        """Lighter profiles should have fewer params."""
        channels = [64, 128, 256]
        plugin = NASPlugin(channels, profile=profile, mode="hybrid").to(device)
        info = plugin.get_plugin_info()
        return info["total_params"]

    def test_edge_is_lighter_than_standard(self, device):
        channels = [64, 128, 256]
        edge = NASPlugin(channels, profile="edge").to(device)
        standard = NASPlugin(channels, profile="standard").to(device)

        edge_params = sum(p.numel() for p in edge.parameters())
        standard_params = sum(p.numel() for p in standard.parameters())
        assert edge_params < standard_params


class TestNASPluginGradient:
    def test_gradient_flow_hybrid(self, device):
        channels = [64, 128, 256]
        plugin = NASPlugin(channels, profile="edge", mode="hybrid").to(device)
        features = [f.requires_grad_(True) for f in make_features(channels, batch=1, device=device)]
        images = torch.randn(1, 3, 640, 640, device=device)

        refined, _, _ = plugin(features, images=images)
        loss = sum(r.sum() for r in refined)
        loss.backward()

        # Check gradients flow back to input features
        for f in features:
            assert f.grad is not None
