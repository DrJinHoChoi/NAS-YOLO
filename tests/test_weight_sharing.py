"""Tests for Enhancement 2: Cross-Scale Weight Sharing.

Verifies that SharedNoiseRouter and SharedConfidenceRefiner correctly share
parameters across scales while maintaining per-scale adaptation.
"""

import pytest
import torch
import torch.nn as nn

from nas_yolo.models.shared_components import SharedNoiseRouter, SharedConfidenceRefiner
from nas_yolo.models.nas_plugin import NASPlugin
from nas_yolo.models.mamba_attention import PROFILE_CONFIGS


@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def channels_list():
    return [64, 128, 256]


class TestSharedNoiseRouter:
    """Test SharedNoiseRouter."""

    def test_output_shape(self, device, channels_list):
        """Router should produce correct output shapes per scale."""
        router = SharedNoiseRouter(
            channels_list=channels_list, noise_dim=32, router_type="simplified"
        ).to(device)

        for i, ch in enumerate(channels_list):
            feat = torch.randn(2, ch, 16, 16, device=device)
            noise = torch.randn(2, 32, device=device)
            alpha = router(i, feat, noise)
            assert alpha.shape == (2, ch, 1, 1), \
                f"Scale {i}: expected (2, {ch}, 1, 1), got {alpha.shape}"

    def test_shared_proj_is_same_object(self, device, channels_list):
        """shared_noise_proj should be the same module across all calls."""
        router = SharedNoiseRouter(
            channels_list=channels_list, noise_dim=32, router_type="simplified"
        ).to(device)
        # The shared_noise_proj is a single nn.Sequential, not per-scale
        assert isinstance(router.shared_noise_proj, nn.Sequential)
        # per_scale_proj should have one per scale
        assert len(router.per_scale_proj) == len(channels_list)

    def test_plugin_level_parameter_savings(self, channels_list):
        """At NASPlugin level, weight sharing should reduce total params.

        The SharedNoiseRouter replaces per-block internal routers. Overall
        plugin-level savings are the definitive comparison.
        """
        plugin_independent = NASPlugin(
            channels_list=channels_list, profile="edge", mode="hybrid",
            weight_sharing=False,
        )
        plugin_shared = NASPlugin(
            channels_list=channels_list, profile="edge", mode="hybrid",
            weight_sharing=True,
        )
        p_ind = sum(p.numel() for p in plugin_independent.parameters())
        p_shared = sum(p.numel() for p in plugin_shared.parameters())
        assert p_shared < p_ind, \
            f"Shared plugin ({p_shared}) should have fewer params than independent ({p_ind})"

    def test_gate_floor_respected(self, device, channels_list):
        """Router output should respect gate_min."""
        gate_min = 0.10
        router = SharedNoiseRouter(
            channels_list=channels_list, noise_dim=32,
            router_type="simplified", gate_min=gate_min,
        ).to(device)

        for i, ch in enumerate(channels_list):
            feat = torch.randn(4, ch, 8, 8, device=device)
            noise = torch.randn(4, 32, device=device)
            alpha = router(i, feat, noise)
            assert (alpha >= gate_min - 1e-5).all(), \
                f"Scale {i}: alpha min {alpha.min().item()} < gate_min {gate_min}"

    def test_fixed_router_returns_half(self, device, channels_list):
        """Fixed router type should return 0.5."""
        router = SharedNoiseRouter(
            channels_list=channels_list, noise_dim=32, router_type="fixed"
        ).to(device)

        feat = torch.randn(2, 64, 8, 8, device=device)
        noise = torch.randn(2, 32, device=device)
        alpha = router(0, feat, noise)
        assert torch.allclose(alpha, torch.tensor(0.5, device=device), atol=1e-5)

    def test_none_noise_returns_half(self, device, channels_list):
        """None noise_level should return 0.5."""
        router = SharedNoiseRouter(
            channels_list=channels_list, noise_dim=32, router_type="simplified"
        ).to(device)

        feat = torch.randn(2, 64, 8, 8, device=device)
        alpha = router(0, feat, noise_level=None)
        assert torch.allclose(alpha, torch.tensor(0.5, device=device), atol=1e-5)

    def test_full_router_type(self, device, channels_list):
        """Full router should use feature statistics too."""
        router = SharedNoiseRouter(
            channels_list=channels_list, noise_dim=32, router_type="full"
        ).to(device)

        assert hasattr(router, 'feat_stats_list')
        assert hasattr(router, 'gate_norms')

        for i, ch in enumerate(channels_list):
            feat = torch.randn(2, ch, 8, 8, device=device)
            noise = torch.randn(2, 32, device=device)
            alpha = router(i, feat, noise)
            assert alpha.shape == (2, ch, 1, 1)

    def test_gradient_flow(self, device, channels_list):
        """Gradients should flow through shared parameters."""
        router = SharedNoiseRouter(
            channels_list=channels_list, noise_dim=32, router_type="simplified"
        ).to(device)

        feat = torch.randn(2, 128, 8, 8, device=device)
        noise = torch.randn(2, 32, device=device, requires_grad=True)
        alpha = router(1, feat, noise)
        loss = alpha.sum()
        loss.backward()

        # Check shared params have gradients
        for p in router.shared_noise_proj.parameters():
            assert p.grad is not None


class TestSharedConfidenceRefiner:
    """Test SharedConfidenceRefiner."""

    def test_output_shape(self, device, channels_list):
        """Should produce [B, 1] confidence per scale."""
        refiner = SharedConfidenceRefiner(
            channels_list=channels_list, common_dim=32
        ).to(device)

        for i, ch in enumerate(channels_list):
            feat = torch.randn(2, ch, 16, 16, device=device)
            conf = refiner(i, feat)
            assert conf.shape == (2, 1), \
                f"Scale {i}: expected (2, 1), got {conf.shape}"

    def test_output_range(self, device, channels_list):
        """Confidence should be in [0, 1] (sigmoid output)."""
        refiner = SharedConfidenceRefiner(
            channels_list=channels_list, common_dim=32
        ).to(device)

        for i, ch in enumerate(channels_list):
            feat = torch.randn(4, ch, 8, 8, device=device)
            conf = refiner(i, feat)
            assert (conf >= 0.0).all()
            assert (conf <= 1.0).all()

    def test_shared_core_is_single(self, channels_list):
        """shared_core should be one module, not per-scale."""
        refiner = SharedConfidenceRefiner(
            channels_list=channels_list, common_dim=32
        )
        assert isinstance(refiner.shared_core, nn.Sequential)
        assert len(refiner.input_projs) == len(channels_list)

    def test_parameter_savings(self, channels_list):
        """Shared refiner should save parameters vs independent refiners."""
        # Simulate independent conf_refiners (from NASPlugin._build_independent)
        independent_params = 0
        for ch in channels_list:
            mid = max(ch // 4, 8)
            refiner = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(ch, mid),
                nn.ReLU(inplace=True),
                nn.Linear(mid, 1),
                nn.Sigmoid(),
            )
            independent_params += sum(p.numel() for p in refiner.parameters())

        # Shared
        shared = SharedConfidenceRefiner(
            channels_list=channels_list, common_dim=32
        )
        shared_params = sum(p.numel() for p in shared.parameters())

        assert shared_params < independent_params, \
            f"Shared ({shared_params}) should have fewer params than independent ({independent_params})"

    def test_gradient_flow(self, device, channels_list):
        """Gradients should flow through shared core."""
        refiner = SharedConfidenceRefiner(
            channels_list=channels_list, common_dim=32
        ).to(device)

        feat = torch.randn(2, 128, 8, 8, device=device, requires_grad=True)
        conf = refiner(1, feat)
        loss = conf.sum()
        loss.backward()

        # Shared core should have gradients
        for p in refiner.shared_core.parameters():
            assert p.grad is not None


class TestNASPluginWeightSharing:
    """Test NASPlugin with weight_sharing enabled/disabled."""

    def test_weight_sharing_false_backward_compat(self, device):
        """weight_sharing=False should work exactly like before."""
        plugin = NASPlugin(
            channels_list=[64, 128, 256],
            profile="standard",
            mode="hybrid",
            weight_sharing=False,
        ).to(device)

        assert plugin.conf_refiners is not None
        assert not hasattr(plugin, 'shared_router') or not hasattr(plugin, 'shared_conf_refiner')

        features = [
            torch.randn(2, 64, 40, 40, device=device),
            torch.randn(2, 128, 20, 20, device=device),
            torch.randn(2, 256, 10, 10, device=device),
        ]
        images = torch.randn(2, 3, 320, 320, device=device)
        refined, states, confs = plugin(features, images=images)

        assert len(refined) == 3
        for i, (r, f) in enumerate(zip(refined, features)):
            assert r.shape == f.shape, f"Scale {i}: shape mismatch"

    def test_weight_sharing_true_creates_shared(self, device):
        """weight_sharing=True should create shared components."""
        plugin = NASPlugin(
            channels_list=[64, 128, 256],
            profile="edge",
            mode="hybrid",
            weight_sharing=True,
        ).to(device)

        assert hasattr(plugin, 'shared_router')
        assert hasattr(plugin, 'shared_conf_refiner')
        assert isinstance(plugin.shared_router, SharedNoiseRouter)
        assert isinstance(plugin.shared_conf_refiner, SharedConfidenceRefiner)

    def test_weight_sharing_forward(self, device):
        """Forward pass should work with weight sharing enabled."""
        plugin = NASPlugin(
            channels_list=[64, 128, 256],
            profile="edge",
            mode="hybrid",
            weight_sharing=True,
        ).to(device)

        features = [
            torch.randn(2, 64, 40, 40, device=device),
            torch.randn(2, 128, 20, 20, device=device),
            torch.randn(2, 256, 10, 10, device=device),
        ]
        images = torch.randn(2, 3, 320, 320, device=device)
        refined, states, confs = plugin(features, images=images)

        assert len(refined) == 3
        for i, (r, f) in enumerate(zip(refined, features)):
            assert r.shape == f.shape, f"Scale {i}: shape mismatch"
        assert len(confs) == 3

    def test_weight_sharing_reduces_params(self, device):
        """Weight sharing should reduce total parameter count."""
        plugin_no_share = NASPlugin(
            channels_list=[64, 128, 256],
            profile="edge",
            mode="hybrid",
            weight_sharing=False,
        ).to(device)

        plugin_shared = NASPlugin(
            channels_list=[64, 128, 256],
            profile="edge",
            mode="hybrid",
            weight_sharing=True,
        ).to(device)

        params_no_share = sum(p.numel() for p in plugin_no_share.parameters())
        params_shared = sum(p.numel() for p in plugin_shared.parameters())

        assert params_shared < params_no_share, \
            f"Shared ({params_shared}) should have fewer params than no-share ({params_no_share})"

    def test_plugin_info_reports_sharing(self, device):
        """get_plugin_info should report weight sharing status."""
        plugin = NASPlugin(
            channels_list=[64, 128, 256],
            profile="edge",
            mode="hybrid",
            weight_sharing=True,
        ).to(device)

        info = plugin.get_plugin_info()
        assert info["weight_sharing"] is True
        assert "shared_component_params" in info
        assert info["shared_component_params"] > 0

    def test_profile_default_weight_sharing(self):
        """Edge and ultra-lite should default to weight_sharing=True."""
        assert PROFILE_CONFIGS["edge"]["weight_sharing"] is True
        assert PROFILE_CONFIGS["ultra-lite"]["weight_sharing"] is True
        assert PROFILE_CONFIGS["standard"]["weight_sharing"] is False
        assert PROFILE_CONFIGS["lite"]["weight_sharing"] is False

    def test_profile_default_applied(self, device):
        """When weight_sharing is not explicitly set, profile default should apply."""
        # Edge profile defaults to weight_sharing=True
        plugin_edge = NASPlugin(
            channels_list=[64, 128, 256],
            profile="edge",
            mode="hybrid",
            # weight_sharing not specified → should use profile default
        ).to(device)
        assert plugin_edge.weight_sharing is True

        # Standard profile defaults to weight_sharing=False
        plugin_std = NASPlugin(
            channels_list=[64, 128, 256],
            profile="standard",
            mode="hybrid",
        ).to(device)
        assert plugin_std.weight_sharing is False

    def test_explicit_override_profile_default(self, device):
        """Explicit weight_sharing should override profile default."""
        # Edge defaults to True, but explicitly set False
        plugin = NASPlugin(
            channels_list=[64, 128, 256],
            profile="edge",
            mode="hybrid",
            weight_sharing=False,
        ).to(device)
        assert plugin.weight_sharing is False
        assert plugin.conf_refiners is not None
