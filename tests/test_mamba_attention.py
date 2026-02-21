"""Tests for Hybrid Mamba-Attention Block."""

import pytest
import torch
from nas_yolo.models.mamba_attention import (
    WindowMHSA,
    DWConvAttention,
    SEAttention,
    NoiseConditionedRouter,
    MambaAttentionBlock,
    PROFILE_CONFIGS,
    build_attention,
    window_partition,
    window_merge,
)


@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


class TestWindowPartitionMerge:
    def test_roundtrip(self, device):
        x = torch.randn(2, 64, 20, 20, device=device)
        windows, H, W = window_partition(x, window_size=4)
        # 20/4 = 5 windows per dim -> 25 windows * 2 batch = 50
        assert windows.shape == (50, 64, 4, 4)
        reconstructed = window_merge(windows, 4, H, W, B=2)
        assert reconstructed.shape == x.shape
        assert torch.allclose(reconstructed, x, atol=1e-6)

    def test_padding(self, device):
        """Feature map not divisible by window_size should be padded."""
        x = torch.randn(1, 32, 13, 13, device=device)
        windows, H, W = window_partition(x, window_size=4)
        assert H == 13
        assert W == 13
        # Padded to 16x16 -> 4x4=16 windows
        assert windows.shape[0] == 16
        reconstructed = window_merge(windows, 4, H, W, B=1)
        assert reconstructed.shape == (1, 32, 13, 13)


class TestAttentionVariants:
    @pytest.mark.parametrize("dim", [64, 128, 256])
    def test_window_mhsa_shape(self, device, dim):
        attn = WindowMHSA(dim, num_heads=4, window_size=4).to(device)
        x = torch.randn(2, dim, 20, 20, device=device)
        out = attn(x)
        assert out.shape == x.shape

    @pytest.mark.parametrize("dim", [64, 128, 256])
    def test_dwconv_attention_shape(self, device, dim):
        attn = DWConvAttention(dim).to(device)
        x = torch.randn(2, dim, 20, 20, device=device)
        out = attn(x)
        assert out.shape == x.shape

    @pytest.mark.parametrize("dim", [64, 128, 256])
    def test_se_attention_shape(self, device, dim):
        attn = SEAttention(dim).to(device)
        x = torch.randn(2, dim, 20, 20, device=device)
        out = attn(x)
        assert out.shape == x.shape

    def test_build_attention_factory(self, device):
        for atype in ["mhsa", "dwconv", "se"]:
            attn = build_attention(atype, 64, num_heads=2, window_size=4).to(device)
            x = torch.randn(1, 64, 16, 16, device=device)
            out = attn(x)
            assert out.shape == x.shape


class TestNoiseConditionedRouter:
    @pytest.mark.parametrize("router_type", ["full", "simplified", "fixed"])
    def test_output_shape(self, device, router_type):
        router = NoiseConditionedRouter(128, noise_dim=32, router_type=router_type).to(device)
        feat = torch.randn(2, 128, 20, 20, device=device)
        noise = torch.randn(2, 32, device=device)
        alpha = router(feat, noise)
        assert alpha.shape[0] == 2
        assert alpha.dim() == 4  # [B, C or 1, 1, 1]

    def test_fixed_router_no_params(self, device):
        router = NoiseConditionedRouter(64, router_type="fixed").to(device)
        params = sum(p.numel() for p in router.parameters())
        assert params == 0

    def test_no_noise_returns_balanced(self, device):
        router = NoiseConditionedRouter(64, router_type="simplified").to(device)
        feat = torch.randn(2, 64, 10, 10, device=device)
        alpha = router(feat, noise_level=None)
        assert torch.allclose(alpha, torch.tensor(0.5, device=device))


class TestMambaAttentionBlock:
    @pytest.mark.parametrize("mode", ["hybrid", "mamba", "attention"])
    def test_mode_shapes(self, device, mode):
        block = MambaAttentionBlock(
            channels=64, d_state=8, expand=1,
            attention_type="dwconv", mode=mode,
        ).to(device)
        x = torch.randn(2, 64, 20, 20, device=device)
        noise = torch.randn(2, 32, device=device) if mode == "hybrid" else None
        out, state = block(x, state=None, noise_level=noise)
        assert out.shape == x.shape

    @pytest.mark.parametrize("profile", list(PROFILE_CONFIGS.keys()))
    def test_all_profiles(self, device, profile):
        pcfg = PROFILE_CONFIGS[profile]
        block = MambaAttentionBlock(
            channels=64, d_state=pcfg["d_state"], expand=pcfg["expand"],
            attention_type=pcfg["attention_type"],
            num_heads=pcfg["num_heads"], window_size=pcfg["window_size"],
            router_type=pcfg["router_type"], mode="hybrid",
        ).to(device)
        x = torch.randn(1, 64, 16, 16, device=device)
        noise = torch.randn(1, 32, device=device)
        out, state = block(x, noise_level=noise)
        assert out.shape == x.shape

    def test_gradient_flow(self, device):
        block = MambaAttentionBlock(
            channels=64, d_state=8, expand=1,
            attention_type="dwconv", mode="hybrid",
        ).to(device)
        x = torch.randn(1, 64, 8, 8, device=device, requires_grad=True)
        noise = torch.randn(1, 32, device=device)
        out, _ = block(x, noise_level=noise)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_temporal_state_propagation(self, device):
        block = MambaAttentionBlock(
            channels=64, d_state=8, expand=1,
            attention_type="dwconv", mode="mamba",
        ).to(device)
        x1 = torch.randn(1, 64, 8, 8, device=device)
        x2 = torch.randn(1, 64, 8, 8, device=device)

        out1, state1 = block(x1, state=None)
        out2, state2 = block(x2, state=state1)

        # Outputs should differ when state is provided
        out2_no_state, _ = block(x2, state=None)
        assert not torch.allclose(out2, out2_no_state, atol=1e-4)

    def test_count_params(self, device):
        block = MambaAttentionBlock(
            channels=64, d_state=8, expand=1,
            attention_type="dwconv", mode="hybrid",
        ).to(device)
        info = block.count_params()
        assert "mamba" in info
        assert "attention" in info
        assert "router" in info
        assert info["total"] > 0
