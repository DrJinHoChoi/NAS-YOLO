"""
Unit Tests for NAS Module (Core Contribution C1)
==================================================
Tests the Selective SSM, SpatialSSMAdapter, and NoiseAwareStateModule
for correct shapes, gradient flow, and temporal behavior.
"""

import torch
import pytest
from nas_yolo.models.nas_module import (
    SelectiveSSM,
    SpatialSSMAdapter,
    NoiseAwareStateModule,
)


class TestSelectiveSSM:
    """Tests for the Selective State Space Model."""

    def test_output_shape(self):
        """Output shape should match input shape."""
        ssm = SelectiveSSM(d_model=64, d_state=16)
        x = torch.randn(2, 100, 64)  # [B, L, D]
        y, state = ssm(x)
        assert y.shape == x.shape, f"Expected {x.shape}, got {y.shape}"

    def test_state_shape(self):
        """State shape should be [B, d_inner, d_state]."""
        ssm = SelectiveSSM(d_model=64, d_state=16, expand=2)
        x = torch.randn(2, 100, 64)
        _, state = ssm(x)
        assert state.shape == (2, 128, 16)  # d_inner = 64 * 2 = 128

    def test_state_propagation(self):
        """Passing state from one call to next should produce different outputs."""
        ssm = SelectiveSSM(d_model=32, d_state=8)
        x = torch.randn(1, 50, 32)

        # First call (no state)
        y1, state1 = ssm(x, state=None)

        # Second call (with state)
        y2, state2 = ssm(x, state=state1)

        # Outputs should differ because of state
        assert not torch.allclose(y1, y2, atol=1e-6), \
            "State propagation should change output"

    def test_gradient_flow(self):
        """Gradients should flow through the SSM."""
        ssm = SelectiveSSM(d_model=32, d_state=8)
        x = torch.randn(1, 20, 32, requires_grad=True)
        y, _ = ssm(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None, "Gradient should flow to input"
        assert x.grad.abs().sum() > 0, "Gradient should be non-zero"

    def test_different_sequence_lengths(self):
        """SSM should handle variable sequence lengths."""
        ssm = SelectiveSSM(d_model=32, d_state=8)
        for L in [1, 10, 50, 200]:
            x = torch.randn(1, L, 32)
            y, _ = ssm(x)
            assert y.shape == (1, L, 32)


class TestSpatialSSMAdapter:
    """Tests for the 2D spatial adapter."""

    def test_output_shape(self):
        """Output should match input spatial dimensions."""
        adapter = SpatialSSMAdapter(channels=64, d_state=16)
        x = torch.randn(2, 64, 20, 20)  # [B, C, H, W]
        out, state = adapter(x)
        assert out.shape == x.shape

    def test_different_spatial_sizes(self):
        """Should handle various spatial dimensions."""
        adapter = SpatialSSMAdapter(channels=32, d_state=8)
        for h, w in [(8, 8), (16, 16), (32, 32), (20, 40)]:
            x = torch.randn(1, 32, h, w)
            out, _ = adapter(x)
            assert out.shape == (1, 32, h, w)


class TestNoiseAwareStateModule:
    """Tests for the full NAS Module."""

    def test_multi_scale_output(self):
        """Should produce outputs for each scale."""
        channels = [64, 128, 256]
        nas = NoiseAwareStateModule(channels, use_noise_gate=False)

        features = [
            torch.randn(2, 64, 80, 80),
            torch.randn(2, 128, 40, 40),
            torch.randn(2, 256, 20, 20),
        ]

        refined, states, conf_weights = nas(features)

        assert len(refined) == 3
        assert len(states) == 3
        assert len(conf_weights) == 3

        for i, (ref, feat) in enumerate(zip(refined, features)):
            assert ref.shape == feat.shape, f"Scale {i}: shape mismatch"

    def test_confidence_weight_range(self):
        """Confidence weights should be in [0, 1] (sigmoid output)."""
        channels = [64, 128]
        nas = NoiseAwareStateModule(channels, use_noise_gate=False)

        features = [
            torch.randn(2, 64, 20, 20),
            torch.randn(2, 128, 10, 10),
        ]

        _, _, conf_weights = nas(features)

        for cw in conf_weights:
            assert (cw >= 0).all() and (cw <= 1).all(), \
                "Confidence weights should be in [0, 1]"

    def test_with_noise_gate(self):
        """Should work with noise gating enabled."""
        channels = [64, 128]
        nas = NoiseAwareStateModule(channels, use_noise_gate=True)

        features = [
            torch.randn(2, 64, 20, 20),
            torch.randn(2, 128, 10, 10),
        ]
        noise_level = torch.randn(2, 32)  # noise descriptor

        refined, states, conf_weights = nas(features, noise_level=noise_level)

        assert len(refined) == 2
        for ref, feat in zip(refined, features):
            assert ref.shape == feat.shape

    def test_temporal_state_continuity(self):
        """State from frame t should influence frame t+1."""
        channels = [64]
        nas = NoiseAwareStateModule(channels, use_noise_gate=False)

        feat1 = [torch.randn(1, 64, 10, 10)]
        feat2 = [torch.randn(1, 64, 10, 10)]

        # Frame 1: no state
        ref1, states1, _ = nas(feat1, states=None)

        # Frame 2: with state from frame 1
        ref2_with_state, _, _ = nas(feat2, states=states1)

        # Frame 2: without state
        ref2_no_state, _, _ = nas(feat2, states=None)

        # With state should produce different output
        assert not torch.allclose(ref2_with_state[0], ref2_no_state[0], atol=1e-5), \
            "Temporal state should influence output"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
