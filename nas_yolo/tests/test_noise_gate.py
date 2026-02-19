"""
Unit Tests for Noise-Aware Gating (Core Contribution C2)
==========================================================
Tests the SpectralNoiseEstimator and NoiseAwareGate modules.
"""

import torch
import pytest
from nas_yolo.models.noise_gate import (
    SpectralNoiseEstimator,
    NoiseAwareGate,
    MultiScaleNoiseEstimator,
)


class TestSpectralNoiseEstimator:
    """Tests for the noise estimation module."""

    def test_output_shape(self):
        """Output should be [B, noise_dim]."""
        estimator = SpectralNoiseEstimator(in_channels=3, noise_dim=32)
        image = torch.randn(2, 3, 640, 640)
        noise_desc = estimator(image)
        assert noise_desc.shape == (2, 32)

    def test_different_input_sizes(self):
        """Should handle different image resolutions."""
        estimator = SpectralNoiseEstimator(in_channels=3, noise_dim=16)
        for size in [224, 416, 640, 1280]:
            image = torch.randn(1, 3, size, size)
            desc = estimator(image)
            assert desc.shape == (1, 16)

    def test_noise_sensitivity(self):
        """Higher noise should produce different descriptors."""
        estimator = SpectralNoiseEstimator(in_channels=3, noise_dim=32)
        estimator.eval()

        clean = torch.rand(1, 3, 256, 256)
        noisy = clean + torch.randn_like(clean) * 0.3

        with torch.no_grad():
            desc_clean = estimator(clean)
            desc_noisy = estimator(noisy)

        # Descriptors should differ for clean vs noisy
        diff = (desc_clean - desc_noisy).abs().mean()
        assert diff > 0.01, "Estimator should distinguish clean from noisy"

    def test_gradient_flow(self):
        """Gradients should flow through the estimator."""
        estimator = SpectralNoiseEstimator(in_channels=3, noise_dim=32)
        image = torch.randn(1, 3, 256, 256, requires_grad=True)
        desc = estimator(image)
        loss = desc.sum()
        loss.backward()
        assert image.grad is not None


class TestNoiseAwareGate:
    """Tests for the noise-conditioned gating module."""

    def test_output_shape(self):
        """Gate should output [B, C, 1, 1]."""
        gate = NoiseAwareGate(channels=128, noise_dim=32)
        features = torch.randn(2, 128, 20, 20)
        noise_level = torch.randn(2, 32)
        weight = gate(features, noise_level)
        assert weight.shape == (2, 128, 1, 1)

    def test_gate_range(self):
        """Gate values should be in [0, 1] (sigmoid)."""
        gate = NoiseAwareGate(channels=64, noise_dim=32)
        features = torch.randn(4, 64, 10, 10)
        noise_level = torch.randn(4, 32)
        weight = gate(features, noise_level)
        assert (weight >= 0).all() and (weight <= 1).all()

    def test_noise_modulation(self):
        """Different noise levels should produce different gate values."""
        gate = NoiseAwareGate(channels=64, noise_dim=32)
        gate.eval()

        features = torch.randn(1, 64, 10, 10)
        noise_low = torch.zeros(1, 32)
        noise_high = torch.ones(1, 32) * 5.0

        with torch.no_grad():
            gate_low = gate(features, noise_low)
            gate_high = gate(features, noise_high)

        assert not torch.allclose(gate_low, gate_high, atol=1e-4), \
            "Gate should respond to different noise levels"


class TestMultiScaleNoiseEstimator:
    """Tests for multi-scale noise estimation."""

    def test_global_output(self):
        """Without features, should return global descriptor."""
        estimator = MultiScaleNoiseEstimator(in_channels=3, noise_dim=32)
        image = torch.randn(2, 3, 640, 640)
        desc = estimator(image)
        assert desc.shape == (2, 32)

    def test_per_scale_output(self):
        """With features, should return per-scale descriptors."""
        channels = [64, 128, 256]
        estimator = MultiScaleNoiseEstimator(
            in_channels=3, channels_list=channels, noise_dim=32
        )

        image = torch.randn(2, 3, 640, 640)
        features = [
            torch.randn(2, 64, 80, 80),
            torch.randn(2, 128, 40, 40),
            torch.randn(2, 256, 20, 20),
        ]

        descriptors = estimator(image, features)
        assert isinstance(descriptors, list)
        assert len(descriptors) == 3
        for desc in descriptors:
            assert desc.shape == (2, 32)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
