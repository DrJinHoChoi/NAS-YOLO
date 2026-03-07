"""
Tests for LTI (Linear Time-Invariant) System Properties
=========================================================
Verifies that the SSM A-matrix provides structural noise filtering
via BIBO stability and low-pass frequency response characteristics.
"""

import os
import sys
import math
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nas_yolo.models.nas_module import SelectiveSSM, SpatialSSMAdapter
from nas_yolo.models.mamba_attention import PROFILE_CONFIGS


class TestLTIStability:
    """Verify BIBO stability across all profiles."""

    @pytest.fixture(params=list(PROFILE_CONFIGS.keys()))
    def ssm_for_profile(self, request):
        profile = request.param
        pcfg = PROFILE_CONFIGS[profile]
        return SelectiveSSM(
            d_model=64,
            d_state=pcfg["d_state"],
            d_conv=min(pcfg.get("d_conv", 4), 4),
            expand=pcfg["expand"],
        ), profile

    def test_a_matrix_eigenvalues_always_negative(self, ssm_for_profile):
        """All eigenvalues of A must be strictly negative (BIBO stability)."""
        ssm, profile = ssm_for_profile
        stability = ssm.verify_lti_stability()
        assert stability["is_stable"], \
            f"Profile {profile}: A-matrix has non-negative eigenvalue"

    def test_stability_margin_positive(self, ssm_for_profile):
        """Stability margin (distance from instability) must be > 0."""
        ssm, profile = ssm_for_profile
        stability = ssm.verify_lti_stability()
        assert stability["stability_margin"] > 0, \
            f"Profile {profile}: zero stability margin"

    def test_memory_timescales_finite(self, ssm_for_profile):
        """All memory timescales (1/|eigenvalue|) must be finite and positive."""
        ssm, profile = ssm_for_profile
        stability = ssm.verify_lti_stability()
        for i, tau in enumerate(stability["memory_timescales"]):
            assert 0 < tau < float("inf"), \
                f"Profile {profile}: mode {i} timescale={tau}"


class TestFrequencyResponse:
    """Verify low-pass filtering characteristics."""

    @pytest.fixture
    def ssm(self):
        return SelectiveSSM(d_model=64, d_state=8, expand=1)

    def test_frequency_response_is_lowpass(self, ssm):
        """Magnitude response must decrease at higher frequencies."""
        resp = ssm.compute_lti_frequency_response(num_freqs=64)
        mag = resp["magnitude_response"]

        # First quarter (low freq) should have higher magnitude than
        # last quarter (high freq)
        n = len(mag)
        q1_mean = mag[:n // 4].mean().item()
        q4_mean = mag[3 * n // 4:].mean().item()

        assert q4_mean < q1_mean, \
            f"Not low-pass: q1={q1_mean:.4f}, q4={q4_mean:.4f}"

    def test_dc_gain_is_maximum(self, ssm):
        """DC gain (omega=0) should be the maximum gain."""
        resp = ssm.compute_lti_frequency_response(num_freqs=64)
        mag = resp["magnitude_response"]
        # Normalized response: DC should be ~1.0 (the maximum)
        assert mag[0].item() >= mag[-1].item(), \
            "DC gain is not maximum"

    def test_cutoff_frequencies_positive(self, ssm):
        """All cutoff frequencies must be positive."""
        resp = ssm.compute_lti_frequency_response()
        for i, fc in enumerate(resp["cutoff_frequencies"]):
            assert fc > 0, f"Mode {i}: cutoff={fc}"

    def test_eigenvalues_match_initialization(self, ssm):
        """Eigenvalues should match -1, -2, ..., -d_state initialization."""
        resp = ssm.compute_lti_frequency_response()
        eigenvalues = resp["eigenvalues"]

        # A_log is initialized as log(arange(1, d_state+1))
        # So A = -exp(A_log) = -arange(1, d_state+1)
        # Mean over d_inner dimension
        mean_eig = eigenvalues.mean(dim=0)  # [d_state]
        expected = -torch.arange(1, ssm.d_state + 1, dtype=torch.float32)

        assert torch.allclose(mean_eig.cpu(), expected, atol=1e-4), \
            f"Eigenvalues don't match expected: {mean_eig} vs {expected}"


class TestDeltaSmallPreservesState:
    """Verify that small Delta preserves state (noise → rely on temporal)."""

    def test_exp_delta_a_near_identity_for_small_delta(self):
        """When Delta is small, exp(Delta*A) should be close to identity."""
        ssm = SelectiveSSM(d_model=64, d_state=8, expand=1)
        A = -torch.exp(ssm.A_log.float())  # [d_inner, d_state]

        # Small delta (noisy input → SSM should preserve state)
        delta_small = 0.001
        exp_dA = torch.exp(delta_small * A)

        # exp(delta*A) should be close to 1 for small delta
        # (first-order: exp(delta*A) ≈ 1 + delta*A ≈ 1)
        assert (exp_dA - 1.0).abs().max().item() < 0.05, \
            "Small delta doesn't preserve state"

    def test_exp_delta_a_decays_for_large_delta(self):
        """When Delta is large, exp(Delta*A) should decay toward 0."""
        ssm = SelectiveSSM(d_model=64, d_state=8, expand=1)
        A = -torch.exp(ssm.A_log.float())

        # Large delta (clean input → SSM should forget old state)
        delta_large = 1.0
        exp_dA = torch.exp(delta_large * A)

        # For the fastest mode (eigenvalue = -d_state = -8):
        # exp(-8) ≈ 0.00034
        assert exp_dA.min().item() < 0.01, \
            "Large delta doesn't decay state"


class TestLTIAcrossChannelSizes:
    """Verify LTI properties hold for different channel dimensions."""

    @pytest.mark.parametrize("channels", [32, 64, 128, 256])
    def test_stability_for_all_channel_sizes(self, channels):
        ssm = SelectiveSSM(d_model=channels, d_state=8, expand=1)
        stability = ssm.verify_lti_stability()
        assert stability["is_stable"]

    @pytest.mark.parametrize("channels", [32, 64, 128, 256])
    def test_lowpass_for_all_channel_sizes(self, channels):
        ssm = SelectiveSSM(d_model=channels, d_state=8, expand=1)
        resp = ssm.compute_lti_frequency_response(num_freqs=32)
        mag = resp["magnitude_response"]
        n = len(mag)
        assert mag[:n // 4].mean() > mag[3 * n // 4:].mean()
