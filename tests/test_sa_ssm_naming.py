"""
Tests for SA-SSM (Self-Attention augmented SSM) Naming
=======================================================
Verifies the SA-SSM alias, STRUCTURAL_PROPERTIES attribute,
and cross-domain compatibility.
"""

import os
import sys
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestSASSMAlias:
    """Verify SA_SSM alias is correctly configured."""

    def test_sa_ssm_importable(self):
        """SA_SSM should be importable from mamba_attention."""
        from nas_yolo.models.mamba_attention import SA_SSM
        assert SA_SSM is not None

    def test_sa_ssm_is_mamba_attention_block(self):
        """SA_SSM must be the same class as MambaAttentionBlock."""
        from nas_yolo.models.mamba_attention import SA_SSM, MambaAttentionBlock
        assert SA_SSM is MambaAttentionBlock

    def test_sa_ssm_importable_from_models(self):
        """SA_SSM should be importable from nas_yolo.models."""
        from nas_yolo.models import SA_SSM
        assert SA_SSM is not None

    def test_sa_ssm_instantiation(self):
        """SA_SSM should instantiate like MambaAttentionBlock."""
        from nas_yolo.models.mamba_attention import SA_SSM
        block = SA_SSM(channels=64, d_state=8, mode="hybrid")
        assert block is not None
        assert block.mode == "hybrid"


class TestStructuralProperties:
    """Verify STRUCTURAL_PROPERTIES attribute."""

    def test_structural_properties_exists(self):
        """MambaAttentionBlock must have STRUCTURAL_PROPERTIES dict."""
        from nas_yolo.models.mamba_attention import MambaAttentionBlock
        assert hasattr(MambaAttentionBlock, "STRUCTURAL_PROPERTIES")
        assert isinstance(MambaAttentionBlock.STRUCTURAL_PROPERTIES, dict)

    def test_structural_properties_has_all_keys(self):
        """STRUCTURAL_PROPERTIES must have all 4 required keys."""
        from nas_yolo.models.mamba_attention import MambaAttentionBlock
        props = MambaAttentionBlock.STRUCTURAL_PROPERTIES
        required_keys = [
            "lti_backbone",
            "attention_locality",
            "predetermined_specialization",
            "cross_domain",
        ]
        for key in required_keys:
            assert key in props, f"Missing key: {key}"

    def test_structural_properties_on_instance(self):
        """STRUCTURAL_PROPERTIES should be accessible on instances."""
        from nas_yolo.models.mamba_attention import SA_SSM
        block = SA_SSM(channels=64, d_state=8, mode="hybrid")
        assert hasattr(block, "STRUCTURAL_PROPERTIES")
        assert "lti_backbone" in block.STRUCTURAL_PROPERTIES


class TestSASSMForward:
    """Verify SA-SSM forward pass works for vision inputs."""

    @pytest.fixture
    def sa_ssm_block(self):
        from nas_yolo.models.mamba_attention import SA_SSM
        return SA_SSM(
            channels=64,
            d_state=8,
            expand=1,
            attention_type="se",
            mode="hybrid",
            router_type="fixed",
        )

    def test_forward_2d_vision(self, sa_ssm_block):
        """SA-SSM should process 2D spatial feature maps (vision)."""
        sa_ssm_block.eval()
        x = torch.randn(2, 64, 20, 20)
        with torch.no_grad():
            out, state = sa_ssm_block(x)
        assert out.shape == x.shape, f"Shape mismatch: {out.shape} vs {x.shape}"

    def test_forward_preserves_batch_dim(self, sa_ssm_block):
        """Batch dimension must be preserved."""
        sa_ssm_block.eval()
        for B in [1, 2, 4]:
            x = torch.randn(B, 64, 16, 16)
            with torch.no_grad():
                out, _ = sa_ssm_block(x)
            assert out.shape[0] == B

    def test_dual_branch_output(self, sa_ssm_block):
        """Hybrid mode should use both branches."""
        assert hasattr(sa_ssm_block, "mamba_branch")
        assert hasattr(sa_ssm_block, "attn_branch")

    def test_no_nan_in_output(self, sa_ssm_block):
        """Output should not contain NaN."""
        sa_ssm_block.eval()
        x = torch.randn(2, 64, 16, 16)
        with torch.no_grad():
            out, _ = sa_ssm_block(x)
        assert not torch.isnan(out).any(), "NaN in SA-SSM output"
