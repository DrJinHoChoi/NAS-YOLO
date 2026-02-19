"""
Noise-Aware State Module (NAS Module)
======================================
Core contribution C1: Temporal Feature State Modeling via Selective State Spaces.

Turns frame-wise detection features into temporally-aware state representations.
The Selective SSM learns to adaptively integrate temporal context while the
noise-aware gating mechanism modulates the state update based on estimated
input degradation.

Architecture:
    Feature_t → Projection → SelectiveSSM → Noise-Gated Update → Refined Feature_t

The SelectiveSSM is inspired by Mamba (Gu & Dao, 2023) but redesigned for:
  - 2D spatial feature maps (not 1D sequences)
  - Real-time latency constraints (< 1ms per frame on edge GPU)
  - Detection-specific state: box stability over temporal smoothing

References:
  - Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces", 2023
  - Smith et al., "Simplified State Space Layers for Sequence Modeling", ICLR 2023
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class SelectiveSSM(nn.Module):
    """Selective State Space Model for temporal feature aggregation.

    Unlike standard SSMs that use fixed discretization, this module learns
    input-dependent discretization parameters (Δ, B, C) enabling selective
    information propagation — crucial for filtering noisy frames while
    retaining clean temporal cues.

    The spatial feature map is flattened to a 1D sequence per channel,
    processed through the SSM recurrence, and reshaped back. This preserves
    spatial structure while enabling temporal reasoning.

    Args:
        d_model: Feature dimension (channel count).
        d_state: SSM hidden state dimension (N in Mamba notation).
        d_conv: Local convolution width for input preprocessing.
        expand: Expansion factor for inner dimension.
        dt_rank: Rank of Δ projection (controls parameter efficiency).
        dt_min: Minimum discretization step (prevents vanishing updates).
        dt_max: Maximum discretization step (prevents exploding updates).
        dt_init: Initialization strategy for Δ ('random' or 'constant').
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: str = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)

        # dt_rank defaults to ceil(d_model / d_state)
        if dt_rank == "auto":
            self.dt_rank = math.ceil(self.d_model / self.d_state)
        else:
            self.dt_rank = int(dt_rank)

        # === Input projection: x → (z, x_ssm) ===
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)

        # === Local convolution for causal preprocessing ===
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True,
        )

        # === Selective parameters (input-dependent) ===
        # x → Δ (discretization step)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        # dt_rank → d_inner
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # Initialize dt bias for uniform distribution in [dt_min, dt_max]
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        )
        # Inverse of softplus for initialization
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # === SSM parameters ===
        # A: state transition (initialized as negative log-spaced, ensuring stability)
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        # D: skip connection (initialized to ones)
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # === Output projection ===
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

    def forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through Selective SSM.

        Args:
            x: Input features [B, L, D] where L is the flattened spatial dim.
            state: Previous hidden state [B, D_inner, N] or None for fresh start.

        Returns:
            y: Output features [B, L, D].
            new_state: Updated hidden state [B, D_inner, N].
        """
        B, L, D = x.shape

        # Input projection: split into gating branch (z) and SSM branch (x_ssm)
        xz = self.in_proj(x)  # [B, L, 2*d_inner]
        x_ssm, z = xz.chunk(2, dim=-1)  # each [B, L, d_inner]

        # Local convolution (causal)
        x_ssm = x_ssm.transpose(1, 2)  # [B, d_inner, L]
        x_ssm = self.conv1d(x_ssm)[:, :, :L]  # causal: trim future
        x_ssm = x_ssm.transpose(1, 2)  # [B, L, d_inner]
        x_ssm = F.silu(x_ssm)

        # Compute selective parameters from input
        x_dbl = self.x_proj(x_ssm)  # [B, L, dt_rank + 2*N]
        dt, B_sel, C_sel = x_dbl.split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1
        )

        # Δ: discretization step (input-dependent, always positive)
        dt = self.dt_proj(dt)  # [B, L, d_inner]
        dt = F.softplus(dt)  # ensure positive

        # A: state transition matrix (always negative for stability)
        A = -torch.exp(self.A_log.float())  # [d_inner, N]

        # === Selective scan (sequential recurrence) ===
        # For edge deployment, we use the sequential form.
        # For training, this can be parallelized via associative scan.
        if state is None:
            state = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)

        outputs = []
        for t in range(L):
            # Discretize: A_bar = exp(Δ * A), B_bar = Δ * B
            dt_t = dt[:, t, :]  # [B, d_inner]
            B_t = B_sel[:, t, :]  # [B, N]
            C_t = C_sel[:, t, :]  # [B, N]
            x_t = x_ssm[:, t, :]  # [B, d_inner]

            # State update: h_t = A_bar * h_{t-1} + B_bar * x_t
            A_bar = torch.exp(dt_t.unsqueeze(-1) * A.unsqueeze(0))  # [B, d_inner, N]
            B_bar = dt_t.unsqueeze(-1) * B_t.unsqueeze(1)  # [B, d_inner, N]
            state = A_bar * state + B_bar * x_t.unsqueeze(-1)

            # Output: y_t = C_t * h_t + D * x_t
            y_t = torch.sum(state * C_t.unsqueeze(1), dim=-1)  # [B, d_inner]
            y_t = y_t + self.D * x_t
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)  # [B, L, d_inner]

        # Gating with SiLU activation
        y = y * F.silu(z)

        # Output projection
        y = self.out_proj(y)  # [B, L, D]

        return y, state


class SpatialSSMAdapter(nn.Module):
    """Adapts the 1D SelectiveSSM for 2D spatial feature maps.

    Uses spatial pooling to reduce sequence length for efficient SSM processing
    (from H*W to pool_size^2). Full-resolution features are preserved via
    residual connection with bilinearly upsampled SSM output.

    Without pooling, a 640×640 input at stride 8 produces 80×80 = 6400 sequential
    SSM steps per frame — impractical for training. Pooling to 8×8 reduces this
    to 64 steps while capturing sufficient spatial structure for temporal state.

    Args:
        channels: Number of feature channels.
        d_state: SSM hidden state dimension.
        d_conv: Local convolution width.
        pool_size: Spatial pooling target (sequence length = pool_size^2).
    """

    def __init__(self, channels: int, d_state: int = 16, d_conv: int = 4, pool_size: int = 8):
        super().__init__()
        self.channels = channels
        self.pool_size = pool_size
        self.ssm = SelectiveSSM(
            d_model=channels,
            d_state=d_state,
            d_conv=d_conv,
            expand=2,
        )
        # Layer norm before SSM (pre-norm architecture)
        self.norm = nn.LayerNorm(channels)

    def forward(
        self, x: torch.Tensor, state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Feature map [B, C, H, W].
            state: Previous SSM state or None.

        Returns:
            out: Refined feature map [B, C, H, W].
            new_state: Updated SSM state.
        """
        B, C, H, W = x.shape

        # Pool to fixed spatial size for efficient SSM processing
        if H > self.pool_size or W > self.pool_size:
            x_pooled = F.adaptive_avg_pool2d(x, self.pool_size)
        else:
            x_pooled = x

        _, _, pH, pW = x_pooled.shape

        # Flatten spatial dimensions: [B, C, pH, pW] → [B, pH*pW, C]
        x_flat = x_pooled.flatten(2).transpose(1, 2)

        # Pre-norm + SSM
        x_norm = self.norm(x_flat)
        ssm_out, new_state = self.ssm(x_norm, state)

        # Reshape: [B, pH*pW, C] → [B, C, pH, pW]
        ssm_spatial = ssm_out.transpose(1, 2).reshape(B, C, pH, pW)

        # Upsample to original resolution if needed
        if pH != H or pW != W:
            ssm_spatial = F.interpolate(
                ssm_spatial, size=(H, W), mode='bilinear', align_corners=False
            )

        # Residual connection
        out = x + ssm_spatial

        return out, new_state


class NoiseAwareStateModule(nn.Module):
    """Full Noise-Aware State Module combining SSM with noise-conditioned gating.

    This is the core contribution of NAS-YOLO. It processes multi-scale detection
    features through:
      1. Spatial SSM for temporal state propagation
      2. Noise-aware gating for adaptive state update
      3. Confidence refinement for box stability

    The module maintains per-scale hidden states across frames, enabling
    the detector to "remember" object locations and suppress noise-induced
    false positives.

    Args:
        channels_list: List of channel counts for each feature scale.
        d_state: SSM hidden state dimension.
        d_conv: Local convolution width.
        use_noise_gate: Whether to apply noise-aware gating (for ablation).
    """

    def __init__(
        self,
        channels_list: list,
        d_state: int = 16,
        d_conv: int = 4,
        use_noise_gate: bool = True,
        noise_dim: int = 32,
    ):
        super().__init__()
        self.use_noise_gate = use_noise_gate
        self.num_scales = len(channels_list)

        # Per-scale SSM adapters
        self.ssm_adapters = nn.ModuleList([
            SpatialSSMAdapter(ch, d_state=d_state, d_conv=d_conv)
            for ch in channels_list
        ])

        # Noise-aware gating (imported separately for modularity)
        if use_noise_gate:
            from .noise_gate import NoiseAwareGate
            self.noise_gates = nn.ModuleList([
                NoiseAwareGate(ch, noise_dim=noise_dim) for ch in channels_list
            ])

        # Confidence refinement head: smooths objectness scores temporally
        self.conf_refiners = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(ch, ch // 4),
                nn.ReLU(inplace=True),
                nn.Linear(ch // 4, 1),
                nn.Sigmoid(),
            )
            for ch in channels_list
        ])

    def forward(
        self,
        features: list,
        states: Optional[list] = None,
        noise_level: Optional[torch.Tensor] = None,
    ) -> Tuple[list, list, list]:
        """Process multi-scale features through noise-aware state modeling.

        Args:
            features: List of feature maps [B, C_i, H_i, W_i] per scale.
            states: List of previous SSM states per scale, or None.
            noise_level: Estimated noise descriptor [B, noise_dim] from NoiseEstimator.

        Returns:
            refined_features: Noise-suppressed, temporally-smoothed features.
            new_states: Updated SSM states for next frame.
            confidence_weights: Per-scale confidence modulation weights [B, 1].
        """
        if states is None:
            states = [None] * self.num_scales

        refined_features = []
        new_states = []
        confidence_weights = []

        for i, (feat, state) in enumerate(zip(features, states)):
            # Step 1: Temporal state propagation via SSM
            ssm_out, new_state = self.ssm_adapters[i](feat, state)

            # Step 2: Noise-aware gating (adaptive blend of current vs. state)
            if self.use_noise_gate and noise_level is not None:
                # Support both per-scale list and global noise descriptor
                nl = noise_level[i] if isinstance(noise_level, list) else noise_level
                gate_weight = self.noise_gates[i](feat, nl)
                # Higher noise → rely more on temporal state (gate_weight → 1)
                # Lower noise → trust current frame more (gate_weight → 0)
                refined = gate_weight * ssm_out + (1 - gate_weight) * feat
            else:
                refined = ssm_out

            # Step 3: Confidence refinement
            conf_weight = self.conf_refiners[i](refined)

            refined_features.append(refined)
            new_states.append(new_state)
            confidence_weights.append(conf_weight)

        return refined_features, new_states, confidence_weights
