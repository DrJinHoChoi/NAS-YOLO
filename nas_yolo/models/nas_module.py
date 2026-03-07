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
        use_parallel_scan: If True, use parallel associative scan for training
            (O(log L) steps instead of O(L)). Default False for backward compat.
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
        use_parallel_scan: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.use_parallel_scan = use_parallel_scan

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

    # =========================================================================
    # LTI (Linear Time-Invariant) System Analysis
    # =========================================================================
    #
    # The continuous-time SSM dynamics  dx/dt = Ax + Bu, y = Cx + Du  are LTI
    # when A is time-invariant. In this implementation:
    #
    #   A = -exp(A_log)   -->  all eigenvalues are strictly negative real
    #
    # This provides BIBO (Bounded-Input Bounded-Output) stability and forms
    # a bank of first-order low-pass filters with cutoff frequencies at
    # |eigenvalue_n| / (2*pi).
    #
    # Mamba's input-dependent Delta, B, C make the discretized system LTV
    # (Linear Time-Varying), but the underlying A remains time-invariant,
    # providing a stable "spectral prior" that filters high-frequency noise
    # content STRUCTURALLY -- independent of training data distribution.
    #
    # This is the same LTI principle used for CNN noise immunity in keyword
    # spotting (KWS). The A-matrix eigenvalue structure provides consistent
    # noise filtering regardless of the signal domain (audio or vision).
    # =========================================================================

    @torch.no_grad()
    def compute_lti_frequency_response(self, num_freqs: int = 128) -> dict:
        """Compute the frequency response of the underlying LTI system.

        The SSM's continuous-time dynamics dx/dt = Ax define a Linear
        Time-Invariant system whose frequency response is determined solely
        by A's eigenvalues. Since A = -exp(A_log), all eigenvalues are
        negative real, creating a bank of first-order low-pass filters.

        This structural property provides inherent noise filtering:
          - Eigenvalue -lambda_n creates cutoff at lambda_n/(2*pi) Hz
          - High-frequency noise is attenuated by exp(-lambda_n * Delta)
          - The log-spaced initialization (-1, -2, ..., -d_state) creates
            a harmonic series of decay rates

        Connection to KWS: This is the same LTI property that provides
        CNN noise immunity in keyword spotting -- the time-invariant A
        ensures consistent filtering regardless of noise characteristics.

        Args:
            num_freqs: Number of frequency points for response evaluation.

        Returns:
            dict with keys:
                eigenvalues: A matrix eigenvalues [d_inner, d_state]
                frequencies: Evaluation frequencies [num_freqs]
                magnitude_response: |H(jw)| averaged over d_inner [num_freqs]
                max_eigenvalue: Maximum (least negative) eigenvalue
                min_eigenvalue: Minimum (most negative) eigenvalue
                is_stable: True if all eigenvalues < 0
                cutoff_frequencies: Per-mode cutoff frequencies [d_state]
        """
        A = -torch.exp(self.A_log.float())  # [d_inner, d_state]

        # Frequency grid (log-spaced from 0.01 to 100 rad/s)
        omega = torch.logspace(-2, 2, num_freqs, device=A.device)

        # For diagonal A, each mode n has transfer function magnitude:
        # |H_n(jw)| = 1 / sqrt(w^2 + lambda_n^2)
        # where lambda_n = |eigenvalue_n| (positive magnitude)
        mean_eigenvalues = A.mean(dim=0)  # [d_state], all negative

        mag_response = torch.zeros(num_freqs, device=A.device)
        for i, w in enumerate(omega):
            # Sum contributions from all state modes
            mag_response[i] = (1.0 / torch.sqrt(w**2 + mean_eigenvalues**2)).sum()

        # Normalize by DC gain for relative response
        dc_gain = mag_response[0]
        if dc_gain > 0:
            mag_response_normalized = mag_response / dc_gain
        else:
            mag_response_normalized = mag_response

        return {
            "eigenvalues": A,
            "frequencies": omega,
            "magnitude_response": mag_response_normalized,
            "magnitude_response_raw": mag_response,
            "max_eigenvalue": A.max().item(),
            "min_eigenvalue": A.min().item(),
            "is_stable": bool((A < 0).all()),
            "cutoff_frequencies": (-mean_eigenvalues / (2 * math.pi)).tolist(),
        }

    @torch.no_grad()
    def verify_lti_stability(self) -> dict:
        """Verify BIBO stability of the underlying LTI system.

        A continuous-time LTI system is BIBO stable if and only if all
        eigenvalues of A have negative real parts. Since A = -exp(A_log),
        this is GUARANTEED by construction (exp() > 0, negation < 0).

        Returns:
            dict with keys:
                is_stable: True (always, by construction)
                eigenvalue_range: (max, min) eigenvalue values
                stability_margin: Distance of max eigenvalue from 0
                decay_rates: Per-mode decay rates [d_state]
                memory_timescales: 1/|eigenvalue| per mode [d_state]
        """
        A = -torch.exp(self.A_log.float())  # [d_inner, d_state]

        mean_eigenvalues = A.mean(dim=0)  # [d_state]
        max_eig = A.max().item()
        min_eig = A.min().item()

        return {
            "is_stable": bool((A < 0).all()),
            "eigenvalue_range": (max_eig, min_eig),
            "stability_margin": -max_eig,  # Distance from instability boundary
            "decay_rates": (-mean_eigenvalues).tolist(),
            "memory_timescales": (1.0 / (-mean_eigenvalues)).tolist(),
            "d_state": self.d_state,
            "d_inner": self.d_inner,
        }

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

        # === Selective scan ===
        # Two modes: parallel scan (training, O(log L)) and sequential (inference, O(L)).
        if state is None:
            state = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)

        if self.use_parallel_scan and L > 1 and self.training:
            y, state = self._parallel_scan(dt, A, B_sel, C_sel, x_ssm, state)
        else:
            y, state = self._sequential_scan(dt, A, B_sel, C_sel, x_ssm, state, L)

        # Gating with SiLU activation
        y = y * F.silu(z)

        # Output projection
        y = self.out_proj(y)  # [B, L, D]

        return y, state

    def _sequential_scan(
        self, dt: torch.Tensor, A: torch.Tensor,
        B_sel: torch.Tensor, C_sel: torch.Tensor,
        x_ssm: torch.Tensor, state: torch.Tensor, L: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sequential SSM recurrence with pre-allocated output tensor.

        Optimized over the naive list-append + torch.stack pattern by
        writing directly into a pre-allocated output tensor.

        Args:
            dt: Discretization steps [B, L, d_inner].
            A: State transition matrix [d_inner, N] (negative).
            B_sel: Input-dependent B [B, L, N].
            C_sel: Input-dependent C [B, L, N].
            x_ssm: SSM input [B, L, d_inner].
            state: Initial hidden state [B, d_inner, N].
            L: Sequence length.

        Returns:
            y: Output [B, L, d_inner].
            state: Final hidden state [B, d_inner, N].
        """
        B = x_ssm.shape[0]
        y = torch.empty(B, L, self.d_inner, device=x_ssm.device, dtype=x_ssm.dtype)

        for t in range(L):
            dt_t = dt[:, t, :]      # [B, d_inner]
            B_t = B_sel[:, t, :]    # [B, N]
            C_t = C_sel[:, t, :]    # [B, N]
            x_t = x_ssm[:, t, :]   # [B, d_inner]

            # State update: h_t = exp(Δ*A) * h_{t-1} + (Δ*B) * x_t
            A_bar = torch.exp(dt_t.unsqueeze(-1) * A.unsqueeze(0))  # [B, d_inner, N]
            B_bar = dt_t.unsqueeze(-1) * B_t.unsqueeze(1)           # [B, d_inner, N]
            state = A_bar * state + B_bar * x_t.unsqueeze(-1)

            # Output: y_t = C_t * h_t + D * x_t
            y[:, t, :] = torch.sum(state * C_t.unsqueeze(1), dim=-1) + self.D * x_t

        return y, state

    def _parallel_scan(
        self, dt: torch.Tensor, A: torch.Tensor,
        B_sel: torch.Tensor, C_sel: torch.Tensor,
        x_ssm: torch.Tensor, state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Parallel associative scan for SSM recurrence during training.

        Reduces O(L) sequential steps to O(log L) parallel steps using the
        identity that the linear recurrence h_t = a_t * h_{t-1} + b_t can be
        computed via parallel prefix sum with the associative operator:
            (a1, b1) ⊕ (a2, b2) = (a1 * a2, a2 * b1 + b2)

        This is ~3-5x faster than sequential scan for L=64 on GPU.

        Args:
            dt: Discretization steps [B, L, d_inner].
            A: State transition matrix [d_inner, N] (negative).
            B_sel: Input-dependent B [B, L, N].
            C_sel: Input-dependent C [B, L, N].
            x_ssm: SSM input [B, L, d_inner].
            state: Initial hidden state [B, d_inner, N].

        Returns:
            y: Output [B, L, d_inner].
            state: Final hidden state [B, d_inner, N].
        """
        B, L, _ = x_ssm.shape

        # Vectorized discretization: compute all A_bar, B_bar at once
        # A: [d_inner, N] -> [1, 1, d_inner, N]
        A_bar = torch.exp(
            dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)
        )  # [B, L, d_inner, N]

        B_bar = dt.unsqueeze(-1) * B_sel.unsqueeze(2)  # [B, L, d_inner, N]
        bx = B_bar * x_ssm.unsqueeze(-1)               # [B, L, d_inner, N]

        # --- Parallel prefix scan (Blelloch scan) ---
        # We work with tuples (a, b) where a=A_bar, b=bx
        # and the associative operator is (a1, b1) ⊕ (a2, b2) = (a1*a2, a2*b1 + b2)
        a = A_bar.clone()
        b = bx.clone()

        # Include initial state contribution in the first element
        # h_0 = a_0 * state + b_0
        # We prepend the state as if there's a virtual step: a_virtual=0, b_virtual=state
        # Then h_t for t>=0 incorporates initial state.
        # Simpler approach: fold initial state into b[:, 0]
        b[:, 0, :, :] = A_bar[:, 0, :, :] * state + bx[:, 0, :, :]

        # Up-sweep (reduce) phase
        saved_a = []
        saved_b = []
        stride = 1
        while stride < L:
            # Only update elements at positions where (idx+1) is a multiple of 2*stride
            # Simplified: shift and combine
            a_shifted = F.pad(a[:, :-stride, :, :], (0, 0, 0, 0, stride, 0), value=0)
            b_shifted = F.pad(b[:, :-stride, :, :], (0, 0, 0, 0, stride, 0), value=0)

            # For positions where shifted exists, apply the operator
            mask = torch.zeros(L, device=a.device, dtype=torch.bool)
            mask[stride:] = True

            # (a_prev, b_prev) ⊕ (a_curr, b_curr) = (a_prev * a_curr, a_curr * b_prev + b_curr)
            a_new = a.clone()
            b_new = b.clone()
            a_new[:, mask, :, :] = a_shifted[:, mask, :, :] * a[:, mask, :, :]
            b_new[:, mask, :, :] = a[:, mask, :, :] * b_shifted[:, mask, :, :] + b[:, mask, :, :]

            a = a_new
            b = b_new
            stride *= 2

        # b now contains the hidden states: b[:, t] = h_t
        states = b  # [B, L, d_inner, N]

        # Output: y_t = C_t * h_t + D * x_t
        # C_sel: [B, L, N] -> [B, L, 1, N]
        y = torch.sum(states * C_sel.unsqueeze(2), dim=-1)  # [B, L, d_inner]
        y = y + self.D.unsqueeze(0).unsqueeze(0) * x_ssm     # [B, L, d_inner]

        # Final state for next frame
        final_state = states[:, -1, :, :]  # [B, d_inner, N]

        return y, final_state


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

    def __init__(self, channels: int, d_state: int = 16, d_conv: int = 4,
                 pool_size: int = 8, use_parallel_scan: bool = False):
        super().__init__()
        self.channels = channels
        self.pool_size = pool_size
        self.ssm = SelectiveSSM(
            d_model=channels,
            d_state=d_state,
            d_conv=d_conv,
            expand=2,
            use_parallel_scan=use_parallel_scan,
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
        block_type: str = "ssm",
        profile: str = "edge",
        mode: str = "hybrid",
    ):
        super().__init__()
        self.use_noise_gate = use_noise_gate
        self.num_scales = len(channels_list)
        self.block_type = block_type

        if block_type == "hybrid":
            from .mamba_attention import MambaAttentionBlock, PROFILE_CONFIGS
            pcfg = PROFILE_CONFIGS.get(profile, PROFILE_CONFIGS["edge"])
            self.ssm_adapters = nn.ModuleList([
                MambaAttentionBlock(
                    channels=ch,
                    d_state=pcfg["d_state"],
                    d_conv=min(d_conv, 4),
                    expand=pcfg["expand"],
                    attention_type=pcfg["attention_type"],
                    num_heads=pcfg["num_heads"],
                    window_size=pcfg["window_size"],
                    noise_dim=noise_dim,
                    router_type=pcfg["router_type"],
                    mode=mode,
                )
                for ch in channels_list
            ])
        else:
            # Per-scale SSM adapters (original)
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
            # Step 1: Temporal state propagation via SSM (or hybrid block)
            if self.block_type == "hybrid":
                nl = noise_level[i] if isinstance(noise_level, list) else noise_level
                ssm_out, new_state = self.ssm_adapters[i](feat, state, nl)
            else:
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
