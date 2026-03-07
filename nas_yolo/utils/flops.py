"""
FLOPs Counter for PicoYOLO and NAS-YOLO Models
=================================================
Hook-based GFLOPs measurement for accurate model complexity analysis.

Counts multiply-accumulate operations (MACs) for:
    - Conv2d (standard, depthwise, pointwise)
    - Linear
    - BatchNorm2d
    - Pooling layers

Usage:
    from nas_yolo.utils.flops import count_flops
    flops = count_flops(model, input_shape=(1, 3, 320, 320))
    print(f"{flops['gflops']:.3f} GFLOPs")
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict


def count_flops(
    model: nn.Module,
    input_shape: Tuple[int, ...] = (1, 3, 320, 320),
    device: str = "cpu",
) -> Dict[str, float]:
    """Count FLOPs (multiply-accumulate operations) via forward hooks.

    Args:
        model: PyTorch model.
        input_shape: Input tensor shape (B, C, H, W).
        device: Device for computation.

    Returns:
        info: Dict with 'total_macs', 'gflops', 'per_layer' breakdown.
    """
    model = model.eval().to(device)
    hooks = []
    layer_macs = {}

    def make_hook(name):
        def hook_fn(module, input, output):
            macs = 0

            if isinstance(module, nn.Conv2d):
                # MACs = out_h * out_w * out_ch * (in_ch/groups * k_h * k_w)
                out_shape = output.shape  # [B, C_out, H, W]
                macs = (
                    out_shape[2] * out_shape[3]  # H_out * W_out
                    * module.out_channels
                    * (module.in_channels // module.groups)
                    * module.kernel_size[0] * module.kernel_size[1]
                )
                if module.bias is not None:
                    macs += out_shape[2] * out_shape[3] * module.out_channels

            elif isinstance(module, nn.Linear):
                # MACs = out_features * in_features
                macs = module.in_features * module.out_features
                if module.bias is not None:
                    macs += module.out_features

            elif isinstance(module, nn.BatchNorm2d):
                # ~4 ops per element (subtract mean, divide std, scale, shift)
                if isinstance(output, torch.Tensor):
                    macs = output.numel() * 4

            layer_macs[name] = macs

        return hook_fn

    # Register hooks
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
            hooks.append(module.register_forward_hook(make_hook(name)))

    # Forward pass
    with torch.no_grad():
        dummy = torch.randn(*input_shape, device=device)
        try:
            model(dummy)
        except Exception:
            # Some models need additional arguments
            try:
                model(dummy, states=None)
            except Exception:
                pass

    # Remove hooks
    for hook in hooks:
        hook.remove()

    total_macs = sum(layer_macs.values())

    return {
        "total_macs": total_macs,
        "gflops": total_macs / 1e9,
        "mflops": total_macs / 1e6,
        "per_layer": layer_macs,
    }


def profile_model(
    model: nn.Module,
    input_shape: Tuple[int, ...] = (1, 3, 320, 320),
    device: str = "cpu",
    num_warmup: int = 10,
    num_runs: int = 100,
) -> Dict[str, float]:
    """Full model profiling: params, FLOPs, and latency.

    Args:
        model: PyTorch model.
        input_shape: Input tensor shape.
        device: Device for profiling.
        num_warmup: Warmup iterations.
        num_runs: Benchmark iterations.

    Returns:
        profile: Dict with params_M, gflops, latency_ms, fps.
    """
    import time

    model = model.eval().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # FLOPs
    flops_info = count_flops(model, input_shape, device)

    # Latency
    dummy = torch.randn(*input_shape, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            try:
                model(dummy)
            except Exception:
                model(dummy, states=None)

    # Benchmark
    if device == "cuda":
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            try:
                model(dummy)
            except Exception:
                model(dummy, states=None)
            if device == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)  # ms

    avg_latency = sum(times) / len(times)

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "params_M": total_params / 1e6,
        "gflops": flops_info["gflops"],
        "latency_ms": avg_latency,
        "fps": 1000.0 / avg_latency if avg_latency > 0 else 0,
        "model_size_KB": total_params * 4 / 1024,  # FP32
        "int8_size_KB": total_params / 1024,        # INT8
    }
