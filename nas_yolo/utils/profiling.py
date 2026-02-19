"""
Model Profiling Utilities
===========================
Tools for measuring model complexity and efficiency:
  - Parameter counting (total and per-component)
  - FLOPs estimation
  - Latency profiling on target hardware
  - Memory usage analysis
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional


def count_parameters(model: nn.Module, verbose: bool = False) -> Dict[str, int]:
    """Count model parameters (total and per-module).

    Args:
        model: PyTorch model.
        verbose: Print per-module breakdown.

    Returns:
        Dict with parameter counts.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    result = {
        "total": total,
        "trainable": trainable,
        "non_trainable": total - trainable,
    }

    if verbose:
        print(f"{'Module':<50} {'Params':>12} {'%':>6}")
        print("-" * 70)
        for name, module in model.named_children():
            params = sum(p.numel() for p in module.parameters())
            pct = params / total * 100 if total > 0 else 0
            print(f"{name:<50} {params:>12,} {pct:>5.1f}%")
        print("-" * 70)
        print(f"{'Total':<50} {total:>12,}")
        print(f"{'Trainable':<50} {trainable:>12,}")

    return result


def compute_flops(
    model: nn.Module,
    input_size: Tuple[int, ...] = (1, 3, 640, 640),
    device: str = "cpu",
) -> Dict[str, float]:
    """Estimate model FLOPs (multiply-accumulate operations).

    Uses a simple hook-based approach to count operations per layer.
    For accurate FLOPs, consider using fvcore or ptflops libraries.

    Args:
        model: PyTorch model.
        input_size: Input tensor size (B, C, H, W).
        device: Device for computation.

    Returns:
        Dict with 'total_flops', 'total_gflops'.
    """
    total_flops = 0
    hooks = []

    def conv_hook(module, input, output):
        nonlocal total_flops
        batch_size = input[0].size(0)
        output_channels, output_height, output_width = output.size(1), output.size(2), output.size(3)
        kernel_h, kernel_w = module.kernel_size
        in_channels = module.in_channels // module.groups

        flops = (
            batch_size * output_channels * output_height * output_width *
            kernel_h * kernel_w * in_channels * 2  # multiply + add
        )
        total_flops += flops

    def linear_hook(module, input, output):
        nonlocal total_flops
        batch_size = input[0].size(0) if input[0].dim() > 1 else 1
        flops = batch_size * module.in_features * module.out_features * 2
        total_flops += flops

    def bn_hook(module, input, output):
        nonlocal total_flops
        total_flops += input[0].numel() * 2  # normalize + scale/shift

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))
        elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
            hooks.append(module.register_forward_hook(bn_hook))

    model.eval()
    model = model.to(device)
    dummy_input = torch.randn(*input_size, device=device)

    with torch.no_grad():
        model(dummy_input)

    for hook in hooks:
        hook.remove()

    return {
        "total_flops": total_flops,
        "total_gflops": total_flops / 1e9,
        "total_mflops": total_flops / 1e6,
    }


def profile_model(
    model: nn.Module,
    input_size: Tuple[int, ...] = (1, 3, 640, 640),
    device: str = "cpu",
    num_warmup: int = 10,
    num_runs: int = 100,
) -> Dict:
    """Complete model profiling: params, FLOPs, latency.

    Args:
        model: PyTorch model.
        input_size: Input tensor size.
        device: Target device.
        num_warmup: Warmup iterations.
        num_runs: Timed iterations.

    Returns:
        Dict with all profiling metrics.
    """
    import time

    # Parameter count
    params = count_parameters(model, verbose=False)

    # FLOPs
    flops = compute_flops(model, input_size, device)

    # Latency
    model.eval()
    model = model.to(device)
    dummy_input = torch.randn(*input_size, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            model(dummy_input)

    # Timed runs
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        with torch.no_grad():
            for _ in range(num_runs):
                model(dummy_input)
        end.record()
        torch.cuda.synchronize()
        total_ms = start.elapsed_time(end)
    else:
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(num_runs):
                model(dummy_input)
        total_ms = (time.perf_counter() - start) * 1000

    latency_ms = total_ms / num_runs
    fps = 1000.0 / latency_ms * input_size[0]

    # Memory (GPU only)
    gpu_memory_mb = 0
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            model(dummy_input)
        gpu_memory_mb = torch.cuda.max_memory_allocated() / 1e6

    return {
        "params_total": params["total"],
        "params_M": params["total"] / 1e6,
        "flops_G": flops["total_gflops"],
        "latency_ms": round(latency_ms, 2),
        "fps": round(fps, 1),
        "gpu_memory_MB": round(gpu_memory_mb, 1),
        "device": device,
        "input_size": input_size,
    }
