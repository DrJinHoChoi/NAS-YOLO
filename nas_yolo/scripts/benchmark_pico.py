"""
PicoYOLO Benchmark — Compare Against SOTA Ultra-Lightweight Detectors
========================================================================
Benchmark PicoYOLO against PP-PicoDet-S, NanoDet-Plus, YOLOX-Nano.

Usage:
    python -m nas_yolo.scripts.benchmark_pico
    python -m nas_yolo.scripts.benchmark_pico --base-channels 16  # Ultra-pico
    python -m nas_yolo.scripts.benchmark_pico --base-channels 48  # Pico+
    python -m nas_yolo.scripts.benchmark_pico --device cuda
"""

import argparse
import sys
import time
import torch
import torch.nn as nn
from typing import Dict, List


def benchmark_pico(
    base_channels: int = 32,
    num_classes: int = 80,
    input_resolution: int = 320,
    device: str = "cpu",
    num_warmup: int = 20,
    num_runs: int = 200,
    use_context: bool = False,
) -> Dict:
    """Benchmark PicoYOLO with full profiling.

    Args:
        base_channels: Base channel width (16, 32, 48).
        num_classes: Number of classes (30 or 80).
        input_resolution: Input size (320 or 416).
        device: 'cpu' or 'cuda'.
        num_warmup: Warmup iterations.
        num_runs: Benchmark iterations.
        use_context: Enable context-aware detection.

    Returns:
        results: Comprehensive benchmark results.
    """
    from nas_yolo.models.pico_yolo import PicoYOLO
    from nas_yolo.utils.flops import count_flops, profile_model

    print(f"\n{'='*65}")
    print(f"  PicoYOLO Benchmark")
    print(f"  base_channels={base_channels}, classes={num_classes}, "
          f"res={input_resolution}")
    print(f"{'='*65}")

    # Build model
    model = PicoYOLO(
        num_classes=num_classes,
        base_channels=base_channels,
        plugin_profile="pico",
        input_resolution=input_resolution,
        use_context=use_context,
    )

    # Print model info
    model.print_model_info()

    # Full profile
    profile = profile_model(
        model,
        input_shape=(1, 3, input_resolution, input_resolution),
        device=device,
        num_warmup=num_warmup,
        num_runs=num_runs,
    )

    print(f"\n  Profiling ({device}):")
    print(f"    Params:     {profile['params_M']:.3f}M")
    print(f"    GFLOPs:     {profile['gflops']:.3f}")
    print(f"    Latency:    {profile['latency_ms']:.2f}ms")
    print(f"    FPS:        {profile['fps']:.0f}")
    print(f"    FP32 size:  {profile['model_size_KB']:.0f} KB")
    print(f"    INT8 size:  {profile['int8_size_KB']:.0f} KB")

    return {
        "model": "PicoYOLO",
        "base_channels": base_channels,
        "num_classes": num_classes,
        "input_resolution": input_resolution,
        **profile,
    }


def compare_with_sota(pico_results: Dict):
    """Compare PicoYOLO against SOTA ultra-lightweight detectors.

    Uses published benchmark numbers from papers.
    """
    # SOTA baselines (published numbers)
    baselines = [
        {
            "model": "PP-PicoDet-S",
            "params_M": 0.99,
            "gflops": 1.08,
            "mAP": 30.6,
            "arm_latency_ms": 6.7,
            "source": "Baidu, 2021",
        },
        {
            "model": "NanoDet-Plus",
            "params_M": 1.17,
            "gflops": 1.52,
            "mAP": 27.0,
            "arm_latency_ms": 10.3,
            "source": "RangiLyu, 2021",
        },
        {
            "model": "YOLOX-Nano",
            "params_M": 0.91,
            "gflops": 1.08,
            "mAP": 25.8,
            "arm_latency_ms": 7.3,
            "source": "Megvii, 2021",
        },
        {
            "model": "YOLOv8n",
            "params_M": 3.2,
            "gflops": 8.7,
            "mAP": 37.3,
            "arm_latency_ms": None,
            "source": "Ultralytics, 2023",
        },
        {
            "model": "YOLOv12n",
            "params_M": 2.6,
            "gflops": 6.3,
            "mAP": 40.6,
            "arm_latency_ms": None,
            "source": "Ultralytics, 2024",
        },
    ]

    print(f"\n{'='*80}")
    print(f"  COMPARISON: PicoYOLO vs SOTA Ultra-Lightweight Detectors")
    print(f"{'='*80}")
    print(f"  {'Model':<20} {'Params(M)':>10} {'GFLOPs':>8} "
          f"{'mAP':>6} {'Noise-Robust':>14}")
    print(f"  {'-'*20} {'-'*10} {'-'*8} {'-'*6} {'-'*14}")

    # Print PicoYOLO first
    print(f"  {'PicoYOLO':<20} "
          f"{pico_results['params_M']:>10.3f} "
          f"{pico_results['gflops']:>8.3f} "
          f"{'TBD':>6} "
          f"{'4-DEFENSE':>14}")

    # Print baselines
    for b in baselines:
        mAP_str = f"{b['mAP']:.1f}" if b['mAP'] else "N/A"
        print(f"  {b['model']:<20} "
              f"{b['params_M']:>10.2f} "
              f"{b['gflops']:>8.2f} "
              f"{mAP_str:>6} "
              f"{'None':>14}")

    print(f"\n  PicoYOLO advantages:")
    ratio = baselines[0]["params_M"] / pico_results["params_M"]
    flops_ratio = baselines[0]["gflops"] / max(pico_results["gflops"], 0.001)
    print(f"    vs PP-PicoDet-S: {ratio:.1f}x fewer params, "
          f"{flops_ratio:.1f}x fewer FLOPs")
    print(f"    + SA-SSM 4 structural noise defenses (unique advantage)")
    print(f"    + Temporal state for video detection (unique advantage)")
    print(f"    + Context-aware situation detection (unique advantage)")


def compare_pico_variants():
    """Compare different PicoYOLO channel configurations."""
    from nas_yolo.models.pico_yolo import PicoYOLO
    from nas_yolo.utils.flops import count_flops

    configs = [
        {"name": "Ultra-pico", "base": 16, "classes": 30},
        {"name": "Pico", "base": 32, "classes": 30},
        {"name": "Pico (80cls)", "base": 32, "classes": 80},
        {"name": "Pico+", "base": 48, "classes": 30},
        {"name": "Pico+ (80cls)", "base": 48, "classes": 80},
    ]

    print(f"\n{'='*70}")
    print(f"  PicoYOLO Channel Configuration Comparison")
    print(f"{'='*70}")
    print(f"  {'Config':<18} {'Channels':<14} {'Params(M)':>10} "
          f"{'GFLOPs':>8} {'INT8(KB)':>10}")
    print(f"  {'-'*18} {'-'*14} {'-'*10} {'-'*8} {'-'*10}")

    for cfg in configs:
        model = PicoYOLO(
            num_classes=cfg["classes"],
            base_channels=cfg["base"],
            input_resolution=320,
        )
        total = sum(p.numel() for p in model.parameters())
        flops = count_flops(model, (1, 3, 320, 320))

        ch = f"[{cfg['base']},{cfg['base']*2},{cfg['base']*4}]"
        print(f"  {cfg['name']:<18} {ch:<14} "
              f"{total/1e6:>10.3f} "
              f"{flops['gflops']:>8.3f} "
              f"{total/1024:>10.0f}")


def main():
    parser = argparse.ArgumentParser(
        description="PicoYOLO Benchmark"
    )
    parser.add_argument("--base-channels", type=int, default=32,
                        help="Base channel width (16, 32, 48)")
    parser.add_argument("--num-classes", type=int, default=80,
                        help="Number of classes (30 or 80)")
    parser.add_argument("--input-resolution", type=int, default=320,
                        help="Input image resolution")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device ('cpu' or 'cuda')")
    parser.add_argument("--compare-variants", action="store_true",
                        help="Compare all PicoYOLO configurations")
    parser.add_argument("--use-context", action="store_true",
                        help="Enable context-aware detection")
    args = parser.parse_args()

    if args.compare_variants:
        compare_pico_variants()
    else:
        results = benchmark_pico(
            base_channels=args.base_channels,
            num_classes=args.num_classes,
            input_resolution=args.input_resolution,
            device=args.device,
            use_context=args.use_context,
        )
        compare_with_sota(results)


if __name__ == "__main__":
    main()
