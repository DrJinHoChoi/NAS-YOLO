#!/usr/bin/env python3
"""
NAS-YOLO Latency & Throughput Benchmark
==========================================
Benchmarks model variants on target hardware.

Usage:
    # Benchmark all model scales
    python -m nas_yolo.scripts.benchmark --all-scales

    # Benchmark specific scale
    python -m nas_yolo.scripts.benchmark --scale nano --device cuda

    # Compare with/without NAS module (overhead analysis)
    python -m nas_yolo.scripts.benchmark --overhead-analysis
"""

import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from nas_yolo.models import NASYOLO
from nas_yolo.utils.profiling import profile_model, count_parameters


def parse_args():
    parser = argparse.ArgumentParser(description="NAS-YOLO Benchmark")
    parser.add_argument("--scale", type=str, default="small",
                        choices=["nano", "small", "medium", "large"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-runs", type=int, default=200)
    parser.add_argument("--all-scales", action="store_true")
    parser.add_argument("--overhead-analysis", action="store_true")
    parser.add_argument("--model-scale", type=str, default=None,
                        help="Alias for --scale")
    parser.add_argument("--input-size", type=int, default=None,
                        help="Alias for --img-size")
    parser.add_argument("--output", type=str, default=None,
                        help="Save benchmark results to JSON file")
    return parser.parse_args()


def benchmark_scale(scale: str, device: str, img_size: int,
                    batch_size: int, num_runs: int, use_temporal: bool = True):
    """Benchmark a single model scale."""
    model = NASYOLO(
        num_classes=80,
        model_scale=scale,
        use_noise_gate=use_temporal,
        use_temporal=use_temporal,
    )

    result = profile_model(
        model,
        input_size=(batch_size, 3, img_size, img_size),
        device=device,
        num_runs=num_runs,
    )

    model_info = model.get_model_info()
    result.update(model_info)

    return result


def main():
    args = parse_args()
    device = args.device

    # Apply aliases
    if args.model_scale is not None:
        args.scale = args.model_scale
    if args.input_size is not None:
        args.img_size = args.input_size

    if not torch.cuda.is_available() and device == "cuda":
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    print("=" * 70)
    print("NAS-YOLO Benchmark")
    print("=" * 70)

    if args.all_scales:
        scales = ["nano", "small", "medium", "large"]
    else:
        scales = [args.scale]

    print(f"\n{'Scale':<10} {'Params(M)':>10} {'GFLOPs':>10} {'Latency(ms)':>12} "
          f"{'FPS':>8} {'NAS(%)':<8}")
    print("-" * 70)

    for scale in scales:
        result = benchmark_scale(
            scale, device, args.img_size, args.batch_size, args.num_runs
        )
        print(
            f"{scale:<10} "
            f"{result['params_M']:>10.2f} "
            f"{result['flops_G']:>10.2f} "
            f"{result['latency_ms']:>12.2f} "
            f"{result['fps']:>8.1f} "
            f"{result['nas_overhead_pct']:>6.1f}%"
        )

    # Overhead analysis
    if args.overhead_analysis:
        print(f"\n{'=' * 70}")
        print("NAS Module Overhead Analysis")
        print("=" * 70)

        print(f"\n{'Scale':<10} {'w/ NAS':>12} {'w/o NAS':>12} {'Overhead':>12}")
        print("-" * 50)

        for scale in scales:
            with_nas = benchmark_scale(
                scale, device, args.img_size, args.batch_size, args.num_runs,
                use_temporal=True,
            )
            without_nas = benchmark_scale(
                scale, device, args.img_size, args.batch_size, args.num_runs,
                use_temporal=False,
            )
            overhead_ms = with_nas["latency_ms"] - without_nas["latency_ms"]
            overhead_pct = overhead_ms / without_nas["latency_ms"] * 100

            print(
                f"{scale:<10} "
                f"{with_nas['latency_ms']:>10.2f}ms "
                f"{without_nas['latency_ms']:>10.2f}ms "
                f"+{overhead_ms:.2f}ms ({overhead_pct:.1f}%)"
            )

    # Save results to JSON if requested
    if args.output:
        import json
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
        all_results = {}
        for scale in scales:
            r = benchmark_scale(scale, device, args.img_size, args.batch_size, args.num_runs)
            all_results[scale] = {
                "params_M": r["params_M"],
                "flops_G": r["flops_G"],
                "latency_ms": r["latency_ms"],
                "fps": r["fps"],
                "nas_overhead_pct": r["nas_overhead_pct"],
            }
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.output}")

    print()


if __name__ == "__main__":
    main()
