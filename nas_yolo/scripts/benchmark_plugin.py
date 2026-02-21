#!/usr/bin/env python3
"""
NAS Plugin Benchmark Script
==============================
Measures parameter overhead, latency, and FPS for YOLO + NASPlugin.

Usage:
    # Benchmark single model
    python -m nas_yolo.scripts.benchmark_plugin \\
        --config nas_yolo/configs/yolo_variants/yolov8n_plugin.yaml

    # Compare all profiles
    python -m nas_yolo.scripts.benchmark_plugin \\
        --config nas_yolo/configs/yolo_variants/yolov8n_plugin.yaml \\
        --compare-profiles

    # Benchmark all variants
    python -m nas_yolo.scripts.benchmark_plugin --benchmark-all
"""

import os
import sys
import argparse
import yaml
import time
import torch
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [Benchmark] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="NAS Plugin Benchmark")
    parser.add_argument("--config", type=str)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--warmup", type=int, default=50,
                        help="Warmup iterations")
    parser.add_argument("--iterations", type=int, default=200,
                        help="Benchmark iterations")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--compare-profiles", action="store_true",
                        help="Compare all 4 profiles")
    parser.add_argument("--benchmark-all", action="store_true",
                        help="Benchmark all YOLO variant configs")
    return parser.parse_args()


def count_params(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def measure_latency(model, input_shape, device, warmup=50, iterations=200):
    """Measure inference latency in milliseconds."""
    model.eval()
    dummy = torch.randn(*input_shape, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            model(dummy)

    if device == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(iterations):
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(dummy)
            if device == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)  # ms

    avg = sum(times) / len(times)
    std = (sum((t - avg) ** 2 for t in times) / len(times)) ** 0.5
    fps = 1000.0 / avg

    return {"avg_ms": avg, "std_ms": std, "fps": fps}


def benchmark_plugin_standalone(channels_list, profile, mode, device,
                                img_size=640, batch_size=1):
    """Benchmark NASPlugin module in isolation."""
    from nas_yolo.models.nas_plugin import NASPlugin

    plugin = NASPlugin(
        channels_list=channels_list,
        profile=profile,
        mode=mode,
    ).to(device).eval()

    # Count params
    total, _ = count_params(plugin)
    info = plugin.get_plugin_info()

    # Create dummy features matching YOLO neck output shapes
    strides = [8, 16, 32]
    feat_size = img_size
    features = []
    for i, ch in enumerate(channels_list):
        feat_size_i = img_size // strides[i]
        features.append(torch.randn(batch_size, ch, feat_size_i, feat_size_i, device=device))

    dummy_images = torch.randn(batch_size, 3, img_size, img_size, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(20):
            plugin(features, images=dummy_images)
            plugin.reset_state()

    if device == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(100):
            plugin.reset_state()
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            plugin(features, images=dummy_images)
            if device == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

    avg_ms = sum(times) / len(times)

    return {
        "profile": profile,
        "mode": mode,
        "params": total,
        "params_M": total / 1e6,
        "latency_ms": avg_ms,
        "per_block": info.get("per_block", []),
    }


def main():
    args = parse_args()
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    if args.compare_profiles:
        # Compare all 4 profiles for nano and small scales
        logger.info("=" * 70)
        logger.info("NASPlugin Profile Comparison")
        logger.info("=" * 70)

        profiles = ["standard", "lite", "edge", "ultra-lite"]
        for scale_name, channels in [("nano", [64, 128, 256]), ("small", [128, 256, 512])]:
            logger.info(f"\n--- {scale_name} scale (channels={channels}) ---")
            logger.info(f"{'Profile':<12} {'Params':>10} {'Params(M)':>10} {'Latency':>10}")
            logger.info("-" * 45)

            for profile in profiles:
                result = benchmark_plugin_standalone(
                    channels, profile, "hybrid", device,
                    img_size=args.img_size, batch_size=args.batch_size,
                )
                logger.info(
                    f"{profile:<12} {result['params']:>10,} "
                    f"{result['params_M']:>10.4f} "
                    f"{result['latency_ms']:>8.2f}ms"
                )

        # Mode comparison (edge profile, small scale)
        logger.info(f"\n--- Mode comparison (edge, small) ---")
        for mode in ["hybrid", "mamba", "attention"]:
            result = benchmark_plugin_standalone(
                [128, 256, 512], "edge", mode, device,
                img_size=args.img_size, batch_size=args.batch_size,
            )
            logger.info(
                f"{mode:<12} {result['params']:>10,} "
                f"{result['params_M']:>10.4f} "
                f"{result['latency_ms']:>8.2f}ms"
            )

    elif args.benchmark_all:
        variant_dir = "nas_yolo/configs/yolo_variants/"
        for f in sorted(os.listdir(variant_dir)):
            if f.endswith("_plugin.yaml"):
                config_path = os.path.join(variant_dir, f)
                with open(config_path) as fh:
                    config = yaml.safe_load(fh)
                channels = config.get("model", {}).get("channels_list", [128, 256, 512])
                profile = config.get("plugin", {}).get("profile", "edge")
                mode = config.get("plugin", {}).get("mode", "hybrid")
                variant = config.get("model", {}).get("yolo_variant", "unknown")

                result = benchmark_plugin_standalone(
                    channels, profile, mode, device,
                    img_size=args.img_size, batch_size=args.batch_size,
                )
                logger.info(
                    f"{variant:<12} + NASPlugin({profile}): "
                    f"{result['params_M']:.4f}M, "
                    f"{result['latency_ms']:.2f}ms"
                )

    elif args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)

        channels = config.get("model", {}).get("channels_list", [128, 256, 512])
        profile = config.get("plugin", {}).get("profile", "edge")
        mode = config.get("plugin", {}).get("mode", "hybrid")
        variant = config.get("model", {}).get("yolo_variant", "unknown")

        logger.info(f"Benchmarking {variant} + NASPlugin({profile}, {mode})")
        result = benchmark_plugin_standalone(
            channels, profile, mode, device,
            img_size=args.img_size, batch_size=args.batch_size,
        )

        logger.info(f"\nResults:")
        logger.info(f"  Plugin params: {result['params']:,} ({result['params_M']:.4f}M)")
        logger.info(f"  Latency: {result['latency_ms']:.2f}ms")
        logger.info(f"\nPer-block breakdown:")
        for block in result.get("per_block", []):
            logger.info(f"  Scale {block['scale']} ({block['channels']}ch): "
                        f"{block['params']:,} params")
            for k, v in block.get("breakdown", {}).items():
                logger.info(f"    {k}: {v:,}")
    else:
        logger.error("Provide --config, --compare-profiles, or --benchmark-all")


if __name__ == "__main__":
    main()
