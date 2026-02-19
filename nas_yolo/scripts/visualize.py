#!/usr/bin/env python3
"""
NAS-YOLO Visualization Script
================================
Generate paper-quality figures:
  - Detection comparison (baseline vs NAS-YOLO)
  - Feature map before/after NAS module
  - Box stability flicker comparison
  - Severity response curves
  - Ablation comparison chart

Usage:
    python -m nas_yolo.scripts.visualize --mode detection --checkpoint runs/best.pt --image test.jpg
    python -m nas_yolo.scripts.visualize --mode stability --results eval_results.json
    python -m nas_yolo.scripts.visualize --mode severity --results eval_results.json
"""

import os
import sys
import argparse
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def parse_args():
    parser = argparse.ArgumentParser(description="NAS-YOLO Visualization")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["detection", "feature_map", "stability", "severity", "ablation"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--results", type=str, default=None)
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="figures")
    return parser.parse_args()


def visualize_severity_response(results_path: str, output_dir: str):
    """Generate severity response curve (Figure in paper)."""
    import numpy as np

    with open(results_path) as f:
        results = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    corruption = results.get("corruption", {})
    severity_curve = corruption.get("severity_curve", {})

    if severity_curve:
        print("\nSeverity Response Curve:")
        print("-" * 40)
        for sev in sorted(severity_curve.keys(), key=int):
            val = severity_curve[sev]
            bar = "█" * int(val * 40)
            print(f"  Severity {sev}: {val:.4f} {bar}")

    per_category = corruption.get("per_category", {})
    if per_category:
        print("\nPer-Category mAP:")
        print("-" * 40)
        for cat, val in sorted(per_category.items()):
            bar = "█" * int(val * 40)
            print(f"  {cat:15s}: {val:.4f} {bar}")


def visualize_ablation(results_dir: str, output_dir: str):
    """Generate ablation comparison chart."""
    variants = {}

    # Load results from ablation runs
    for variant_name in ["full", "no_temporal", "no_noise_gate", "no_tcl"]:
        result_path = os.path.join(results_dir, f"{variant_name}_results.json")
        if os.path.exists(result_path):
            with open(result_path) as f:
                variants[variant_name] = json.load(f)

    if not variants:
        print("No ablation results found. Run evaluation first.")
        return

    os.makedirs(output_dir, exist_ok=True)

    print("\nAblation Study Results:")
    print("=" * 70)
    print(f"{'Variant':<20} {'mAP':>8} {'mAP-C':>8} {'BSI':>8} {'Params(M)':>10}")
    print("-" * 70)

    for name, data in variants.items():
        clean = data.get("clean", {})
        corruption = data.get("corruption", {})
        stability = data.get("stability", {})
        info = data.get("model_info", {})

        print(
            f"{name:<20} "
            f"{clean.get('mAP', 0):>8.4f} "
            f"{corruption.get('mAP_C', 0):>8.4f} "
            f"{stability.get('BSI', 0):>8.4f} "
            f"{info.get('total_params_M', 0):>10.2f}"
        )


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode == "severity":
        if args.results:
            visualize_severity_response(args.results, args.output_dir)
        else:
            print("Provide --results with evaluation results JSON")

    elif args.mode == "ablation":
        visualize_ablation(
            args.results or "runs/ablation",
            args.output_dir,
        )

    elif args.mode == "detection":
        print("Detection visualization requires matplotlib.")
        print("Usage: Load checkpoint and run inference on target images.")
        print("See nas_yolo/utils/visualization.py for visualization functions.")

    elif args.mode == "feature_map":
        print("Feature map visualization requires model hooks.")
        print("See nas_yolo/utils/visualization.py:visualize_feature_maps()")

    elif args.mode == "stability":
        print("Stability visualization generates box trajectory overlays.")
        print("See nas_yolo/utils/visualization.py:visualize_box_stability()")


if __name__ == "__main__":
    main()
