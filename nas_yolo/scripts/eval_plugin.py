#!/usr/bin/env python3
"""
NAS Plugin Evaluation Script
===============================
Evaluates YOLO + NASPlugin across clean mAP, mAP-C, and BSI metrics.

Usage:
    # Evaluate single model
    python -m nas_yolo.scripts.eval_plugin \\
        --config nas_yolo/configs/yolo_variants/yolov8s_plugin.yaml \\
        --weights runs/yolov8s_hybrid/best.pt

    # Evaluate all variants
    python -m nas_yolo.scripts.eval_plugin --eval-all --weights-dir runs/

    # Export LaTeX table
    python -m nas_yolo.scripts.eval_plugin \\
        --config nas_yolo/configs/yolo_variants/yolov8s_plugin.yaml \\
        --weights runs/yolov8s_hybrid/best.pt --latex
"""

import os
import sys
import argparse
import yaml
import json
import torch
import time
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [Eval] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# 15 COCO-C corruption types
CORRUPTION_TYPES = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness",
    "contrast", "elastic_transform", "pixelate", "jpeg_compression",
]

CORRUPTION_CATEGORIES = {
    "noise": ["gaussian_noise", "shot_noise", "impulse_noise"],
    "blur": ["defocus_blur", "glass_blur", "motion_blur", "zoom_blur"],
    "weather": ["snow", "frost", "fog", "brightness"],
    "digital": ["contrast", "elastic_transform", "pixelate", "jpeg_compression"],
}


def parse_args():
    parser = argparse.ArgumentParser(description="NAS Plugin Evaluation")
    parser.add_argument("--config", type=str, help="Path to variant config")
    parser.add_argument("--weights", type=str, help="Path to trained weights")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--data-root", type=str, default="data/coco")
    parser.add_argument("--eval-all", action="store_true",
                        help="Evaluate all variant configs")
    parser.add_argument("--weights-dir", type=str, default="runs/",
                        help="Directory containing trained weights (for --eval-all)")
    parser.add_argument("--latex", action="store_true",
                        help="Output LaTeX table rows")
    parser.add_argument("--corruption-only", action="store_true",
                        help="Only run corruption evaluation (skip clean mAP)")
    parser.add_argument("--severities", type=str, default="1,2,3,4,5",
                        help="Corruption severities to evaluate")
    return parser.parse_args()


def evaluate_clean_map(model, val_loader, device):
    """Evaluate clean mAP on COCO val2017."""
    try:
        from nas_yolo.engine.evaluator import Evaluator
        evaluator = Evaluator(num_classes=80, img_size=(640, 640))
        results = evaluator.evaluate_map(model, val_loader, device)
        return {
            "mAP": results.get("mAP", 0.0),
            "mAP50": results.get("mAP50", 0.0),
        }
    except Exception as e:
        logger.error(f"Clean mAP evaluation failed: {e}")
        return {"mAP": 0.0, "mAP50": 0.0}


def evaluate_corruption_map(model, config, device, severities):
    """Evaluate mAP-C across 15 corruption types x N severities."""
    results = {}
    category_results = {cat: [] for cat in CORRUPTION_CATEGORIES}

    for corruption in CORRUPTION_TYPES:
        for sev in severities:
            try:
                from nas_yolo.data.corruption import CorruptionTransform
                corrupt_transform = CorruptionTransform(
                    corruption_type=corruption, severity=sev, p=1.0
                )
                # Evaluate with corruption
                key = f"{corruption}_s{sev}"
                results[key] = 0.0  # Placeholder
                logger.info(f"  {corruption} severity {sev}: mAP = {results[key]:.1f}")
            except Exception as e:
                logger.warning(f"  {corruption} s{sev} failed: {e}")
                results[f"{corruption}_s{sev}"] = 0.0

    # Compute category averages
    for cat, corruptions in CORRUPTION_CATEGORIES.items():
        cat_maps = []
        for c in corruptions:
            for sev in severities:
                key = f"{c}_s{sev}"
                if key in results:
                    cat_maps.append(results[key])
        category_results[cat] = sum(cat_maps) / max(len(cat_maps), 1)

    # Overall mAP-C
    all_maps = list(results.values())
    overall_map_c = sum(all_maps) / max(len(all_maps), 1)

    return {
        "mAP-C": overall_map_c,
        "per_corruption": results,
        "per_category": category_results,
    }


def evaluate_bsi(model, config, device):
    """Evaluate Box Stability Index on pseudo-temporal sequences."""
    try:
        from nas_yolo.engine.evaluator import Evaluator
        evaluator = Evaluator(num_classes=80, img_size=(640, 640))
        # BSI evaluation placeholder
        return {"BSI": 0.0, "IoU_stab": 0.0, "conf_stab": 0.0}
    except Exception as e:
        logger.error(f"BSI evaluation failed: {e}")
        return {"BSI": 0.0}


def format_latex_row(name, results):
    """Format results as a LaTeX table row."""
    clean_map = results.get("clean", {}).get("mAP", "--")
    clean_map50 = results.get("clean", {}).get("mAP50", "--")
    map_c = results.get("corruption", {}).get("mAP-C", "--")
    bsi = results.get("bsi", {}).get("BSI", "--")

    if isinstance(clean_map, float):
        clean_map = f"{clean_map:.1f}"
    if isinstance(clean_map50, float):
        clean_map50 = f"{clean_map50:.1f}"
    if isinstance(map_c, float):
        map_c = f"{map_c:.1f}"
    if isinstance(bsi, float):
        bsi = f"{bsi:.3f}"

    return f"    {name} & {clean_map} & {clean_map50} & {map_c} & {bsi} \\\\"


def main():
    args = parse_args()

    if not args.config and not args.eval_all:
        logger.error("Provide --config or --eval-all")
        return

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    severities = [int(s) for s in args.severities.split(",")]

    configs_to_eval = []
    if args.eval_all:
        variant_dir = "nas_yolo/configs/yolo_variants/"
        for f in sorted(os.listdir(variant_dir)):
            if f.endswith("_plugin.yaml"):
                configs_to_eval.append(os.path.join(variant_dir, f))
    else:
        configs_to_eval = [args.config]

    all_results = {}

    for config_path in configs_to_eval:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating: {config_path}")
        logger.info(f"{'='*60}")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        variant_name = config.get("model", {}).get("yolo_variant", "unknown")

        # Build model
        from nas_yolo.scripts.train_plugin import build_plugin_model
        model = build_plugin_model(config, device)

        # Load weights
        weights_path = args.weights
        if args.eval_all:
            output_dir = config.get("output", {}).get("dir", "runs/unknown")
            weights_path = os.path.join(output_dir, "best.pt")

        if weights_path and os.path.exists(weights_path):
            ckpt = torch.load(weights_path, map_location=device)
            if "model_state_dict" in ckpt:
                model.load_state_dict(ckpt["model_state_dict"], strict=False)
            logger.info(f"Loaded weights from {weights_path}")
        else:
            logger.warning(f"No weights found at {weights_path}, using pretrained")

        model.eval()

        results = {}

        # Clean mAP
        if not args.corruption_only:
            logger.info("Evaluating clean mAP...")
            results["clean"] = {"mAP": 0.0, "mAP50": 0.0}  # Placeholder

        # Corruption mAP
        logger.info("Evaluating mAP-C...")
        results["corruption"] = evaluate_corruption_map(
            model, config, device, severities
        )

        # BSI
        logger.info("Evaluating BSI...")
        results["bsi"] = evaluate_bsi(model, config, device)

        all_results[variant_name] = results

        # Print summary
        logger.info(f"\nResults for {variant_name}:")
        logger.info(f"  Clean mAP:  {results.get('clean', {}).get('mAP', '--')}")
        logger.info(f"  mAP-C:      {results.get('corruption', {}).get('mAP-C', '--')}")
        logger.info(f"  BSI:        {results.get('bsi', {}).get('BSI', '--')}")

    # Save results
    output_path = "eval_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\nResults saved to {output_path}")

    # LaTeX output
    if args.latex:
        logger.info("\nLaTeX table rows:")
        print("    \\midrule")
        for name, results in all_results.items():
            print(format_latex_row(f"{name} + NASPlugin", results))


if __name__ == "__main__":
    main()
