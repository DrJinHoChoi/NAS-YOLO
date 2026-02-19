#!/usr/bin/env python3
"""
NAS-YOLO Evaluation Script
=============================
Comprehensive evaluation: mAP + mAP-C + BSI + Latency

Usage:
    # Standard mAP evaluation
    python -m nas_yolo.scripts.evaluate --checkpoint runs/nas_yolo_s/best.pt --data val

    # Full evaluation (mAP + corruption + stability + latency)
    python -m nas_yolo.scripts.evaluate --checkpoint runs/nas_yolo_s/best.pt --full

    # Corruption-only evaluation
    python -m nas_yolo.scripts.evaluate --checkpoint runs/nas_yolo_s/best.pt --corruption-only

    # Box Stability evaluation
    python -m nas_yolo.scripts.evaluate --checkpoint runs/nas_yolo_s/best.pt --stability-only
"""

import os
import sys
import argparse
import yaml
import json
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from nas_yolo.models import NASYOLO
from nas_yolo.data.dataset import COCODetectionDataset, TemporalVideoDataset, build_dataloader
from nas_yolo.data.transforms import DetectionTransform
from nas_yolo.data.corruption import CorruptionTransform
from nas_yolo.engine.evaluator import Evaluator
from nas_yolo.metrics.corruption_robustness import CorruptionRobustness


def parse_args():
    parser = argparse.ArgumentParser(description="NAS-YOLO Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--config", type=str, default="nas_yolo/configs/default.yaml")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--full", action="store_true", help="Run all evaluations")
    parser.add_argument("--corruption-only", action="store_true")
    parser.add_argument("--stability-only", action="store_true")
    parser.add_argument("--output", type=str, default="eval_results.json")
    # CLI overrides
    parser.add_argument("--model-scale", type=str, default=None,
                        choices=["nano", "small", "medium", "large"],
                        help="Override model scale from config")
    parser.add_argument("--data-root", type=str, default=None,
                        help="Override COCO data root directory")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory for output files (overrides --output path)")
    parser.add_argument("--eval-type", type=str, default=None,
                        choices=["clean", "corruption", "bsi", "stability", "latency", "all"],
                        help="Evaluation type shorthand")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})

    # Apply CLI overrides
    if args.model_scale is not None:
        model_cfg["model_scale"] = args.model_scale
    if args.data_root is not None:
        data_cfg["val_root"] = os.path.join(args.data_root, "val2017")
        data_cfg["val_ann"] = os.path.join(args.data_root, "annotations/instances_val2017.json")
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        args.output = os.path.join(args.output_dir, "eval_results.json")

    # Map eval-type to flags
    if args.eval_type == "all":
        args.full = True
    elif args.eval_type == "corruption":
        args.corruption_only = True
    elif args.eval_type in ("bsi", "stability"):
        args.stability_only = True

    img_size = tuple(data_cfg.get("img_size", [640, 640]))

    # Load model
    model = NASYOLO(
        num_classes=model_cfg.get("num_classes", 80),
        model_scale=model_cfg.get("model_scale", "small"),
        use_noise_gate=model_cfg.get("use_noise_gate", True),
        use_temporal=model_cfg.get("use_temporal", True),
    )

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)

    device = torch.device(args.device)
    model = model.to(device)
    model.eval()

    evaluator = Evaluator(
        num_classes=model_cfg.get("num_classes", 80),
        img_size=img_size,
        device=args.device,
    )

    results = {}

    # Model info
    results["model_info"] = model.get_model_info()
    print(f"\nModel: NAS-YOLO-{model_cfg.get('model_scale', 'small')}")
    print(f"Params: {results['model_info']['total_params_M']:.2f}M")

    # Standard mAP
    if not args.corruption_only and not args.stability_only:
        print("\n[1/4] Evaluating clean mAP...")
        val_transform = DetectionTransform(img_size=img_size, augment=False)
        val_dataset = COCODetectionDataset(
            root_dir=data_cfg.get("val_root", "data/coco/val2017"),
            ann_file=data_cfg.get("val_ann", "data/coco/annotations/instances_val2017.json"),
            transform=val_transform,
            img_size=img_size,
        )
        val_loader = build_dataloader(val_dataset, batch_size=args.batch_size, shuffle=False)
        map_results = evaluator.evaluate_map(model, val_loader, device)
        results["clean"] = map_results
        print(f"  mAP: {map_results['mAP']:.4f}")
        print(f"  mAP@50: {map_results['mAP50']:.4f}")
        print(f"  mAP@75: {map_results['mAP75']:.4f}")

    # Corruption robustness
    if args.full or args.corruption_only:
        print("\n[2/4] Evaluating corruption robustness (mAP-C)...")
        corruption_eval = CorruptionRobustness(
            num_classes=model_cfg.get("num_classes", 80),
        )

        def corruption_eval_fn(corruption_type, severity):
            corruption = None
            if corruption_type is not None:
                corruption = CorruptionTransform(corruption_type=corruption_type, severity=severity)
            val_transform = DetectionTransform(img_size=img_size, augment=False)
            dataset = COCODetectionDataset(
                root_dir=data_cfg.get("val_root", "data/coco/val2017"),
                ann_file=data_cfg.get("val_ann", "data/coco/annotations/instances_val2017.json"),
                transform=val_transform,
                corruption=corruption,
                img_size=img_size,
            )
            loader = build_dataloader(dataset, batch_size=args.batch_size, shuffle=False)
            return evaluator.evaluate_map(model, loader, device)

        report = corruption_eval.evaluate(corruption_eval_fn)
        results["corruption"] = {
            "mAP_C": report.map_c,
            "mean_rCE": report.mean_rce,
            "per_category": report.per_category,
            "severity_curve": report.severity_curve,
        }
        print(corruption_eval.format_report(report))

    # Box Stability
    if args.full or args.stability_only:
        print("\n[3/4] Evaluating Box Stability Index (BSI)...")
        # For BSI, we need temporal sequences
        # If no video data available, use pseudo-temporal from val set
        val_transform = DetectionTransform(img_size=img_size, augment=False)
        val_dataset = COCODetectionDataset(
            root_dir=data_cfg.get("val_root", "data/coco/val2017"),
            ann_file=data_cfg.get("val_ann", "data/coco/annotations/instances_val2017.json"),
            transform=val_transform,
            img_size=img_size,
        )
        from nas_yolo.data.dataset import PseudoTemporalDataset
        temporal_dataset = PseudoTemporalDataset(val_dataset, sequence_length=5)
        temporal_loader = build_dataloader(temporal_dataset, batch_size=4, shuffle=False)

        stability_results = evaluator.evaluate_stability(model, temporal_loader, device)
        results["stability"] = stability_results
        print(f"  BSI: {stability_results['BSI']:.4f}")
        print(f"  Box IoU stability: {stability_results['BoxIoU_stability']:.4f}")
        print(f"  Confidence stability: {stability_results['Conf_stability']:.4f}")

    # Latency benchmark
    if args.full or (not args.corruption_only and not args.stability_only):
        print("\n[4/4] Benchmarking latency...")
        latency = evaluator.benchmark_latency(model, batch_size=1, device=device)
        results["latency"] = latency
        print(f"  Latency: {latency['latency_ms']}ms")
        print(f"  Throughput: {latency['throughput_fps']} FPS")

    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {args.output}")

    # Print LaTeX table row
    if args.full:
        combined = {**results.get("clean", {}), **results.get("corruption", {}),
                    **results.get("stability", {}), **results.get("latency", {})}
        print(f"\nLaTeX row: {evaluator.format_results_table(combined)}")


if __name__ == "__main__":
    main()
