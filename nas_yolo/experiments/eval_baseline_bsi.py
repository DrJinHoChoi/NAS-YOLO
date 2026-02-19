#!/usr/bin/env python3
"""
Evaluate BSI for Baseline Models (YOLOv8, etc.)
==================================================
Computes Box Stability Index for baseline detectors using
pseudo-temporal sequences, enabling fair comparison with NAS-YOLO.

Since baseline models don't have temporal state, each frame is
processed independently — any instability comes purely from
noise sensitivity.

Usage:
    python -m nas_yolo.experiments.eval_baseline_bsi \
        --model yolov8s \
        --data data/coco/val2017 \
        --ann data/coco/annotations/instances_val2017.json \
        --output baselines/yolov8s_bsi.json
"""

import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from nas_yolo.metrics.box_stability import BoxStabilityIndex, FrameDetections
from nas_yolo.data.corruption import CorruptionTransform


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="yolov8s",
                        choices=["yolov8n", "yolov8s", "yolov9t", "yolov9s", "rtdetr_r18"])
    parser.add_argument("--data", type=str, default="data/coco/val2017")
    parser.add_argument("--ann", type=str, default="data/coco/annotations/instances_val2017.json")
    parser.add_argument("--num-images", type=int, default=500)
    parser.add_argument("--sequence-length", type=int, default=10)
    parser.add_argument("--output", type=str, default="baselines/bsi_results.json")
    return parser.parse_args()


def load_baseline_model(model_name: str):
    """Load a baseline model using ultralytics or other framework."""
    try:
        from ultralytics import YOLO
        model_map = {
            "yolov8n": "yolov8n.pt",
            "yolov8s": "yolov8s.pt",
            "yolov9t": "yolov9t.pt",
            "yolov9s": "yolov9s.pt",
        }
        if model_name in model_map:
            return YOLO(model_map[model_name])
    except ImportError:
        print("ultralytics not installed. Install with: pip install ultralytics")
        return None

    return None


def evaluate_bsi_for_baseline(
    model,
    image_paths: list,
    corruption_types: list = None,
    severity: int = 3,
    sequence_length: int = 10,
):
    """Evaluate BSI for a baseline model on pseudo-temporal sequences.

    For each image:
      1. Create a sequence of progressively corrupted versions
      2. Run the baseline model on each frame independently
      3. Compute BSI across the sequence
    """
    if corruption_types is None:
        corruption_types = ["gaussian_noise", "motion_blur", "fog"]

    bsi_calculator = BoxStabilityIndex()
    all_results = {}

    for corruption_type in corruption_types:
        bsi_scores = []

        for img_path in image_paths:
            # Load image
            from PIL import Image
            img = np.array(Image.open(img_path).convert("RGB"))

            # Generate pseudo-temporal sequence with progressive corruption
            sequence_detections = []
            for t in range(sequence_length):
                # Progressive severity
                sev = max(1, min(5, int(1 + t * (severity - 1) / (sequence_length - 1))))
                corrupt = CorruptionTransform(corruption_type=corruption_type, severity=sev)
                corrupted = corrupt(img)

                # Run baseline model
                results = model(corrupted, verbose=False)

                # Extract detections
                if len(results) > 0 and results[0].boxes is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    scores = results[0].boxes.conf.cpu().numpy()
                    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                else:
                    boxes = np.zeros((0, 4))
                    scores = np.zeros(0)
                    class_ids = np.zeros(0, dtype=int)

                fd = FrameDetections(
                    boxes=boxes, scores=scores, class_ids=class_ids, frame_idx=t
                )
                sequence_detections.append(fd)

            # Compute BSI for this sequence
            result = bsi_calculator.compute(sequence_detections)
            bsi_scores.append(result.bsi)

        all_results[corruption_type] = {
            "mean_bsi": float(np.mean(bsi_scores)),
            "std_bsi": float(np.std(bsi_scores)),
            "num_sequences": len(bsi_scores),
        }

    # Overall BSI
    overall_bsi = np.mean([r["mean_bsi"] for r in all_results.values()])
    all_results["overall"] = {"mean_bsi": float(overall_bsi)}

    return all_results


def main():
    args = parse_args()

    print(f"Evaluating BSI for {args.model}...")

    model = load_baseline_model(args.model)
    if model is None:
        print("Could not load model. Exiting.")
        return

    # Get image paths
    image_dir = args.data
    image_paths = []
    for fname in sorted(os.listdir(image_dir))[:args.num_images]:
        if fname.endswith((".jpg", ".png", ".jpeg")):
            image_paths.append(os.path.join(image_dir, fname))

    print(f"  Using {len(image_paths)} images, T={args.sequence_length}")

    results = evaluate_bsi_for_baseline(
        model, image_paths,
        sequence_length=args.sequence_length,
    )

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"model": args.model, "bsi_results": results}, f, indent=2)

    print(f"\nResults:")
    for key, val in results.items():
        print(f"  {key}: BSI = {val['mean_bsi']:.4f}")

    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
