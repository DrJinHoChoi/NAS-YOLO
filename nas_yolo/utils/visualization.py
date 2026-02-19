"""
Visualization Tools for NAS-YOLO
==================================
Paper-quality visualization utilities:
  - Detection result visualization
  - Feature map comparison (before/after NAS Module)
  - Box stability flicker comparison
  - Noise level analysis plots
  - Severity response curves
  - Ablation comparison charts
"""

import numpy as np
from typing import List, Dict, Optional, Tuple


def visualize_detections(
    image: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    class_names: Optional[List[str]] = None,
    score_threshold: float = 0.25,
    line_width: int = 2,
) -> np.ndarray:
    """Draw detection results on an image.

    Args:
        image: Input image [H, W, 3] uint8.
        boxes: Bounding boxes [N, 4] xyxy format.
        scores: Detection scores [N].
        class_ids: Class indices [N].
        class_names: Optional class name list.
        score_threshold: Minimum score to display.
        line_width: Box line width in pixels.

    Returns:
        Annotated image [H, W, 3] uint8.
    """
    result = image.copy()
    H, W = result.shape[:2]

    # Color palette (20 distinct colors)
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
        (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
        (255, 128, 0), (255, 0, 128), (128, 255, 0), (0, 255, 128),
        (0, 128, 255), (128, 0, 255), (200, 200, 200), (100, 100, 100),
    ]

    for i in range(len(boxes)):
        if scores[i] < score_threshold:
            continue

        x1, y1, x2, y2 = boxes[i].astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W - 1, x2), min(H - 1, y2)

        cls_id = int(class_ids[i])
        color = colors[cls_id % len(colors)]

        # Draw box
        for lw in range(line_width):
            result[max(0, y1 - lw):min(H, y1 + lw + 1), x1:x2 + 1] = color  # top
            result[max(0, y2 - lw):min(H, y2 + lw + 1), x1:x2 + 1] = color  # bottom
            result[y1:y2 + 1, max(0, x1 - lw):min(W, x1 + lw + 1)] = color  # left
            result[y1:y2 + 1, max(0, x2 - lw):min(W, x2 + lw + 1)] = color  # right

    return result


def visualize_feature_maps(
    features_before: np.ndarray,
    features_after: np.ndarray,
    num_channels: int = 8,
) -> np.ndarray:
    """Create side-by-side feature map comparison.

    Shows feature channels before and after NAS Module processing,
    highlighting the noise suppression and temporal smoothing effects.

    Args:
        features_before: Feature map before NAS [C, H, W].
        features_after: Feature map after NAS [C, H, W].
        num_channels: Number of channels to visualize.

    Returns:
        Comparison grid [grid_H, grid_W] float32.
    """
    C, H, W = features_before.shape
    n = min(num_channels, C)

    # Select most activated channels
    activation_before = features_before.mean(axis=(1, 2))
    top_channels = np.argsort(-activation_before)[:n]

    rows = []
    for ch in top_channels:
        before_ch = _normalize_feature(features_before[ch])
        after_ch = _normalize_feature(features_after[ch])

        # Difference map (highlights what changed)
        diff = np.abs(after_ch - before_ch)
        diff = _normalize_feature(diff)

        row = np.concatenate([before_ch, after_ch, diff], axis=1)
        rows.append(row)

    grid = np.concatenate(rows, axis=0)
    return grid


def visualize_box_stability(
    frame_detections: List[Dict],
    image_width: int = 640,
    image_height: int = 640,
) -> np.ndarray:
    """Visualize bounding box trajectories across frames.

    Creates a temporal overlay showing how box positions change
    over time. Stable detections appear as sharp boxes; jittery
    detections appear as smeared/ghosted outlines.

    Args:
        frame_detections: List of dicts with 'boxes', 'scores', 'class_ids'.
        image_width: Canvas width.
        image_height: Canvas height.

    Returns:
        Visualization [H, W, 3] uint8.
    """
    canvas = np.zeros((image_height, image_width, 3), dtype=np.float32)
    num_frames = len(frame_detections)

    if num_frames == 0:
        return (canvas * 255).astype(np.uint8)

    for t, det in enumerate(frame_detections):
        alpha = (t + 1) / num_frames  # newer frames are more visible
        boxes = np.array(det.get("boxes", [])).reshape(-1, 4)

        for box in boxes:
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image_width - 1, x2), min(image_height - 1, y2)

            # Draw semi-transparent box
            color = np.array([0.2, 0.8, 0.2]) * alpha  # green with temporal fade
            canvas[y1:y1 + 2, x1:x2] += color
            canvas[y2:y2 + 2, x1:x2] += color
            canvas[y1:y2, x1:x1 + 2] += color
            canvas[y1:y2, x2:x2 + 2] += color

    canvas = np.clip(canvas, 0, 1)
    return (canvas * 255).astype(np.uint8)


def generate_flicker_comparison(
    baseline_dets: List[Dict],
    nasyolo_dets: List[Dict],
    image_size: Tuple[int, int] = (640, 640),
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate side-by-side flicker comparison visualizations.

    Creates overlay images for baseline vs. NAS-YOLO showing
    bounding box stability. This is the key visual for the paper.

    Args:
        baseline_dets: Baseline detector frame-by-frame detections.
        nasyolo_dets: NAS-YOLO frame-by-frame detections.
        image_size: (H, W) of canvas.

    Returns:
        (baseline_viz, nasyolo_viz): Both [H, W, 3] uint8.
    """
    H, W = image_size

    baseline_viz = visualize_box_stability(baseline_dets, W, H)
    nasyolo_viz = visualize_box_stability(nasyolo_dets, W, H)

    return baseline_viz, nasyolo_viz


def plot_severity_response(
    results: Dict[str, Dict[int, float]],
    model_names: Optional[List[str]] = None,
) -> Dict:
    """Prepare data for severity response curve plot.

    Shows how mAP degrades as corruption severity increases,
    comparing NAS-YOLO against baselines.

    Args:
        results: {model_name: {severity: mAP}}.
        model_names: Order of model names for legend.

    Returns:
        Plot data dict (for matplotlib or other rendering).
    """
    if model_names is None:
        model_names = list(results.keys())

    plot_data = {
        "x": sorted(list(next(iter(results.values())).keys())),
        "series": {},
    }

    for name in model_names:
        if name in results:
            severities = sorted(results[name].keys())
            maps = [results[name][s] for s in severities]
            plot_data["series"][name] = maps

    return plot_data


def plot_ablation_comparison(
    ablation_results: Dict[str, Dict[str, float]],
) -> Dict:
    """Prepare data for ablation study comparison chart.

    Compares full model vs. ablated variants (no state, no gate, etc.)

    Args:
        ablation_results: {variant_name: {metric_name: value}}.

    Returns:
        Plot data dict.
    """
    variants = list(ablation_results.keys())
    metrics = list(next(iter(ablation_results.values())).keys())

    return {
        "variants": variants,
        "metrics": metrics,
        "values": {
            variant: [ablation_results[variant].get(m, 0) for m in metrics]
            for variant in variants
        },
    }


def _normalize_feature(feat: np.ndarray) -> np.ndarray:
    """Normalize feature map to [0, 1] for visualization."""
    fmin = feat.min()
    fmax = feat.max()
    if fmax - fmin > 1e-8:
        return (feat - fmin) / (fmax - fmin)
    return np.zeros_like(feat)
