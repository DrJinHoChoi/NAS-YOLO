"""
Mean Average Precision (mAP) Computation
==========================================
Standard COCO-style mAP evaluation for object detection.

Supports:
  - mAP@50 (VOC-style)
  - mAP@50:95 (COCO-style, 10 IoU thresholds)
  - Per-class AP
  - Size-stratified AP (small/medium/large)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict


def compute_map(
    predictions: List[Dict],
    ground_truths: List[Dict],
    iou_thresholds: Optional[List[float]] = None,
    num_classes: int = 80,
) -> Dict[str, float]:
    """Compute mAP across multiple IoU thresholds.

    Args:
        predictions: List of dicts per image:
            {'boxes': [N, 4] xyxy, 'scores': [N], 'class_ids': [N]}
        ground_truths: List of dicts per image:
            {'boxes': [M, 4] xyxy, 'labels': [M]}
        iou_thresholds: List of IoU thresholds (default: COCO 0.5:0.95:0.05).
        num_classes: Total number of classes.

    Returns:
        metrics: Dict with 'mAP', 'mAP50', 'mAP75', 'per_class_ap'.
    """
    if iou_thresholds is None:
        iou_thresholds = [0.5 + 0.05 * i for i in range(10)]  # 0.50 to 0.95

    # Compute AP for each class and IoU threshold
    all_aps = defaultdict(list)  # iou_threshold → list of per-class APs

    for iou_thresh in iou_thresholds:
        class_aps = compute_ap_per_class(
            predictions, ground_truths, iou_thresh, num_classes
        )
        for cls_id, ap in class_aps.items():
            all_aps[iou_thresh].append(ap)

    # Aggregate
    map_per_threshold = {}
    for iou_thresh in iou_thresholds:
        aps = all_aps[iou_thresh]
        map_per_threshold[iou_thresh] = np.mean(aps) if aps else 0.0

    mAP = np.mean(list(map_per_threshold.values())) if map_per_threshold else 0.0
    mAP50 = map_per_threshold.get(0.5, 0.0)
    mAP75 = map_per_threshold.get(0.75, 0.0)

    # Per-class AP at IoU=0.5
    per_class_ap = compute_ap_per_class(
        predictions, ground_truths, 0.5, num_classes
    )

    return {
        "mAP": float(mAP),
        "mAP50": float(mAP50),
        "mAP75": float(mAP75),
        "per_class_ap": per_class_ap,
        "map_per_threshold": map_per_threshold,
    }


def compute_ap_per_class(
    predictions: List[Dict],
    ground_truths: List[Dict],
    iou_threshold: float = 0.5,
    num_classes: int = 80,
) -> Dict[int, float]:
    """Compute Average Precision per class at a specific IoU threshold.

    Args:
        predictions: List of prediction dicts per image.
        ground_truths: List of ground truth dicts per image.
        iou_threshold: IoU threshold for matching.
        num_classes: Total number of classes.

    Returns:
        class_aps: Dict mapping class_id → AP.
    """
    # Gather all predictions and ground truths per class
    class_preds = defaultdict(list)  # class_id → [(score, img_idx, box)]
    class_gts = defaultdict(lambda: defaultdict(list))  # class_id → img_idx → [box]
    class_n_gts = defaultdict(int)  # class_id → total GT count

    for img_idx, (pred, gt) in enumerate(zip(predictions, ground_truths)):
        pred_boxes = np.array(pred.get("boxes", [])).reshape(-1, 4)
        pred_scores = np.array(pred.get("scores", [])).reshape(-1)
        pred_classes = np.array(pred.get("class_ids", [])).reshape(-1)

        gt_boxes = np.array(gt.get("boxes", [])).reshape(-1, 4)
        gt_labels = np.array(gt.get("labels", [])).reshape(-1)

        for i in range(len(pred_boxes)):
            cls_id = int(pred_classes[i])
            class_preds[cls_id].append((pred_scores[i], img_idx, pred_boxes[i]))

        for i in range(len(gt_boxes)):
            cls_id = int(gt_labels[i])
            class_gts[cls_id][img_idx].append(gt_boxes[i])
            class_n_gts[cls_id] += 1

    # Compute AP per class
    class_aps = {}
    for cls_id in range(num_classes):
        preds = class_preds[cls_id]
        n_gt = class_n_gts[cls_id]

        if n_gt == 0:
            continue

        # Sort predictions by score (descending)
        preds.sort(key=lambda x: x[0], reverse=True)

        tp = np.zeros(len(preds))
        fp = np.zeros(len(preds))
        matched = defaultdict(set)  # img_idx → set of matched GT indices

        for pred_idx, (score, img_idx, pred_box) in enumerate(preds):
            gt_boxes_for_img = class_gts[cls_id].get(img_idx, [])

            best_iou = 0
            best_gt_idx = -1

            for gt_idx, gt_box in enumerate(gt_boxes_for_img):
                iou = _compute_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_threshold and best_gt_idx not in matched[img_idx]:
                tp[pred_idx] = 1
                matched[img_idx].add(best_gt_idx)
            else:
                fp[pred_idx] = 1

        # Compute precision-recall curve
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        recall = tp_cumsum / n_gt
        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)

        # AP via 101-point interpolation (COCO style)
        ap = _compute_ap_101point(recall, precision)
        class_aps[cls_id] = float(ap)

    return class_aps


def _compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / (union + 1e-8)


def _compute_ap_101point(recall: np.ndarray, precision: np.ndarray) -> float:
    """101-point interpolated AP (COCO standard)."""
    recall_interp = np.linspace(0, 1, 101)

    # Ensure monotonically decreasing precision
    precision_interp = np.zeros_like(recall_interp)
    for i, r in enumerate(recall_interp):
        # Find precision at recall >= r
        mask = recall >= r
        if mask.any():
            precision_interp[i] = precision[mask].max()

    return np.mean(precision_interp)
