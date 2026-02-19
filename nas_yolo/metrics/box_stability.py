"""
Box Stability Index (BSI) — Novel Evaluation Metric
=====================================================
Core contribution C4: A new metric for measuring detection temporal stability.

Motivation:
  Existing metrics (mAP, mAP@50) evaluate single-frame accuracy but ignore
  temporal consistency. In video/streaming applications, bounding box "jitter"
  (frame-to-frame oscillation) is visually distracting and operationally
  problematic, even when per-frame mAP is high.

Definition:
  BSI measures how stable predicted bounding boxes are across consecutive
  frames for the same object instance. Higher BSI = more stable predictions.

  BSI = 1 - mean(ΔIoU_t)  where ΔIoU_t = 1 - IoU(box_t, box_{t-1})

  for matched object instances across frames.

Components:
  1. Instance Matching: Track same object across frames via IoU + class matching
  2. Box Jitter: Frame-to-frame IoU stability
  3. Confidence Jitter: Frame-to-frame score oscillation
  4. Size Stability: Relative size change across frames
  5. Center Stability: Center point drift

Composite BSI:
  BSI = α × BoxIoU_stability + β × Confidence_stability + γ × Center_stability

  where α=0.5, β=0.3, γ=0.2 (default weights)

References:
  This metric is proposed by NAS-YOLO and has no direct precedent in literature.
  Related work: TAO benchmark tracking metrics, MOT stability analysis.
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class FrameDetections:
    """Detections for a single frame."""
    boxes: np.ndarray  # [N, 4] xyxy format
    scores: np.ndarray  # [N]
    class_ids: np.ndarray  # [N]
    frame_idx: int = 0


@dataclass
class StabilityResult:
    """Results from Box Stability Index computation."""
    bsi: float  # Composite Box Stability Index (0-1, higher = more stable)
    box_iou_stability: float  # IoU-based box stability
    confidence_stability: float  # Score consistency
    center_stability: float  # Center point stability
    size_stability: float  # Size consistency
    num_matched_instances: int  # Number of tracked instances
    num_frames: int  # Total frames evaluated
    per_class_bsi: Dict[int, float] = field(default_factory=dict)  # Per-class BSI
    jitter_histogram: Optional[np.ndarray] = None  # Distribution of jitter values


class BoxStabilityIndex:
    """Computes the Box Stability Index over a video sequence.

    Tracks objects across frames using greedy IoU matching and measures
    the temporal consistency of their predicted bounding boxes.

    Args:
        iou_threshold: Minimum IoU for matching detections across frames.
        score_threshold: Minimum confidence to consider a detection.
        alpha: Weight for box IoU stability component.
        beta: Weight for confidence stability component.
        gamma: Weight for center stability component.
    """

    def __init__(
        self,
        iou_threshold: float = 0.5,
        score_threshold: float = 0.25,
        alpha: float = 0.5,
        beta: float = 0.3,
        gamma: float = 0.2,
    ):
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def compute(self, sequence: List[FrameDetections]) -> StabilityResult:
        """Compute BSI over a sequence of detections.

        Args:
            sequence: List of FrameDetections, one per frame (ordered).

        Returns:
            StabilityResult with all stability metrics.
        """
        if len(sequence) < 2:
            return StabilityResult(
                bsi=1.0, box_iou_stability=1.0, confidence_stability=1.0,
                center_stability=1.0, size_stability=1.0,
                num_matched_instances=0, num_frames=len(sequence),
            )

        # Collect frame-pair stability measurements
        iou_stabilities = []
        conf_stabilities = []
        center_stabilities = []
        size_stabilities = []
        per_class_jitters = {}  # class_id → list of jitter values
        all_jitters = []

        for t in range(1, len(sequence)):
            prev_frame = sequence[t - 1]
            curr_frame = sequence[t]

            # Filter by score threshold
            prev_mask = prev_frame.scores >= self.score_threshold
            curr_mask = curr_frame.scores >= self.score_threshold

            prev_boxes = prev_frame.boxes[prev_mask]
            prev_scores = prev_frame.scores[prev_mask]
            prev_classes = prev_frame.class_ids[prev_mask]

            curr_boxes = curr_frame.boxes[curr_mask]
            curr_scores = curr_frame.scores[curr_mask]
            curr_classes = curr_frame.class_ids[curr_mask]

            if len(prev_boxes) == 0 or len(curr_boxes) == 0:
                continue

            # Match detections across frames
            matches = self._match_detections(
                prev_boxes, prev_classes, curr_boxes, curr_classes
            )

            for prev_idx, curr_idx in matches:
                # Box IoU stability
                iou = self._compute_iou(
                    prev_boxes[prev_idx], curr_boxes[curr_idx]
                )
                iou_stabilities.append(iou)
                all_jitters.append(1.0 - iou)

                # Confidence stability
                conf_diff = abs(prev_scores[prev_idx] - curr_scores[curr_idx])
                conf_stabilities.append(1.0 - conf_diff)

                # Center stability
                prev_center = self._box_center(prev_boxes[prev_idx])
                curr_center = self._box_center(curr_boxes[curr_idx])
                prev_diag = self._box_diagonal(prev_boxes[prev_idx])
                center_drift = np.sqrt(
                    (prev_center[0] - curr_center[0]) ** 2 +
                    (prev_center[1] - curr_center[1]) ** 2
                )
                # Normalize by box diagonal
                norm_drift = center_drift / (prev_diag + 1e-8)
                center_stabilities.append(max(0, 1.0 - norm_drift))

                # Size stability
                prev_area = self._box_area(prev_boxes[prev_idx])
                curr_area = self._box_area(curr_boxes[curr_idx])
                size_ratio = min(prev_area, curr_area) / (max(prev_area, curr_area) + 1e-8)
                size_stabilities.append(size_ratio)

                # Per-class tracking
                cls_id = int(prev_classes[prev_idx])
                if cls_id not in per_class_jitters:
                    per_class_jitters[cls_id] = []
                per_class_jitters[cls_id].append(iou)

        # Aggregate
        if not iou_stabilities:
            return StabilityResult(
                bsi=1.0, box_iou_stability=1.0, confidence_stability=1.0,
                center_stability=1.0, size_stability=1.0,
                num_matched_instances=0, num_frames=len(sequence),
            )

        box_iou_stab = float(np.mean(iou_stabilities))
        conf_stab = float(np.mean(conf_stabilities))
        center_stab = float(np.mean(center_stabilities))
        size_stab = float(np.mean(size_stabilities))

        # Composite BSI
        bsi = self.alpha * box_iou_stab + self.beta * conf_stab + self.gamma * center_stab

        # Per-class BSI
        per_class_bsi = {}
        for cls_id, jitters in per_class_jitters.items():
            per_class_bsi[cls_id] = float(np.mean(jitters))

        # Jitter histogram (for visualization)
        jitter_hist = np.histogram(all_jitters, bins=20, range=(0, 1))[0] if all_jitters else None

        return StabilityResult(
            bsi=bsi,
            box_iou_stability=box_iou_stab,
            confidence_stability=conf_stab,
            center_stability=center_stab,
            size_stability=size_stab,
            num_matched_instances=len(iou_stabilities),
            num_frames=len(sequence),
            per_class_bsi=per_class_bsi,
            jitter_histogram=jitter_hist,
        )

    def _match_detections(
        self,
        prev_boxes: np.ndarray,
        prev_classes: np.ndarray,
        curr_boxes: np.ndarray,
        curr_classes: np.ndarray,
    ) -> List[Tuple[int, int]]:
        """Greedy IoU matching between consecutive frame detections.

        Only matches detections of the same class.
        """
        if len(prev_boxes) == 0 or len(curr_boxes) == 0:
            return []

        # Compute pairwise IoU
        iou_matrix = self._pairwise_iou(prev_boxes, curr_boxes)

        # Mask out different-class pairs
        class_match = prev_classes[:, None] == curr_classes[None, :]
        iou_matrix = iou_matrix * class_match

        # Greedy matching
        matches = []
        used_curr = set()

        # Sort by IoU (highest first)
        indices = np.unravel_index(np.argsort(-iou_matrix.ravel()), iou_matrix.shape)

        for prev_idx, curr_idx in zip(indices[0], indices[1]):
            if iou_matrix[prev_idx, curr_idx] < self.iou_threshold:
                break
            if prev_idx in {m[0] for m in matches}:
                continue
            if curr_idx in used_curr:
                continue

            matches.append((int(prev_idx), int(curr_idx)))
            used_curr.add(curr_idx)

        return matches

    @staticmethod
    def _compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
        """Compute IoU between two boxes in xyxy format."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter

        return inter / (union + 1e-8)

    @staticmethod
    def _pairwise_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
        """Compute pairwise IoU between two sets of boxes."""
        n1, n2 = len(boxes1), len(boxes2)
        iou_matrix = np.zeros((n1, n2), dtype=np.float32)

        for i in range(n1):
            for j in range(n2):
                x1 = max(boxes1[i, 0], boxes2[j, 0])
                y1 = max(boxes1[i, 1], boxes2[j, 1])
                x2 = min(boxes1[i, 2], boxes2[j, 2])
                y2 = min(boxes1[i, 3], boxes2[j, 3])

                inter = max(0, x2 - x1) * max(0, y2 - y1)
                area1 = (boxes1[i, 2] - boxes1[i, 0]) * (boxes1[i, 3] - boxes1[i, 1])
                area2 = (boxes2[j, 2] - boxes2[j, 0]) * (boxes2[j, 3] - boxes2[j, 1])
                union = area1 + area2 - inter

                iou_matrix[i, j] = inter / (union + 1e-8)

        return iou_matrix

    @staticmethod
    def _box_center(box: np.ndarray) -> Tuple[float, float]:
        return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

    @staticmethod
    def _box_diagonal(box: np.ndarray) -> float:
        w = box[2] - box[0]
        h = box[3] - box[1]
        return np.sqrt(w ** 2 + h ** 2)

    @staticmethod
    def _box_area(box: np.ndarray) -> float:
        return (box[2] - box[0]) * (box[3] - box[1])


def compute_box_stability(
    predictions_sequence: List[Dict],
    score_threshold: float = 0.25,
    iou_threshold: float = 0.5,
) -> StabilityResult:
    """Convenience function to compute BSI from model output dicts.

    Args:
        predictions_sequence: List of dicts with 'boxes', 'scores', 'class_ids' keys.
        score_threshold: Minimum detection confidence.
        iou_threshold: Minimum IoU for instance matching.

    Returns:
        StabilityResult.
    """
    bsi = BoxStabilityIndex(
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
    )

    sequence = []
    for t, pred in enumerate(predictions_sequence):
        fd = FrameDetections(
            boxes=np.array(pred["boxes"]) if len(pred["boxes"]) > 0 else np.zeros((0, 4)),
            scores=np.array(pred["scores"]) if len(pred["scores"]) > 0 else np.zeros(0),
            class_ids=np.array(pred["class_ids"]) if len(pred["class_ids"]) > 0 else np.zeros(0),
            frame_idx=t,
        )
        sequence.append(fd)

    return bsi.compute(sequence)
