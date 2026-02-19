"""
Evaluation Engine for NAS-YOLO
================================
Unified evaluation supporting:
  - Standard mAP (COCO-style)
  - Corruption robustness (mAP-C)
  - Box Stability Index (BSI)
  - Latency and throughput benchmarking
  - Per-class, per-size, per-corruption analysis
"""

import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, Optional, List, Tuple

from ..metrics.map import compute_map
from ..metrics.box_stability import BoxStabilityIndex, FrameDetections
from ..metrics.corruption_robustness import CorruptionRobustness, RobustnessReport
from ..data.corruption import CorruptionTransform


class Evaluator:
    """Comprehensive evaluation engine for NAS-YOLO.

    Combines accuracy, robustness, stability, and efficiency metrics
    into a unified evaluation framework suitable for paper reporting.

    Args:
        num_classes: Number of object classes.
        img_size: Input image size (H, W).
        conf_threshold: Confidence threshold for detection filtering.
        nms_threshold: NMS IoU threshold.
        device: Evaluation device.
    """

    def __init__(
        self,
        num_classes: int = 80,
        img_size: Tuple[int, int] = (640, 640),
        conf_threshold: float = 0.25,
        nms_threshold: float = 0.45,
        device: str = "cuda",
    ):
        self.num_classes = num_classes
        self.img_size = img_size
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.device = torch.device(device)

    @torch.no_grad()
    def evaluate_map(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: Optional[torch.device] = None,
    ) -> Dict[str, float]:
        """Evaluate mAP on a dataset.

        Args:
            model: Detection model.
            dataloader: Evaluation data loader.
            device: Override device.

        Returns:
            Dict with 'mAP', 'mAP50', 'mAP75' and per-class AP.
        """
        device = device or self.device
        model.eval()
        model = model.to(device)

        all_predictions = []
        all_ground_truths = []

        for images, targets in dataloader:
            if isinstance(images, list):
                # Temporal dataset: evaluate last frame
                images = images[-1]
                targets = targets[-1]

            images = images.to(device)
            outputs = model(images)

            # Decode outputs
            boxes, scores, class_ids = model.head.decode_outputs(
                outputs, self.img_size, self.conf_threshold
            )

            # Apply NMS per image
            batch_size = images.shape[0]
            for b in range(batch_size):
                # Simple NMS
                keep = self._nms(boxes[b], scores[b], self.nms_threshold)

                pred = {
                    "boxes": boxes[b][keep].cpu().numpy(),
                    "scores": scores[b][keep].cpu().numpy(),
                    "class_ids": class_ids[b][keep].cpu().numpy(),
                }
                all_predictions.append(pred)

                gt = {
                    "boxes": targets[b]["boxes"].cpu().numpy(),
                    "labels": targets[b]["labels"].cpu().numpy(),
                }
                all_ground_truths.append(gt)

        return compute_map(all_predictions, all_ground_truths, num_classes=self.num_classes)

    @torch.no_grad()
    def evaluate_stability(
        self,
        model: nn.Module,
        temporal_loader: DataLoader,
        device: Optional[torch.device] = None,
    ) -> Dict[str, float]:
        """Evaluate Box Stability Index on temporal sequences.

        Args:
            model: Detection model (with temporal support).
            temporal_loader: DataLoader providing frame sequences.
            device: Override device.

        Returns:
            Dict with BSI metrics.
        """
        device = device or self.device
        model.eval()
        model = model.to(device)

        bsi_calculator = BoxStabilityIndex()
        all_results = []

        for image_sequence, targets_sequence in temporal_loader:
            batch_size = image_sequence[0].shape[0]

            # Reset temporal state
            if hasattr(model, "reset_temporal_state"):
                model.reset_temporal_state()

            # Process sequence
            sequence_preds = [[] for _ in range(batch_size)]

            for t, images in enumerate(image_sequence):
                images = images.to(device)
                outputs = model(images)

                boxes, scores, class_ids = model.head.decode_outputs(
                    outputs, self.img_size, self.conf_threshold
                )

                for b in range(batch_size):
                    keep = self._nms(boxes[b], scores[b], self.nms_threshold)
                    fd = FrameDetections(
                        boxes=boxes[b][keep].cpu().numpy(),
                        scores=scores[b][keep].cpu().numpy(),
                        class_ids=class_ids[b][keep].cpu().numpy(),
                        frame_idx=t,
                    )
                    sequence_preds[b].append(fd)

            # Compute BSI per sequence in batch
            for b in range(batch_size):
                result = bsi_calculator.compute(sequence_preds[b])
                all_results.append(result)

        # Aggregate BSI across all sequences
        if not all_results:
            return {"BSI": 0.0, "BoxIoU_stability": 0.0, "Conf_stability": 0.0}

        return {
            "BSI": float(np.mean([r.bsi for r in all_results])),
            "BoxIoU_stability": float(np.mean([r.box_iou_stability for r in all_results])),
            "Conf_stability": float(np.mean([r.confidence_stability for r in all_results])),
            "Center_stability": float(np.mean([r.center_stability for r in all_results])),
            "Size_stability": float(np.mean([r.size_stability for r in all_results])),
            "Num_instances": int(np.sum([r.num_matched_instances for r in all_results])),
        }

    @torch.no_grad()
    def benchmark_latency(
        self,
        model: nn.Module,
        batch_size: int = 1,
        num_warmup: int = 50,
        num_iterations: int = 200,
        device: Optional[torch.device] = None,
    ) -> Dict[str, float]:
        """Benchmark model latency and throughput.

        Args:
            model: Detection model.
            batch_size: Input batch size.
            num_warmup: Number of warmup iterations.
            num_iterations: Number of timed iterations.
            device: Override device.

        Returns:
            Dict with 'latency_ms', 'throughput_fps', 'params_M', 'flops_G'.
        """
        device = device or self.device
        model.eval()
        model = model.to(device)

        dummy_input = torch.randn(batch_size, 3, *self.img_size, device=device)

        # Warmup
        for _ in range(num_warmup):
            _ = model(dummy_input)

        # Timed iterations
        if device.type == "cuda":
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            for _ in range(num_iterations):
                _ = model(dummy_input)
            end.record()
            torch.cuda.synchronize()

            total_ms = start.elapsed_time(end)
        else:
            start_time = time.perf_counter()
            for _ in range(num_iterations):
                _ = model(dummy_input)
            total_ms = (time.perf_counter() - start_time) * 1000

        latency_ms = total_ms / num_iterations
        throughput_fps = 1000.0 / latency_ms * batch_size

        # Model info
        model_info = model.get_model_info() if hasattr(model, "get_model_info") else {}

        return {
            "latency_ms": round(latency_ms, 2),
            "throughput_fps": round(throughput_fps, 1),
            "params_M": model_info.get("total_params_M", 0),
            "nas_overhead_pct": model_info.get("nas_overhead_pct", 0),
        }

    @staticmethod
    def _nms(
        boxes: torch.Tensor,
        scores: torch.Tensor,
        iou_threshold: float,
    ) -> torch.Tensor:
        """Simple NMS implementation.

        Args:
            boxes: [N, 4] xyxy format.
            scores: [N] confidence scores.
            iou_threshold: NMS IoU threshold.

        Returns:
            keep: Indices of kept detections.
        """
        if len(boxes) == 0:
            return torch.zeros(0, dtype=torch.long, device=boxes.device)

        # Sort by score
        order = scores.argsort(descending=True)
        keep = []

        while len(order) > 0:
            idx = order[0]
            keep.append(idx)

            if len(order) == 1:
                break

            # Compute IoU with remaining
            remaining = order[1:]
            xx1 = torch.max(boxes[idx, 0], boxes[remaining, 0])
            yy1 = torch.max(boxes[idx, 1], boxes[remaining, 1])
            xx2 = torch.min(boxes[idx, 2], boxes[remaining, 2])
            yy2 = torch.min(boxes[idx, 3], boxes[remaining, 3])

            inter = torch.clamp(xx2 - xx1, min=0) * torch.clamp(yy2 - yy1, min=0)
            area_idx = (boxes[idx, 2] - boxes[idx, 0]) * (boxes[idx, 3] - boxes[idx, 1])
            area_rem = (boxes[remaining, 2] - boxes[remaining, 0]) * (boxes[remaining, 3] - boxes[remaining, 1])
            union = area_idx + area_rem - inter

            iou = inter / (union + 1e-8)

            # Keep boxes with IoU below threshold
            mask = iou <= iou_threshold
            order = remaining[mask]

        return torch.tensor(keep, dtype=torch.long, device=boxes.device)

    def format_results_table(
        self,
        results: Dict[str, float],
        model_name: str = "NAS-YOLO",
    ) -> str:
        """Format evaluation results as a LaTeX-compatible table row.

        Useful for directly inserting into paper tables.
        """
        row = (
            f"{model_name} & "
            f"{results.get('mAP', 0):.1f} & "
            f"{results.get('mAP50', 0):.1f} & "
            f"{results.get('mAP_C', results.get('map_c', 0)):.1f} & "
            f"{results.get('BSI', 0):.3f} & "
            f"{results.get('latency_ms', 0):.1f} & "
            f"{results.get('params_M', 0):.1f}M \\\\"
        )
        return row
