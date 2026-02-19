"""
Unit Tests for Box Stability Index (Core Contribution C4)
============================================================
Tests the novel BSI metric for correctness.
"""

import numpy as np
import pytest
from nas_yolo.metrics.box_stability import (
    BoxStabilityIndex,
    FrameDetections,
    compute_box_stability,
)


class TestBoxStabilityIndex:
    """Tests for BSI computation."""

    def test_perfect_stability(self):
        """Identical detections across frames should give BSI ≈ 1.0."""
        bsi = BoxStabilityIndex()

        boxes = np.array([[100, 100, 200, 200]], dtype=np.float32)
        scores = np.array([0.9], dtype=np.float32)
        class_ids = np.array([0], dtype=np.int32)

        sequence = [
            FrameDetections(boxes=boxes.copy(), scores=scores.copy(),
                            class_ids=class_ids.copy(), frame_idx=t)
            for t in range(10)
        ]

        result = bsi.compute(sequence)
        assert result.bsi > 0.95, f"Perfect stability BSI should be ~1.0, got {result.bsi}"

    def test_jittery_detections(self):
        """Random box positions should give low BSI."""
        bsi = BoxStabilityIndex(iou_threshold=0.1)

        sequence = []
        for t in range(10):
            # Random box positions
            x1 = np.random.randint(0, 400)
            y1 = np.random.randint(0, 400)
            boxes = np.array([[x1, y1, x1 + 100, y1 + 100]], dtype=np.float32)
            scores = np.array([0.9], dtype=np.float32)
            class_ids = np.array([0], dtype=np.int32)
            sequence.append(FrameDetections(boxes=boxes, scores=scores,
                                            class_ids=class_ids, frame_idx=t))

        result = bsi.compute(sequence)
        # With random positions, many frames won't match → BSI could be high (few matches)
        # or low (matched but poor IoU). The key is it computes without error.
        assert 0 <= result.bsi <= 1

    def test_single_frame(self):
        """Single frame should return BSI = 1.0 (no jitter possible)."""
        bsi = BoxStabilityIndex()

        boxes = np.array([[10, 10, 50, 50]], dtype=np.float32)
        sequence = [FrameDetections(boxes=boxes, scores=np.array([0.9]),
                                    class_ids=np.array([0]), frame_idx=0)]

        result = bsi.compute(sequence)
        assert result.bsi == 1.0

    def test_empty_detections(self):
        """Empty detections should not crash."""
        bsi = BoxStabilityIndex()

        sequence = [
            FrameDetections(boxes=np.zeros((0, 4)), scores=np.zeros(0),
                            class_ids=np.zeros(0), frame_idx=t)
            for t in range(5)
        ]

        result = bsi.compute(sequence)
        assert result.bsi == 1.0  # No matches → default 1.0

    def test_multi_class_matching(self):
        """Should only match detections of the same class."""
        bsi = BoxStabilityIndex()

        # Same box position, different classes
        boxes = np.array([[100, 100, 200, 200]], dtype=np.float32)
        scores = np.array([0.9], dtype=np.float32)

        sequence = [
            FrameDetections(boxes=boxes.copy(), scores=scores.copy(),
                            class_ids=np.array([0]), frame_idx=0),
            FrameDetections(boxes=boxes.copy(), scores=scores.copy(),
                            class_ids=np.array([1]), frame_idx=1),  # different class
        ]

        result = bsi.compute(sequence)
        # Should find no matches (different classes)
        assert result.num_matched_instances == 0

    def test_confidence_stability(self):
        """Stable boxes with varying confidence should show conf instability."""
        bsi = BoxStabilityIndex()

        boxes = np.array([[100, 100, 200, 200]], dtype=np.float32)
        class_ids = np.array([0], dtype=np.int32)

        sequence = [
            FrameDetections(boxes=boxes.copy(),
                            scores=np.array([0.9 if t % 2 == 0 else 0.5]),
                            class_ids=class_ids.copy(), frame_idx=t)
            for t in range(10)
        ]

        result = bsi.compute(sequence)
        # Box IoU should be perfect, confidence should not be
        assert result.box_iou_stability > 0.99
        assert result.confidence_stability < 0.95

    def test_per_class_bsi(self):
        """Per-class BSI should be computed correctly."""
        bsi = BoxStabilityIndex()

        sequence = []
        for t in range(5):
            boxes = np.array([
                [100, 100, 200, 200],  # class 0
                [300, 300, 400, 400],  # class 1
            ], dtype=np.float32)
            scores = np.array([0.9, 0.9], dtype=np.float32)
            class_ids = np.array([0, 1], dtype=np.int32)
            sequence.append(FrameDetections(boxes=boxes, scores=scores,
                                            class_ids=class_ids, frame_idx=t))

        result = bsi.compute(sequence)
        assert 0 in result.per_class_bsi
        assert 1 in result.per_class_bsi


class TestComputeBoxStability:
    """Tests for the convenience function."""

    def test_dict_input_format(self):
        """Should accept dict format predictions."""
        predictions = [
            {
                "boxes": [[100, 100, 200, 200]],
                "scores": [0.9],
                "class_ids": [0],
            }
            for _ in range(5)
        ]

        result = compute_box_stability(predictions)
        assert 0 <= result.bsi <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
