"""
COCO Detection Datasets and DataLoader Builder
================================================
Provides dataset classes for COCO object detection with support for
standard training, pseudo-temporal sequences, and video inputs.

All datasets return targets in {"boxes": [N,4] xyxy, "labels": [N]} format.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from typing import Tuple, Dict, List, Optional, Union

from pycocotools.coco import COCO

from .transforms import DetectionTransform
from .corruption import CorruptionTransform


class COCODetectionDataset(Dataset):
    """COCO format detection dataset.

    Args:
        root_dir: Path to image directory (e.g., "data/coco/train2017").
        ann_file: Path to annotation JSON (e.g., "data/coco/annotations/instances_train2017.json").
        transform: DetectionTransform for resize/augmentation.
        corruption: Optional CorruptionTransform for robustness training.
        img_size: Target image size (height, width).
    """

    def __init__(
        self,
        root_dir: str,
        ann_file: str,
        transform: Optional[DetectionTransform] = None,
        corruption: Optional[CorruptionTransform] = None,
        img_size: Tuple[int, int] = (640, 640),
    ):
        self.root_dir = root_dir
        self.img_size = img_size
        self.transform = transform
        self.corruption = corruption

        # Load COCO annotations
        print(f"  Loading COCO annotations: {ann_file}")
        self.coco = COCO(ann_file)
        self.img_ids = list(sorted(self.coco.getImgIds()))

        # Build category ID mapping: COCO non-contiguous (1-90 with gaps) → continuous 0-79
        cat_ids = sorted(self.coco.getCatIds())
        self.cat_id_to_continuous = {cid: idx for idx, cid in enumerate(cat_ids)}
        self.num_classes = len(cat_ids)

        print(f"  Dataset: {len(self.img_ids)} images, {self.num_classes} classes")

    def __len__(self) -> int:
        return len(self.img_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        """Load image and annotations.

        Returns:
            (image_tensor [3, H, W], target {"boxes": [N,4], "labels": [N]})
        """
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]

        # Load image
        img_path = os.path.join(self.root_dir, img_info["file_name"])
        image = np.array(Image.open(img_path).convert("RGB"))

        # Load annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns = self.coco.loadAnns(ann_ids)

        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann["bbox"]  # COCO format: [x, y, width, height]
            if w < 1 or h < 1:
                continue  # Skip degenerate boxes
            # Convert to xyxy
            x1, y1, x2, y2 = x, y, x + w, y + h
            boxes.append([x1, y1, x2, y2])
            labels.append(self.cat_id_to_continuous[ann["category_id"]])

        boxes = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)
        labels = np.array(labels, dtype=np.int64) if labels else np.zeros((0,), dtype=np.int64)

        # Apply corruption (before geometric transforms)
        if self.corruption is not None:
            image = self.corruption(image)

        # Apply transform (resize, augmentation, normalize)
        target = {"boxes": boxes, "labels": labels}

        if self.transform is not None:
            image, target = self.transform(image, target)
        else:
            # Default: just letterbox and convert to tensor
            from .transforms import letterbox_image, adjust_boxes_letterbox
            image, scale, pad = letterbox_image(image, self.img_size)
            if len(boxes) > 0:
                target["boxes"] = adjust_boxes_letterbox(boxes, scale, pad, self.img_size)
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            target["boxes"] = torch.from_numpy(target["boxes"]).float() if isinstance(target["boxes"], np.ndarray) else target["boxes"]
            target["labels"] = torch.from_numpy(target["labels"]).long() if isinstance(target["labels"], np.ndarray) else target["labels"]

        return image, target


class PseudoTemporalDataset(Dataset):
    """Wrap a detection dataset to produce pseudo-temporal sequences.

    Generates temporal sequences from static images by applying progressive
    corruption, simulating real-world temporal degradation (e.g., increasing
    noise over time in a video stream).

    Args:
        base_dataset: Underlying COCODetectionDataset.
        sequence_length: Number of frames per sequence.
        corruption_type: Type of corruption for temporal simulation.
    """

    def __init__(
        self,
        base_dataset: COCODetectionDataset,
        sequence_length: int = 5,
        corruption_type: str = "gaussian_noise",
    ):
        self.base_dataset = base_dataset
        self.sequence_length = sequence_length
        self.corruption_type = corruption_type

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Tuple[List[torch.Tensor], List[Dict]]:
        """Generate a pseudo-temporal sequence from a single image.

        Returns:
            (image_sequence: List[Tensor [3,H,W]], target_sequence: List[Dict])
        """
        # Get base image info for raw loading
        img_id = self.base_dataset.img_ids[idx]
        img_info = self.base_dataset.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.base_dataset.root_dir, img_info["file_name"])
        raw_image = np.array(Image.open(img_path).convert("RGB"))

        # Get annotations
        ann_ids = self.base_dataset.coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns = self.base_dataset.coco.loadAnns(ann_ids)

        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w < 1 or h < 1:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(self.base_dataset.cat_id_to_continuous[ann["category_id"]])

        boxes = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)
        labels = np.array(labels, dtype=np.int64) if labels else np.zeros((0,), dtype=np.int64)

        # Generate sequence with progressive corruption
        image_sequence = []
        target_sequence = []

        for t in range(self.sequence_length):
            # Progressive severity: frame 0 is clean, last frame has max corruption
            if t == 0:
                corrupted = raw_image.copy()
            else:
                severity = min(5, max(1, int(t / (self.sequence_length - 1) * 4) + 1))
                corrupt_fn = CorruptionTransform(
                    corruption_type=self.corruption_type,
                    severity=severity,
                    p=1.0,
                )
                corrupted = corrupt_fn(raw_image.copy())

            # Apply same transform (no augmentation for temporal consistency)
            target = {"boxes": boxes.copy(), "labels": labels.copy()}

            if self.base_dataset.transform is not None:
                # Use eval mode (no augmentation) for temporal
                from .transforms import DetectionTransform
                eval_transform = DetectionTransform(
                    img_size=self.base_dataset.img_size,
                    augment=False,
                )
                img_tensor, adj_target = eval_transform(corrupted, target)
            else:
                from .transforms import letterbox_image, adjust_boxes_letterbox
                img_lb, scale, pad = letterbox_image(corrupted, self.base_dataset.img_size)
                if len(boxes) > 0:
                    target["boxes"] = adjust_boxes_letterbox(boxes.copy(), scale, pad, self.base_dataset.img_size)
                img_tensor = torch.from_numpy(img_lb).permute(2, 0, 1).float() / 255.0
                adj_target = {
                    "boxes": torch.from_numpy(target["boxes"]).float() if isinstance(target["boxes"], np.ndarray) else target["boxes"],
                    "labels": torch.from_numpy(target["labels"]).long() if isinstance(target["labels"], np.ndarray) else target["labels"],
                }

            image_sequence.append(img_tensor)
            target_sequence.append(adj_target)

        return image_sequence, target_sequence


class TemporalVideoDataset(Dataset):
    """Dataset for real video sequences (future extension).

    Loads consecutive frames from video clips for temporal evaluation.
    Currently a minimal implementation to prevent import errors.

    Args:
        video_root: Path to video frame directories.
        ann_file: Path to video annotation file.
        sequence_length: Frames per clip.
        transform: Optional detection transform.
        img_size: Target image size.
    """

    def __init__(
        self,
        video_root: str,
        ann_file: str = "",
        sequence_length: int = 5,
        transform: Optional[DetectionTransform] = None,
        img_size: Tuple[int, int] = (640, 640),
    ):
        self.video_root = video_root
        self.sequence_length = sequence_length
        self.transform = transform
        self.img_size = img_size

        # Discover video directories
        self.clips = []
        if os.path.exists(video_root):
            for clip_dir in sorted(os.listdir(video_root)):
                clip_path = os.path.join(video_root, clip_dir)
                if os.path.isdir(clip_path):
                    frames = sorted([f for f in os.listdir(clip_path)
                                     if f.endswith(('.jpg', '.png', '.jpeg'))])
                    if len(frames) >= sequence_length:
                        self.clips.append((clip_path, frames))

    def __len__(self) -> int:
        return max(1, len(self.clips))

    def __getitem__(self, idx: int) -> Tuple[List[torch.Tensor], List[Dict]]:
        if not self.clips:
            # Return dummy data
            dummy_img = torch.zeros(3, self.img_size[0], self.img_size[1])
            dummy_target = {"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.int64)}
            return [dummy_img] * self.sequence_length, [dummy_target] * self.sequence_length

        clip_path, frames = self.clips[idx % len(self.clips)]
        start = np.random.randint(0, max(1, len(frames) - self.sequence_length))

        image_sequence = []
        target_sequence = []

        for t in range(self.sequence_length):
            frame_path = os.path.join(clip_path, frames[start + t])
            image = np.array(Image.open(frame_path).convert("RGB"))

            from .transforms import letterbox_image
            image, _, _ = letterbox_image(image, self.img_size)
            img_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

            target = {"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.int64)}

            image_sequence.append(img_tensor)
            target_sequence.append(target)

        return image_sequence, target_sequence


# ============================================================================
# Collate functions
# ============================================================================

def detection_collate_fn(batch):
    """Collate function for standard detection datasets.

    Args:
        batch: List of (image_tensor [3,H,W], target_dict) tuples.

    Returns:
        images: Tensor [B, 3, H, W]
        targets: List[Dict] of length B
    """
    images = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    return images, targets


def temporal_collate_fn(batch):
    """Collate function for temporal/video datasets.

    Reorganizes from batch-of-sequences to sequence-of-batches.

    Args:
        batch: List of (List[T images], List[T targets]) tuples.

    Returns:
        image_sequence: List[Tensor [B,3,H,W]] of length T
        targets_sequence: List[List[Dict]] of length T, each inner list has B items
    """
    seq_len = len(batch[0][0])
    image_sequence = []
    targets_sequence = []

    for t in range(seq_len):
        images_t = torch.stack([item[0][t] for item in batch])
        targets_t = [item[1][t] for item in batch]
        image_sequence.append(images_t)
        targets_sequence.append(targets_t)

    return image_sequence, targets_sequence


# ============================================================================
# DataLoader builder
# ============================================================================

def build_dataloader(
    dataset: Dataset,
    batch_size: int = 16,
    num_workers: int = 8,
    shuffle: bool = True,
) -> DataLoader:
    """Build DataLoader with appropriate collate function.

    Automatically selects detection_collate_fn for standard datasets
    and temporal_collate_fn for temporal/video datasets.

    Args:
        dataset: PyTorch Dataset instance.
        batch_size: Batch size.
        num_workers: Number of data loading workers.
        shuffle: Whether to shuffle data.

    Returns:
        DataLoader instance.
    """
    is_temporal = isinstance(dataset, (PseudoTemporalDataset, TemporalVideoDataset))
    collate_fn = temporal_collate_fn if is_temporal else detection_collate_fn

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=shuffle,  # drop_last for training only
        persistent_workers=num_workers > 0,
    )
    return loader
