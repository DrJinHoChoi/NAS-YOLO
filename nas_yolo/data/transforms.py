"""
Detection Transform for YOLO Training and Evaluation
=====================================================
Handles image resize (letterbox), augmentation, and normalization
while maintaining bounding box consistency in xyxy absolute coordinates.
"""

import numpy as np
import torch
from PIL import Image
from typing import Tuple, Dict, Optional


def letterbox_image(
    image: np.ndarray,
    target_size: Tuple[int, int] = (640, 640),
    pad_value: int = 114,
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """Resize image with letterbox (maintain aspect ratio + pad).

    Args:
        image: RGB numpy array [H, W, 3].
        target_size: (height, width) target dimensions.
        pad_value: Padding pixel value (YOLO default=114).

    Returns:
        resized: Letterboxed image [target_h, target_w, 3].
        scale: Scale factor applied.
        pad: (pad_top, pad_left) padding offsets.
    """
    h, w = image.shape[:2]
    target_h, target_w = target_size

    # Compute scale to fit within target
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize
    pil_img = Image.fromarray(image)
    pil_img = pil_img.resize((new_w, new_h), Image.BILINEAR)
    resized = np.array(pil_img)

    # Pad
    pad_top = (target_h - new_h) // 2
    pad_left = (target_w - new_w) // 2
    pad_bottom = target_h - new_h - pad_top
    pad_right = target_w - new_w - pad_left

    padded = np.full((target_h, target_w, 3), pad_value, dtype=np.uint8)
    padded[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized

    return padded, scale, (pad_top, pad_left)


def adjust_boxes_letterbox(
    boxes: np.ndarray,
    scale: float,
    pad: Tuple[int, int],
    target_size: Tuple[int, int],
) -> np.ndarray:
    """Adjust bounding boxes for letterbox resize.

    Args:
        boxes: [N, 4] xyxy absolute coordinates (original image).
        scale: Scale factor from letterbox.
        pad: (pad_top, pad_left) from letterbox.
        target_size: (target_h, target_w).

    Returns:
        Adjusted boxes [N, 4] xyxy in target image coordinates.
    """
    if len(boxes) == 0:
        return boxes

    boxes = boxes.copy().astype(np.float32)
    pad_top, pad_left = pad
    target_h, target_w = target_size

    # Scale and shift
    boxes[:, 0] = boxes[:, 0] * scale + pad_left  # x1
    boxes[:, 1] = boxes[:, 1] * scale + pad_top    # y1
    boxes[:, 2] = boxes[:, 2] * scale + pad_left   # x2
    boxes[:, 3] = boxes[:, 3] * scale + pad_top    # y2

    # Clip to target size
    boxes[:, 0] = np.clip(boxes[:, 0], 0, target_w)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, target_h)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, target_w)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, target_h)

    return boxes


def hsv_augment(
    image: np.ndarray,
    h_gain: float = 0.015,
    s_gain: float = 0.7,
    v_gain: float = 0.4,
) -> np.ndarray:
    """Apply HSV color jitter augmentation.

    Args:
        image: RGB numpy array [H, W, 3] uint8.
        h_gain: Hue gain factor.
        s_gain: Saturation gain factor.
        v_gain: Value gain factor.

    Returns:
        Augmented image [H, W, 3] uint8.
    """
    r = np.random.uniform(-1, 1, 3) * np.array([h_gain, s_gain, v_gain]) + 1
    pil_img = Image.fromarray(image)
    hsv = np.array(pil_img.convert('HSV'), dtype=np.float32)

    hsv[:, :, 0] = (hsv[:, :, 0] * r[0]) % 180  # Hue [0, 180)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * r[1], 0, 255)  # Saturation
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * r[2], 0, 255)  # Value

    hsv = hsv.astype(np.uint8)
    pil_hsv = Image.fromarray(hsv, mode='HSV')
    return np.array(pil_hsv.convert('RGB'))


def random_horizontal_flip(
    image: np.ndarray,
    boxes: np.ndarray,
    p: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Random horizontal flip with box adjustment.

    Args:
        image: [H, W, 3] uint8.
        boxes: [N, 4] xyxy.
        p: Flip probability.

    Returns:
        (flipped_image, flipped_boxes)
    """
    if np.random.random() < p:
        w = image.shape[1]
        image = image[:, ::-1, :].copy()
        if len(boxes) > 0:
            boxes = boxes.copy()
            x1 = boxes[:, 0].copy()
            x2 = boxes[:, 2].copy()
            boxes[:, 0] = w - x2
            boxes[:, 2] = w - x1
    return image, boxes


class DetectionTransform:
    """Detection-aware image transform for YOLO training/evaluation.

    Handles letterbox resize, augmentation, and normalization while
    maintaining bounding box consistency.

    Args:
        img_size: Target (height, width).
        augment: Whether to apply training augmentations.
        hsv_h: Hue augmentation gain.
        hsv_s: Saturation augmentation gain.
        hsv_v: Value augmentation gain.
        flip_p: Horizontal flip probability.
    """

    def __init__(
        self,
        img_size: Tuple[int, int] = (640, 640),
        augment: bool = True,
        hsv_h: float = 0.015,
        hsv_s: float = 0.7,
        hsv_v: float = 0.4,
        flip_p: float = 0.5,
    ):
        self.img_size = img_size
        self.augment = augment
        self.hsv_h = hsv_h
        self.hsv_s = hsv_s
        self.hsv_v = hsv_v
        self.flip_p = flip_p

    def __call__(
        self,
        image: np.ndarray,
        target: Dict,
    ) -> Tuple[torch.Tensor, Dict]:
        """Apply transform to image and target.

        Args:
            image: RGB numpy array [H, W, 3] uint8.
            target: Dict with "boxes" [N, 4] xyxy and "labels" [N].

        Returns:
            (image_tensor [3, H, W] float32 [0,1], adjusted_target)
        """
        boxes = target["boxes"].copy() if len(target["boxes"]) > 0 else target["boxes"]
        labels = target["labels"].copy()

        # Training augmentations
        if self.augment:
            # HSV jitter
            image = hsv_augment(image, self.hsv_h, self.hsv_s, self.hsv_v)
            # Random horizontal flip
            image, boxes = random_horizontal_flip(image, boxes, self.flip_p)

        # Letterbox resize
        image, scale, pad = letterbox_image(image, self.img_size)
        if len(boxes) > 0:
            boxes = adjust_boxes_letterbox(boxes, scale, pad, self.img_size)

            # Filter out invalid boxes (zero area after clip)
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            valid = areas > 1.0
            boxes = boxes[valid]
            labels = labels[valid]

        # Convert to tensor [3, H, W], float32 [0, 1]
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        # Convert target arrays to tensors
        if len(boxes) > 0:
            boxes_tensor = torch.from_numpy(boxes).float()
            labels_tensor = torch.from_numpy(labels).long()
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)

        adjusted_target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
        }

        return image_tensor, adjusted_target

    def __repr__(self) -> str:
        return (f"DetectionTransform(img_size={self.img_size}, "
                f"augment={self.augment})")
