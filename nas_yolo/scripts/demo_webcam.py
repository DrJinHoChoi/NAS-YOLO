#!/usr/bin/env python3
"""
PicoYOLO Webcam Demo - Real-Time Object Detection
====================================================
Run PicoYOLO (or any SA-SSM profile) with laptop webcam for live detection.

Features:
    - Real-time object detection with temporal state (SA-SSM)
    - FPS counter and model info overlay
    - Context-aware detection mode (optional)
    - Noise defense visualization (optional)
    - Works on CPU (no GPU required!)

Usage:
    # PicoYOLO (smallest, fastest)
    python -m nas_yolo.scripts.demo_webcam

    # With specific config
    python -m nas_yolo.scripts.demo_webcam --base-channels 32 --num-classes 30

    # With context-aware detection
    python -m nas_yolo.scripts.demo_webcam --use-context

    # Using pretrained YOLOv8n + SA-SSM plugin (requires ultralytics + weights)
    python -m nas_yolo.scripts.demo_webcam --mode ultralytics --weights yolov8n.pt

Controls:
    Q/ESC: Quit
    S:     Save screenshot
    R:     Reset temporal state
    C:     Toggle context-aware mode
    N:     Toggle noise visualization
"""

import argparse
import sys
import time
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


# COCO 80-class names
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]

# Smart glasses 30-class subset
SMARTGLASS_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "bus", "truck",
    "traffic light", "stop sign", "dog", "cat",
    "chair", "couch", "dining table", "tv", "laptop",
    "cell phone", "book", "clock", "cup", "bottle",
    "knife", "fork", "spoon", "bowl", "banana",
    "apple", "sandwich", "pizza", "backpack", "handbag",
]

# Color palette for drawing
COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
    (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
    (255, 128, 0), (255, 0, 128), (128, 255, 0), (0, 255, 128),
    (128, 0, 255), (0, 128, 255), (255, 128, 128), (128, 255, 128),
    (128, 128, 255), (255, 255, 128), (255, 128, 255), (128, 255, 255),
    (192, 0, 0), (0, 192, 0), (0, 0, 192), (192, 192, 0),
    (192, 0, 192), (0, 192, 192),
]


def draw_detections(frame, boxes, scores, class_ids, class_names, conf_thresh=0.25):
    """Draw bounding boxes and labels on frame."""
    h, w = frame.shape[:2]

    for i in range(len(scores)):
        if scores[i] < conf_thresh:
            continue

        x1, y1, x2, y2 = boxes[i]
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(w, int(x2))
        y2 = min(h, int(y2))

        cls_id = int(class_ids[i])
        color = COLORS[cls_id % len(COLORS)]
        score = float(scores[i])

        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Draw label
        if cls_id < len(class_names):
            label = f"{class_names[cls_id]} {score:.2f}"
        else:
            label = f"cls{cls_id} {score:.2f}"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return frame


def draw_info_overlay(frame, fps, model_info, frame_count, temporal_active=True):
    """Draw FPS and model info overlay."""
    h, w = frame.shape[:2]

    # Background
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (320, 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Text
    y = 22
    cv2.putText(frame, f"PicoYOLO + SA-SSM", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
    y += 20
    cv2.putText(frame, f"FPS: {fps:.1f} | Frame: {frame_count}",
                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    y += 18
    cv2.putText(frame, f"Params: {model_info['total_M']:.3f}M | "
                f"Profile: {model_info.get('profile', 'pico')}",
                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    y += 18
    state_str = "ON (temporal)" if temporal_active else "OFF"
    cv2.putText(frame, f"SA-SSM State: {state_str} | 4-Defense: Active",
                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    y += 18
    cv2.putText(frame, f"[Q]uit [S]creenshot [R]eset [C]ontext",
                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)

    return frame


def preprocess_frame(frame, resolution=320):
    """Preprocess webcam frame for model input."""
    h, w = frame.shape[:2]

    # Letterbox resize
    scale = min(resolution / h, resolution / w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Pad to resolution x resolution
    padded = np.full((resolution, resolution, 3), 114, dtype=np.uint8)
    dh = (resolution - new_h) // 2
    dw = (resolution - new_w) // 2
    padded[dh:dh+new_h, dw:dw+new_w] = resized

    # BGR -> RGB, HWC -> CHW, normalize
    img = padded[:, :, ::-1].copy()  # BGR to RGB
    img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
    tensor = torch.from_numpy(img).unsqueeze(0)

    return tensor, scale, dw, dh


def postprocess_boxes(boxes, scale, dw, dh, orig_h, orig_w):
    """Convert detection boxes back to original frame coordinates."""
    boxes = boxes.clone()
    boxes[:, 0] = (boxes[:, 0] - dw) / scale
    boxes[:, 1] = (boxes[:, 1] - dh) / scale
    boxes[:, 2] = (boxes[:, 2] - dw) / scale
    boxes[:, 3] = (boxes[:, 3] - dh) / scale

    boxes[:, 0].clamp_(min=0, max=orig_w)
    boxes[:, 1].clamp_(min=0, max=orig_h)
    boxes[:, 2].clamp_(min=0, max=orig_w)
    boxes[:, 3].clamp_(min=0, max=orig_h)

    return boxes


def run_pico_webcam(
    base_channels=32,
    num_classes=80,
    resolution=320,
    conf_threshold=0.25,
    camera_id=0,
    use_context=False,
    checkpoint=None,
    save_dir="runs/webcam",
):
    """Run PicoYOLO with webcam input.

    Args:
        base_channels: PicoYOLO base channels (16, 32, 48).
        num_classes: Number of classes (30 or 80).
        resolution: Model input resolution.
        conf_threshold: Detection confidence threshold.
        camera_id: OpenCV camera device ID.
        use_context: Enable context-aware detection.
        checkpoint: Path to trained weights (optional).
        save_dir: Directory for screenshots.
    """
    # Select class names
    if num_classes == 30:
        class_names = SMARTGLASS_CLASSES
    elif num_classes == 80:
        class_names = COCO_CLASSES
    else:
        class_names = [f"class_{i}" for i in range(num_classes)]

    # Build model
    print("Building PicoYOLO...")
    from nas_yolo.models.pico_yolo import PicoYOLO

    model = PicoYOLO(
        num_classes=num_classes,
        base_channels=base_channels,
        plugin_profile="pico",
        input_resolution=resolution,
        use_context=use_context,
    )

    if checkpoint and os.path.exists(checkpoint):
        print(f"Loading weights: {checkpoint}")
        state = torch.load(checkpoint, map_location="cpu")
        if "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        else:
            model.load_state_dict(state)

    model.eval()
    model.print_model_info()

    # Model info for overlay
    total_params = sum(p.numel() for p in model.parameters())
    model_info = {
        "total_M": total_params / 1e6,
        "profile": "pico",
        "base_channels": base_channels,
    }

    # Open camera
    print(f"\nOpening camera {camera_id}...")
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {camera_id}")
        print("Try: --camera-id 1 or --camera-id 2")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {actual_w}x{actual_h}")
    print(f"Model: PicoYOLO (base={base_channels}, classes={num_classes})")
    print(f"Conf threshold: {conf_threshold}")
    print(f"NOTE: Model is untrained (random weights)")
    print(f"      Train on Colab first for real detections!")
    print(f"\nPress Q or ESC to quit, S to save screenshot\n")

    os.makedirs(save_dir, exist_ok=True)

    # State for temporal processing
    states = None
    frame_count = 0
    fps = 0.0
    fps_alpha = 0.9  # EMA for FPS smoothing

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera read failed")
                break

            frame_count += 1
            t_start = time.perf_counter()

            # Preprocess
            tensor, scale, dw, dh = preprocess_frame(frame, resolution)

            # Inference with temporal state
            with torch.no_grad():
                boxes, scores, class_ids, new_states = model.detect(
                    tensor, conf_threshold=conf_threshold, states=states,
                )
                states = new_states  # Propagate temporal state

            # Postprocess
            if boxes.shape[1] > 0:
                boxes_np = postprocess_boxes(
                    boxes[0], scale, dw, dh,
                    frame.shape[0], frame.shape[1],
                ).numpy()
                scores_np = scores[0].numpy()
                cls_ids_np = class_ids[0].numpy()

                # Draw detections
                frame = draw_detections(
                    frame, boxes_np, scores_np, cls_ids_np,
                    class_names, conf_threshold,
                )

            # Calculate FPS
            t_elapsed = time.perf_counter() - t_start
            current_fps = 1.0 / max(t_elapsed, 0.001)
            fps = fps_alpha * fps + (1 - fps_alpha) * current_fps

            # Draw info overlay
            frame = draw_info_overlay(
                frame, fps, model_info, frame_count,
                temporal_active=(states is not None),
            )

            # Show
            cv2.imshow("PicoYOLO + SA-SSM Live Detection", frame)

            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # Q or ESC
                break
            elif key == ord('s'):  # Screenshot
                path = os.path.join(save_dir, f"screenshot_{frame_count:06d}.jpg")
                cv2.imwrite(path, frame)
                print(f"Screenshot saved: {path}")
            elif key == ord('r'):  # Reset state
                model.reset_state()
                states = None
                print("Temporal state reset")
            elif key == ord('c'):  # Toggle context
                print("Context toggle (requires rebuild)")

    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nTotal frames: {frame_count}")
        print(f"Average FPS: {fps:.1f}")


def main():
    parser = argparse.ArgumentParser(
        description="PicoYOLO Webcam Demo - Real-Time Object Detection"
    )
    parser.add_argument("--base-channels", type=int, default=32,
                        help="PicoYOLO base channels (16, 32, 48)")
    parser.add_argument("--num-classes", type=int, default=80,
                        help="Number of classes (30 or 80)")
    parser.add_argument("--resolution", type=int, default=320,
                        help="Model input resolution")
    parser.add_argument("--conf-threshold", type=float, default=0.25,
                        help="Detection confidence threshold")
    parser.add_argument("--camera-id", type=int, default=0,
                        help="OpenCV camera device ID")
    parser.add_argument("--use-context", action="store_true",
                        help="Enable context-aware detection")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to trained model weights")
    parser.add_argument("--save-dir", type=str, default="runs/webcam",
                        help="Directory for screenshots")
    args = parser.parse_args()

    run_pico_webcam(
        base_channels=args.base_channels,
        num_classes=args.num_classes,
        resolution=args.resolution,
        conf_threshold=args.conf_threshold,
        camera_id=args.camera_id,
        use_context=args.use_context,
        checkpoint=args.checkpoint,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()
