#!/usr/bin/env python3
"""
YOLOv8n vs YOLOv8n+SA-SSM Noise Robustness Comparison Demo
=============================================================
Side-by-side webcam demo comparing vanilla YOLOv8n against
YOLOv8n enhanced with SA-SSM NASPlugin.

Shows how SA-SSM's 4 structural noise defenses maintain detection
accuracy under noisy conditions where vanilla YOLO degrades.

Usage:
    python -m nas_yolo.scripts.demo_comparison
    python -m nas_yolo.scripts.demo_comparison --profile edge
    python -m nas_yolo.scripts.demo_comparison --noise-level 50

Controls:
    Q/ESC:  Quit
    N:      Cycle noise type (none -> gaussian -> salt&pepper -> blur -> all)
    +/-:    Increase/decrease noise level
    S:      Save screenshot
    R:      Reset SA-SSM temporal state
    SPACE:  Pause/resume
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


# --- Noise injection functions ---

def add_gaussian_noise(frame, sigma=30):
    """Add Gaussian noise to frame."""
    noise = np.random.normal(0, sigma, frame.shape).astype(np.float32)
    noisy = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy


def add_salt_pepper_noise(frame, amount=0.05):
    """Add salt and pepper noise."""
    noisy = frame.copy()
    h, w, c = noisy.shape
    num_salt = int(amount * h * w)
    num_pepper = int(amount * h * w)
    coords_y = np.random.randint(0, h, num_salt)
    coords_x = np.random.randint(0, w, num_salt)
    noisy[coords_y, coords_x] = 255
    coords_y = np.random.randint(0, h, num_pepper)
    coords_x = np.random.randint(0, w, num_pepper)
    noisy[coords_y, coords_x] = 0
    return noisy


def add_blur_noise(frame, ksize=11):
    """Add Gaussian blur."""
    return cv2.GaussianBlur(frame, (ksize, ksize), 0)


def add_combined_noise(frame, sigma=20, sp_amount=0.03, blur_k=5):
    """Add combination of all noise types."""
    noisy = add_gaussian_noise(frame, sigma)
    noisy = add_salt_pepper_noise(noisy, sp_amount)
    noisy = add_blur_noise(noisy, blur_k)
    return noisy


NOISE_MODES = ["none", "gaussian", "salt_pepper", "blur", "combined"]
NOISE_LABELS = {
    "none": "Clean (No Noise)",
    "gaussian": "Gaussian Noise",
    "salt_pepper": "Salt & Pepper",
    "blur": "Gaussian Blur",
    "combined": "Combined Noise",
}


def apply_noise(frame, mode, level):
    """Apply noise based on mode and level (0-100)."""
    if mode == "none":
        return frame.copy()
    elif mode == "gaussian":
        return add_gaussian_noise(frame, level * 1.0)
    elif mode == "salt_pepper":
        return add_salt_pepper_noise(frame, level / 500.0)
    elif mode == "blur":
        ksize = max(3, int(level / 5) * 2 + 1)
        return add_blur_noise(frame, ksize)
    elif mode == "combined":
        return add_combined_noise(frame, level * 0.6, level / 800.0,
                                   max(3, int(level / 10) * 2 + 1))
    return frame.copy()


# --- Drawing helpers ---

COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
    (0, 128, 255), (128, 0, 255), (255, 128, 128), (128, 255, 128),
    (128, 128, 255), (255, 255, 128), (255, 128, 255), (128, 255, 255),
]


def draw_boxes(frame, results):
    """Draw ultralytics Results on frame. Returns (frame, count)."""
    if results is None or len(results) == 0:
        return frame, 0
    r = results[0]
    boxes = r.boxes
    count = 0
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = r.names.get(cls_id, f"cls{cls_id}")
            color = COLORS[cls_id % len(COLORS)]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{cls_name} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            count += 1
    return frame, count


def draw_sassm_boxes(frame, raw_output, class_names, conf_threshold=0.3):
    """Decode SA-SSM wrapper raw output and draw boxes.

    The wrapper returns Detect head output: [batch, 84, 8400]
    where 84 = 4 (bbox) + 80 (classes), 8400 = sum of anchor points.
    Format: xywh + class_scores
    """
    if raw_output is None:
        return frame, 0

    # Handle list/tuple output from Detect head
    if isinstance(raw_output, (list, tuple)):
        pred = raw_output[0]
    else:
        pred = raw_output

    if not isinstance(pred, torch.Tensor):
        return frame, 0

    # pred shape: [B, 84, N] → transpose to [B, N, 84]
    if pred.dim() == 3 and pred.shape[1] < pred.shape[2]:
        pred = pred.transpose(1, 2)

    # Take first batch
    pred = pred[0]  # [N, 84]

    # Split: first 4 = xywh, rest = class scores
    boxes_xywh = pred[:, :4]
    class_scores = pred[:, 4:]

    # Max class score and class id
    max_scores, class_ids = class_scores.max(dim=1)

    # Filter by confidence
    mask = max_scores > conf_threshold
    boxes_xywh = boxes_xywh[mask]
    max_scores = max_scores[mask]
    class_ids = class_ids[mask]

    if len(max_scores) == 0:
        return frame, 0

    # xywh to xyxy
    x_c, y_c, w, h_box = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]
    x1 = x_c - w / 2
    y1 = y_c - h_box / 2
    x2 = x_c + w / 2
    y2 = y_c + h_box / 2

    # Scale from model input (640) to frame size
    frame_h, frame_w = frame.shape[:2]
    scale_x = frame_w / 640.0
    scale_y = frame_h / 640.0

    # Simple NMS (greedy)
    indices = torch.argsort(max_scores, descending=True)
    keep = []
    suppressed = set()
    for idx in indices.tolist():
        if idx in suppressed:
            continue
        keep.append(idx)
        # Suppress overlapping boxes
        for jdx in indices.tolist():
            if jdx in suppressed or jdx == idx:
                continue
            # IoU check
            ix1 = max(x1[idx].item(), x1[jdx].item())
            iy1 = max(y1[idx].item(), y1[jdx].item())
            ix2 = min(x2[idx].item(), x2[jdx].item())
            iy2 = min(y2[idx].item(), y2[jdx].item())
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            area_i = (x2[idx] - x1[idx]) * (y2[idx] - y1[idx])
            area_j = (x2[jdx] - x1[jdx]) * (y2[jdx] - y1[jdx])
            union = area_i + area_j - inter
            iou = inter / (union + 1e-6)
            if iou > 0.5:
                suppressed.add(jdx)

    count = 0
    for idx in keep[:50]:  # limit to 50 detections
        bx1 = int(x1[idx].item() * scale_x)
        by1 = int(y1[idx].item() * scale_y)
        bx2 = int(x2[idx].item() * scale_x)
        by2 = int(y2[idx].item() * scale_y)
        bx1 = max(0, min(bx1, frame_w))
        by1 = max(0, min(by1, frame_h))
        bx2 = max(0, min(bx2, frame_w))
        by2 = max(0, min(by2, frame_h))

        cls_id = class_ids[idx].item()
        conf = max_scores[idx].item()
        color = COLORS[cls_id % len(COLORS)]
        cls_name = class_names.get(cls_id, f"cls{cls_id}") if isinstance(class_names, dict) else f"cls{cls_id}"

        cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)
        label = f"{cls_name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(frame, (bx1, by1 - th - 6), (bx1 + tw, by1), color, -1)
        cv2.putText(frame, label, (bx1, by1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        count += 1

    return frame, count


def draw_panel_info(frame, title, fps, det_count, is_sassm=False,
                    noise_mode="none", noise_level=0, params_str=""):
    """Draw info overlay panel."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (290, 105), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    color = (0, 255, 100) if is_sassm else (100, 200, 255)
    y = 22
    cv2.putText(frame, title, (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    y += 18
    cv2.putText(frame, f"FPS: {fps:.1f} | Detections: {det_count}",
                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    y += 16
    cv2.putText(frame, f"Noise: {NOISE_LABELS.get(noise_mode, noise_mode)} (lv={noise_level})",
                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    y += 16
    cv2.putText(frame, params_str, (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 180, 180), 1)
    if is_sassm:
        y += 14
        cv2.putText(frame, "4-Defense: LTI+DCT+Gate+HeteroExpert",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 200, 255), 1)
    return frame


def draw_controls_bar(combined, paused):
    """Draw controls bar at bottom."""
    h, w = combined.shape[:2]
    overlay = combined.copy()
    cv2.rectangle(overlay, (0, h - 28), (w, h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.8, combined, 0.2, 0, combined)
    cv2.putText(combined, "[N]oise [+/-]Level [S]ave [R]eset [SPACE]Pause [Q]uit",
                (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
    status = "PAUSED" if paused else "LIVE"
    cv2.putText(combined, status, (w - 80, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                (0, 0, 255) if paused else (0, 255, 0), 1)
    return combined


def run_comparison(
    profile="edge",
    conf_threshold=0.3,
    camera_id=0,
    noise_level=30,
    save_dir="runs/comparison",
):
    """Run side-by-side comparison: YOLOv8n vs YOLOv8n+SA-SSM."""
    from ultralytics import YOLO

    # --- Load vanilla YOLOv8n ---
    print("Loading YOLOv8n (vanilla)...")
    yolo_vanilla = YOLO("yolov8n.pt")
    vanilla_params = sum(p.numel() for p in yolo_vanilla.model.parameters())
    class_names = yolo_vanilla.model.names  # {0: 'person', 1: 'bicycle', ...}
    print(f"  Params: {vanilla_params:,} ({vanilla_params/1e6:.2f}M)")

    # --- Load YOLOv8n + SA-SSM ---
    print(f"\nLoading YOLOv8n + SA-SSM ({profile})...")
    from nas_yolo.integrations import UltralyticsWithNASPlugin

    sassm_model = UltralyticsWithNASPlugin(
        weights_path="yolov8n.pt",
        plugin_cfg={
            "profile": profile,
            "mode": "hybrid",
            "use_noise_gate": True,
            "noise_dim": 32,
            "pool_size": 8,
        },
        freeze_backbone=False,
    )
    sassm_model.eval()
    sassm_params = sum(p.numel() for p in sassm_model.parameters())
    plugin_params = sum(p.numel() for p in sassm_model.nas_plugin.parameters())
    print(f"  Total: {sassm_params:,} ({sassm_params/1e6:.2f}M)")
    print(f"  Plugin: +{plugin_params:,} ({plugin_params/1e6:.3f}M)")
    overhead_pct = (sassm_params - vanilla_params) / vanilla_params * 100
    print(f"  Overhead: +{overhead_pct:.1f}%")

    vanilla_str = f"Params: {vanilla_params/1e6:.2f}M | No noise defense"
    sassm_str = f"Params: {sassm_params/1e6:.2f}M (+{overhead_pct:.0f}%) | SA-SSM"

    # --- Open camera ---
    print(f"\nOpening camera {camera_id}...")
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {camera_id}")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera: {actual_w}x{actual_h}")
    os.makedirs(save_dir, exist_ok=True)

    # State
    noise_idx = 0
    current_noise = NOISE_MODES[noise_idx]
    current_level = noise_level
    paused = False
    frame_count = 0
    fps_v = fps_s = 0.0
    alpha = 0.85
    last_frame = None

    print(f"\n{'='*60}")
    print(f"  YOLOv8n vs YOLOv8n+SA-SSM  Comparison Demo")
    print(f"  Profile: {profile} | Conf: {conf_threshold}")
    print(f"{'='*60}")
    print(f"  Press N to add noise, +/- to adjust level, Q to quit\n")

    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
                last_frame = frame.copy()
            else:
                if last_frame is None:
                    continue
                frame = last_frame.copy()

            frame_count += 1
            noisy_frame = apply_noise(frame, current_noise, current_level)

            # --- LEFT: Vanilla YOLOv8n ---
            t0 = time.perf_counter()
            results_v = yolo_vanilla.predict(
                noisy_frame, conf=conf_threshold, verbose=False, imgsz=640,
            )
            t_v = time.perf_counter() - t0
            left = noisy_frame.copy()
            left, det_v = draw_boxes(left, results_v)
            fps_v = alpha * fps_v + (1 - alpha) * (1.0 / max(t_v, 0.001))

            # --- RIGHT: YOLOv8n + SA-SSM ---
            t0 = time.perf_counter()
            img_rgb = cv2.cvtColor(noisy_frame, cv2.COLOR_BGR2RGB)
            img_t = torch.from_numpy(img_rgb).float().permute(2, 0, 1) / 255.0
            img_t = F.interpolate(
                img_t.unsqueeze(0), size=(640, 640),
                mode='bilinear', align_corners=False,
            )
            with torch.no_grad():
                sassm_output = sassm_model(img_t)
            t_s = time.perf_counter() - t0
            right = noisy_frame.copy()
            right, det_s = draw_sassm_boxes(
                right, sassm_output, class_names, conf_threshold,
            )
            fps_s = alpha * fps_s + (1 - alpha) * (1.0 / max(t_s, 0.001))

            # --- Draw info ---
            left = draw_panel_info(
                left, "YOLOv8n (vanilla)", fps_v, det_v,
                is_sassm=False, noise_mode=current_noise,
                noise_level=current_level, params_str=vanilla_str,
            )
            right = draw_panel_info(
                right, f"YOLOv8n + SA-SSM ({profile})", fps_s, det_s,
                is_sassm=True, noise_mode=current_noise,
                noise_level=current_level, params_str=sassm_str,
            )

            # --- Combine ---
            divider = np.full((actual_h, 3, 3), (0, 200, 255), dtype=np.uint8)
            combined = np.hstack([left, divider, right])
            combined = draw_controls_bar(combined, paused)
            cv2.imshow("YOLOv8n vs YOLOv8n+SA-SSM | Noise Robustness", combined)

            # --- Keys ---
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('n'):
                noise_idx = (noise_idx + 1) % len(NOISE_MODES)
                current_noise = NOISE_MODES[noise_idx]
                print(f"Noise: {NOISE_LABELS[current_noise]} (level={current_level})")
            elif key in (ord('+'), ord('=')):
                current_level = min(100, current_level + 10)
                print(f"Noise level: {current_level}")
            elif key in (ord('-'), ord('_')):
                current_level = max(0, current_level - 10)
                print(f"Noise level: {current_level}")
            elif key == ord('s'):
                path = os.path.join(save_dir, f"comparison_{frame_count:06d}.jpg")
                cv2.imwrite(path, combined)
                print(f"Screenshot: {path}")
            elif key == ord('r'):
                sassm_model.reset_temporal_state()
                print("SA-SSM temporal state reset")
            elif key == ord(' '):
                paused = not paused
                print("Paused" if paused else "Resumed")

    except KeyboardInterrupt:
        print("\nStopped")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nFrames: {frame_count}")
        print(f"FPS - Vanilla: {fps_v:.1f}, SA-SSM: {fps_s:.1f}")


def main():
    parser = argparse.ArgumentParser(
        description="YOLOv8n vs YOLOv8n+SA-SSM Comparison Demo"
    )
    parser.add_argument("--profile", type=str, default="edge",
                        choices=["standard", "lite", "edge", "ultra-lite",
                                 "tiny", "pico"])
    parser.add_argument("--conf-threshold", type=float, default=0.3)
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--noise-level", type=int, default=30)
    parser.add_argument("--save-dir", type=str, default="runs/comparison")
    args = parser.parse_args()

    run_comparison(
        profile=args.profile,
        conf_threshold=args.conf_threshold,
        camera_id=args.camera_id,
        noise_level=args.noise_level,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()
