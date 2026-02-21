#!/usr/bin/env python3
"""
NAS Plugin Training Script
============================
Two-stage training for YOLO + NASPlugin.

Usage:
    # Train YOLOv8-n + NASPlugin (edge profile)
    python -m nas_yolo.scripts.train_plugin \\
        --config nas_yolo/configs/yolo_variants/yolov8n_plugin.yaml

    # Train YOLOv8-s + NASPlugin (standard profile)
    python -m nas_yolo.scripts.train_plugin \\
        --config nas_yolo/configs/yolo_variants/yolov8s_plugin.yaml \\
        --profile standard

    # Ablation: Mamba-only
    python -m nas_yolo.scripts.train_plugin \\
        --config nas_yolo/configs/yolo_variants/yolov8s_plugin.yaml \\
        --mode mamba

    # Quick test (2 epochs, 100 images)
    python -m nas_yolo.scripts.train_plugin \\
        --config nas_yolo/configs/yolo_variants/yolov8n_plugin.yaml \\
        --epochs 2 --subset 100

Two-stage training:
    Stage 1: Freeze backbone+neck, train only NASPlugin (stage1_epochs)
    Stage 2: Unfreeze all, fine-tune with lower LR (stage2_epochs)
"""

import os
import sys
import argparse
import yaml
import time
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [NAS-Plugin] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="NAS Plugin Training")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YOLO variant config YAML")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--profile", type=str, default=None,
                        help="Override plugin profile (standard/lite/edge/ultra-lite)")
    parser.add_argument("--mode", type=str, default=None,
                        help="Override plugin mode (hybrid/mamba/attention)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override total epochs")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--subset", type=int, default=None,
                        help="Use only N images for quick testing")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint")
    parser.add_argument("--data-root", type=str, default=None,
                        help="COCO data root directory")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def build_plugin_model(config: dict, device: str):
    """Build YOLO + NASPlugin model from config."""
    model_cfg = config.get("model", {})
    plugin_cfg = config.get("plugin", {})
    framework = model_cfg.get("framework", "ultralytics")

    if framework == "yolov9":
        from nas_yolo.integrations import YOLOv9WithNASPlugin
        model = YOLOv9WithNASPlugin(
            cfg=model_cfg.get("cfg", "yolov9-t.yaml"),
            weights=model_cfg.get("pretrained_weights"),
            plugin_cfg=plugin_cfg,
            variant=model_cfg.get("yolo_variant", "yolov9-t"),
            freeze_backbone=True,
        )
    else:
        from nas_yolo.integrations import UltralyticsWithNASPlugin
        model = UltralyticsWithNASPlugin(
            weights_path=model_cfg.get("pretrained_weights", "yolov8n.pt"),
            plugin_cfg=plugin_cfg,
            freeze_backbone=True,
        )

    return model.to(device)


def train_one_epoch(model, loader, optimizer, scaler, device, epoch, amp_enabled):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(loader):
        images = batch["images"].to(device)
        targets = batch.get("targets")
        if targets is not None:
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()

        if amp_enabled:
            with autocast():
                outputs = model(images, targets)
                if isinstance(outputs, dict) and "losses" in outputs:
                    loss = outputs["losses"]["loss"]
                else:
                    loss = torch.tensor(0.0, device=device)
        else:
            outputs = model(images, targets)
            if isinstance(outputs, dict) and "losses" in outputs:
                loss = outputs["losses"]["loss"]
            else:
                loss = torch.tensor(0.0, device=device)

        if amp_enabled:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if batch_idx % 50 == 0:
            logger.info(
                f"Epoch {epoch} | Batch {batch_idx}/{len(loader)} | "
                f"Loss: {loss.item():.4f}"
            )

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


def main():
    args = parse_args()
    config = load_config(args.config)

    # Apply CLI overrides
    if args.profile:
        config.setdefault("plugin", {})["profile"] = args.profile
    if args.mode:
        config.setdefault("plugin", {})["mode"] = args.mode
    if args.batch_size:
        config.setdefault("training", {})["batch_size"] = args.batch_size
    if args.lr:
        config.setdefault("training", {})["base_lr"] = args.lr
    if args.output_dir:
        config.setdefault("output", {})["dir"] = args.output_dir

    train_cfg = config.get("training", {})
    stage1_epochs = train_cfg.get("stage1_epochs", 15)
    stage2_epochs = train_cfg.get("stage2_epochs", 85)

    if args.epochs:
        # Override: split proportionally
        total = args.epochs
        stage1_epochs = max(1, total * 15 // 100)
        stage2_epochs = total - stage1_epochs

    total_epochs = stage1_epochs + stage2_epochs

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        logger.warning("CUDA not available, falling back to CPU")

    # Build model
    logger.info(f"Building model from {args.config}")
    model = build_plugin_model(config, device)

    # Print model info
    if hasattr(model, "get_model_info"):
        info = model.get_model_info()
        logger.info(f"Base model: {info.get('base_model', 'unknown')}")
        logger.info(f"Base params: {info.get('base_params_M', 0):.2f}M")
        logger.info(f"Plugin params: {info.get('plugin_params_M', 0):.4f}M")
        logger.info(f"Total params: {info.get('total_params_M', 0):.2f}M")
        logger.info(f"Plugin overhead: {info.get('overhead_pct', 0):.1f}%")

    # Build data loaders
    # For now, use a simple placeholder - users should provide COCO data
    batch_size = train_cfg.get("batch_size", 16)
    amp_enabled = train_cfg.get("amp", True) and device != "cpu"

    logger.info(f"Training config:")
    logger.info(f"  Stage 1: {stage1_epochs} epochs (backbone frozen)")
    logger.info(f"  Stage 2: {stage2_epochs} epochs (all unfrozen)")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  AMP: {amp_enabled}")
    logger.info(f"  Device: {device}")

    # Build dataset
    try:
        from nas_yolo.data.dataset import COCODetectionDataset, build_dataloader
        from nas_yolo.data.transforms import DetectionTransform
        from nas_yolo.data.corruption import CorruptionTransform

        data_cfg = config.get("data", {})
        aug_cfg = config.get("augmentation", {})
        data_root = args.data_root or data_cfg.get("train", "data/coco")
        img_size = data_cfg.get("img_size", 640)

        train_transform = DetectionTransform(
            img_size=(img_size, img_size), augment=True
        )
        corruption = None
        if aug_cfg.get("corruption_aug"):
            corruption = CorruptionTransform(
                corruption_type="random", severity=3,
                p=aug_cfg.get("corruption_p", 0.3),
            )

        train_dataset = COCODetectionDataset(
            root_dir=os.path.join(data_root, "train2017"),
            ann_file=os.path.join(data_root, "annotations/instances_train2017.json"),
            transform=train_transform,
            corruption=corruption,
            img_size=(img_size, img_size),
        )

        if args.subset:
            from torch.utils.data import Subset
            indices = list(range(min(args.subset, len(train_dataset))))
            train_dataset = Subset(train_dataset, indices)
            logger.info(f"Using subset of {len(indices)} images")

        train_loader = build_dataloader(
            train_dataset, batch_size=batch_size,
            num_workers=data_cfg.get("num_workers", 4),
        )
    except ImportError as e:
        logger.error(f"Dataset import failed: {e}")
        logger.error("Please ensure COCO data is available.")
        return

    # Optimizer and scheduler
    base_lr = train_cfg.get("base_lr", 0.01)
    weight_decay = train_cfg.get("weight_decay", 0.0005)
    scaler = GradScaler(enabled=amp_enabled)

    output_dir = config.get("output", {}).get("dir", "runs/plugin_train")
    os.makedirs(output_dir, exist_ok=True)

    # ==========================================
    # Stage 1: Freeze backbone+neck, train plugin
    # ==========================================
    logger.info(f"\n{'='*60}")
    logger.info(f"Stage 1: Training NASPlugin only ({stage1_epochs} epochs)")
    logger.info(f"{'='*60}")

    if hasattr(model, "freeze_backbone"):
        model.freeze_backbone()

    # Only optimize plugin parameters
    plugin_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(plugin_params, lr=base_lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=stage1_epochs, eta_min=base_lr * 0.1
    )

    for epoch in range(1, stage1_epochs + 1):
        t0 = time.time()
        avg_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, device, epoch, amp_enabled
        )
        scheduler.step()
        elapsed = time.time() - t0

        logger.info(
            f"Epoch {epoch}/{stage1_epochs} | "
            f"Loss: {avg_loss:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.6f} | "
            f"Time: {elapsed:.0f}s"
        )

    # Save stage 1 checkpoint
    torch.save({
        "epoch": stage1_epochs,
        "model_state_dict": model.state_dict(),
        "stage": 1,
        "config": config,
    }, os.path.join(output_dir, "stage1.pt"))
    logger.info(f"Stage 1 checkpoint saved to {output_dir}/stage1.pt")

    # ==========================================
    # Stage 2: Unfreeze all, fine-tune
    # ==========================================
    logger.info(f"\n{'='*60}")
    logger.info(f"Stage 2: Fine-tuning all parameters ({stage2_epochs} epochs)")
    logger.info(f"{'='*60}")

    if hasattr(model, "unfreeze_all"):
        model.unfreeze_all()

    fine_tune_lr = base_lr * 0.1
    optimizer = optim.AdamW(
        [
            {"params": model.nas_plugin.parameters(), "lr": fine_tune_lr},
            {"params": [p for n, p in model.named_parameters()
                        if "nas_plugin" not in n and p.requires_grad],
             "lr": fine_tune_lr * 0.1},
        ],
        weight_decay=weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=stage2_epochs, eta_min=fine_tune_lr * 0.01
    )

    for epoch in range(1, stage2_epochs + 1):
        global_epoch = stage1_epochs + epoch
        t0 = time.time()
        avg_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, device, global_epoch, amp_enabled
        )
        scheduler.step()
        elapsed = time.time() - t0

        logger.info(
            f"Epoch {global_epoch}/{total_epochs} | "
            f"Loss: {avg_loss:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.6f} | "
            f"Time: {elapsed:.0f}s"
        )

        # Save checkpoint every 10 epochs
        if epoch % 10 == 0 or epoch == stage2_epochs:
            torch.save({
                "epoch": global_epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "stage": 2,
                "config": config,
            }, os.path.join(output_dir, f"epoch_{global_epoch}.pt"))

    # Save final model
    torch.save({
        "epoch": total_epochs,
        "model_state_dict": model.state_dict(),
        "stage": "final",
        "config": config,
    }, os.path.join(output_dir, "best.pt"))
    logger.info(f"Final model saved to {output_dir}/best.pt")
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
