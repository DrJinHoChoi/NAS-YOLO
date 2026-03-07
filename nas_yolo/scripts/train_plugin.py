#!/usr/bin/env python3
"""
NAS Plugin Training Script (GPU-Optimized)
=============================================
Two-stage training for YOLO + NASPlugin with full GPU optimization.

Features:
    - AMP (mixed precision) with GradScaler
    - Gradient accumulation for effective large batch sizes
    - Linear warmup + cosine annealing LR scheduler
    - Gradient clipping (max norm)
    - Exponential Moving Average (EMA) of model weights
    - Multi-GPU DDP via torchrun
    - Knowledge Distillation with noise conditioning
    - Gradient checkpointing (memory optimization)

Usage:
    # Single-GPU training
    python -m nas_yolo.scripts.train_plugin \\
        --config nas_yolo/configs/yolo_variants/yolov8n_plugin.yaml

    # Multi-GPU training (DDP)
    torchrun --nproc_per_node=2 -m nas_yolo.scripts.train_plugin \\
        --config nas_yolo/configs/yolo_variants/yolov8n_plugin.yaml

    # Quick smoke test
    python -m nas_yolo.scripts.train_plugin \\
        --config nas_yolo/configs/yolo_variants/yolov8n_plugin.yaml \\
        --epochs 2 --subset 100

Two-stage training:
    Stage 1: Freeze backbone+neck, train only NASPlugin (stage1_epochs)
    Stage 2: Unfreeze all, fine-tune with lower LR + optional KD (stage2_epochs)
"""

import os
import sys
import math
import argparse
import yaml
import time
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [NAS-Plugin] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Learning Rate Scheduler
# =============================================================================

class LinearWarmupCosineAnnealing(optim.lr_scheduler._LRScheduler):
    """Linear warmup followed by cosine annealing LR schedule.

    During warmup (epochs 0..warmup_epochs-1), LR linearly increases from
    warmup_start_lr to base_lr. After warmup, cosine annealing decays LR
    from base_lr to eta_min over the remaining epochs.

    Args:
        optimizer: Wrapped optimizer.
        warmup_epochs: Number of warmup epochs.
        total_epochs: Total number of training epochs.
        warmup_start_lr: Starting LR for warmup (default: 1e-6).
        eta_min: Minimum LR for cosine annealing (default: 0).
        last_epoch: Index of last epoch (default: -1).
    """

    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int,
                 warmup_start_lr: float = 1e-6, eta_min: float = 0,
                 last_epoch: int = -1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = self.last_epoch / max(self.warmup_epochs, 1)
            return [
                self.warmup_start_lr + (base_lr - self.warmup_start_lr) * alpha
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_epochs) / max(
                self.total_epochs - self.warmup_epochs, 1
            )
            return [
                self.eta_min + (base_lr - self.eta_min) *
                0.5 * (1.0 + math.cos(math.pi * progress))
                for base_lr in self.base_lrs
            ]


# =============================================================================
# DDP Utilities
# =============================================================================

def setup_ddp(local_rank: int):
    """Initialize distributed training."""
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_world_size()


def cleanup_ddp():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def unwrap_model(model):
    """Unwrap DDP model if needed."""
    if isinstance(model, DDP):
        return model.module
    return model


# =============================================================================
# CLI Argument Parsing
# =============================================================================

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

    # DDP (set automatically by torchrun)
    parser.add_argument("--local-rank", type=int, default=-1,
                        help="Local rank for DDP (set by torchrun)")

    # Knowledge Distillation arguments
    parser.add_argument("--kd-teacher", type=str, default=None,
                        help="Teacher model weights for KD (e.g., yolov8n.pt)")
    parser.add_argument("--kd-alpha-feature", type=float, default=None,
                        help="Feature-level KD loss weight")
    parser.add_argument("--kd-alpha-output", type=float, default=None,
                        help="Output-level KD loss weight")
    parser.add_argument("--kd-temperature", type=float, default=None,
                        help="KD temperature (higher = softer)")
    parser.add_argument("--kd-weight", type=float, default=None,
                        help="Global KD loss weight multiplier")
    parser.add_argument("--no-kd", action="store_true",
                        help="Disable KD even if config enables it")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


# =============================================================================
# Model Building
# =============================================================================

def build_plugin_model(config: dict, device: str, use_kd: bool = False):
    """Build YOLO + NASPlugin model from config.

    Args:
        config: Full configuration dict.
        device: Device string ('cuda', 'cpu').
        use_kd: Whether to build KD-enabled model.
    """
    model_cfg = config.get("model", {})
    plugin_cfg = config.get("plugin", {})
    kd_cfg = config.get("kd", {})
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
    elif use_kd:
        from nas_yolo.integrations import UltralyticsWithKD
        teacher_weights = kd_cfg.get("teacher_weights", model_cfg.get("pretrained_weights", "yolov8n.pt"))
        model = UltralyticsWithKD(
            weights_path=model_cfg.get("pretrained_weights", "yolov8n.pt"),
            teacher_weights=teacher_weights,
            plugin_cfg=plugin_cfg,
            kd_cfg=kd_cfg,
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


# =============================================================================
# Training Loop
# =============================================================================

def train_one_epoch(model, loader, optimizer, scaler, device, epoch, amp_enabled,
                    use_kd=False, grad_accum_steps=1, max_grad_norm=10.0,
                    ema=None):
    """Train for one epoch with gradient accumulation and clipping.

    Args:
        model: The model to train.
        loader: Data loader.
        optimizer: Optimizer.
        scaler: GradScaler for AMP.
        device: Device string.
        epoch: Current epoch number.
        amp_enabled: Whether to use AMP.
        use_kd: Whether KD loss is active.
        grad_accum_steps: Number of gradient accumulation steps.
        max_grad_norm: Maximum gradient norm for clipping (0 = no clip).
        ema: ModelEMA instance or None.

    Returns:
        Average loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    total_kd_loss = 0.0
    num_batches = 0

    optimizer.zero_grad(set_to_none=True)

    for batch_idx, batch in enumerate(loader):
        images = batch["images"].to(device, non_blocking=True)
        targets = batch.get("targets")
        if targets is not None:
            targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]

        # Forward pass
        if amp_enabled:
            with autocast():
                outputs = model(images, targets)
                loss = _extract_loss(outputs, device, use_kd)
                loss = loss / grad_accum_steps  # Scale loss for accumulation
        else:
            outputs = model(images, targets)
            loss = _extract_loss(outputs, device, use_kd)
            loss = loss / grad_accum_steps

        # Backward pass
        if amp_enabled:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Optimizer step every grad_accum_steps (or at end of epoch)
        is_accum_step = (batch_idx + 1) % grad_accum_steps == 0
        is_last_batch = (batch_idx + 1) == len(loader)

        if is_accum_step or is_last_batch:
            if amp_enabled:
                if max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                if max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)

            # EMA update after each optimizer step
            if ema is not None:
                ema.update(unwrap_model(model))

        # Logging (unscale loss for display)
        total_loss += loss.item() * grad_accum_steps
        if use_kd and isinstance(outputs, dict) and "kd_loss" in outputs:
            total_kd_loss += outputs["kd_loss"].item()
        num_batches += 1

        if batch_idx % 50 == 0:
            msg = (f"Epoch {epoch} | Batch {batch_idx}/{len(loader)} | "
                   f"Loss: {loss.item() * grad_accum_steps:.4f}")
            if use_kd and isinstance(outputs, dict) and "kd_losses" in outputs:
                rel = outputs["kd_losses"].get("teacher_reliability", None)
                if rel is not None:
                    msg += f" | Reliability: {rel.item():.3f}"
            logger.info(msg)

    avg_loss = total_loss / max(num_batches, 1)
    avg_kd = total_kd_loss / max(num_batches, 1) if use_kd else 0.0
    if use_kd:
        logger.info(f"  Avg KD loss: {avg_kd:.4f}")
    return avg_loss


def _extract_loss(outputs, device, use_kd=False):
    """Extract total loss from model outputs, including KD if active."""
    loss = torch.tensor(0.0, device=device)

    if isinstance(outputs, dict):
        if "losses" in outputs:
            loss = outputs["losses"]["loss"]
        # Add KD loss (only during training, Stage 2)
        if use_kd and "kd_loss" in outputs:
            loss = loss + outputs["kd_loss"]
    return loss


# =============================================================================
# Main Training Entrypoint
# =============================================================================

def main():
    args = parse_args()
    config = load_config(args.config)

    # --- DDP setup ---
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    is_distributed = local_rank >= 0
    if is_distributed:
        rank, world_size = setup_ddp(local_rank)
        device = f"cuda:{rank}"
        is_main = (rank == 0)
    else:
        rank = -1
        world_size = 1
        device = args.device
        is_main = True

    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
        if is_main:
            logger.warning("CUDA not available, falling back to CPU")

    # --- Apply CLI overrides ---
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

    # KD CLI overrides
    if args.kd_teacher:
        config.setdefault("kd", {})["teacher_weights"] = args.kd_teacher
        config["kd"]["enabled"] = True
    if args.kd_alpha_feature is not None:
        config.setdefault("kd", {})["alpha_feature"] = args.kd_alpha_feature
    if args.kd_alpha_output is not None:
        config.setdefault("kd", {})["alpha_output"] = args.kd_alpha_output
    if args.kd_temperature is not None:
        config.setdefault("kd", {})["temperature"] = args.kd_temperature
    if args.kd_weight is not None:
        config.setdefault("kd", {})["kd_weight"] = args.kd_weight
    if args.no_kd:
        config.setdefault("kd", {})["enabled"] = False

    # --- Read training config ---
    use_kd = config.get("kd", {}).get("enabled", False)
    train_cfg = config.get("training", {})
    stage1_epochs = train_cfg.get("stage1_epochs", 15)
    stage2_epochs = train_cfg.get("stage2_epochs", 85)

    if args.epochs:
        total = args.epochs
        stage1_epochs = max(1, total * 15 // 100)
        stage2_epochs = total - stage1_epochs

    total_epochs = stage1_epochs + stage2_epochs

    # Optimization parameters
    batch_size = train_cfg.get("batch_size", 16)
    base_lr = train_cfg.get("base_lr", 0.01)
    weight_decay = train_cfg.get("weight_decay", 0.0005)
    warmup_epochs = train_cfg.get("warmup_epochs", 3)
    grad_accum_steps = train_cfg.get("grad_accum_steps", 1)
    max_grad_norm = train_cfg.get("clip_grad_norm", 10.0)
    amp_enabled = train_cfg.get("amp", True) and device != "cpu"
    ema_enabled = train_cfg.get("ema", True)
    ema_decay = train_cfg.get("ema_decay", 0.9999)

    # --- Build model ---
    if is_main:
        logger.info(f"Building model from {args.config}")
        if use_kd:
            logger.info("Knowledge Distillation ENABLED")
            logger.info(f"  Teacher: {config.get('kd', {}).get('teacher_weights', 'same as student')}")
    model = build_plugin_model(config, device, use_kd=use_kd)

    # Print model info
    if is_main and hasattr(model, "get_model_info"):
        info = model.get_model_info()
        logger.info(f"Base model: {info.get('base_model', 'unknown')}")
        logger.info(f"Base params: {info.get('base_params_M', 0):.2f}M")
        logger.info(f"Plugin params: {info.get('plugin_params_M', 0):.4f}M")
        logger.info(f"Total params: {info.get('total_params_M', 0):.2f}M")
        logger.info(f"Plugin overhead: {info.get('overhead_pct', 0):.1f}%")

    # --- EMA ---
    ema = None
    if ema_enabled and device != "cpu":
        from nas_yolo.utils.ema import ModelEMA
        ema = ModelEMA(model, decay=ema_decay)
        if is_main:
            logger.info(f"EMA enabled (decay={ema_decay})")

    # --- DDP wrap ---
    if is_distributed:
        model = DDP(model, device_ids=[rank], output_device=rank,
                    find_unused_parameters=True)
        if is_main:
            logger.info(f"DDP enabled: {world_size} GPUs")

    # --- Log training config ---
    if is_main:
        logger.info(f"Training config:")
        logger.info(f"  Stage 1: {stage1_epochs} epochs (backbone frozen)")
        logger.info(f"  Stage 2: {stage2_epochs} epochs (all unfrozen)")
        logger.info(f"  Batch size: {batch_size} (× {grad_accum_steps} accum = {batch_size * grad_accum_steps} effective)")
        logger.info(f"  AMP: {amp_enabled}")
        logger.info(f"  Warmup: {warmup_epochs} epochs")
        logger.info(f"  Grad clip: {max_grad_norm}")
        logger.info(f"  Device: {device}")

    # --- Build dataset ---
    train_sampler = None
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
            if is_main:
                logger.info(f"Using subset of {len(indices)} images")

        # Distributed sampler for multi-GPU
        if is_distributed:
            train_sampler = DistributedSampler(
                train_dataset, num_replicas=world_size, rank=rank,
            )

        train_loader = build_dataloader(
            train_dataset, batch_size=batch_size,
            num_workers=data_cfg.get("num_workers", 4),
            sampler=train_sampler,
        )
    except (ImportError, FileNotFoundError) as e:
        logger.error(f"Dataset setup failed: {e}")
        logger.error("Please ensure COCO data is available.")
        if is_distributed:
            cleanup_ddp()
        return

    # --- Optimizer and scheduler ---
    scaler = GradScaler(enabled=amp_enabled)
    output_dir = config.get("output", {}).get("dir", "runs/plugin_train")
    if is_main:
        os.makedirs(output_dir, exist_ok=True)

    # ==========================================
    # Stage 1: Freeze backbone+neck, train plugin
    # ==========================================
    if is_main:
        logger.info(f"\n{'='*60}")
        logger.info(f"Stage 1: Training NASPlugin only ({stage1_epochs} epochs)")
        logger.info(f"{'='*60}")

    raw_model = unwrap_model(model)
    if hasattr(raw_model, "freeze_backbone"):
        raw_model.freeze_backbone()

    # Only optimize plugin parameters
    plugin_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(plugin_params, lr=base_lr, weight_decay=weight_decay)
    scheduler = LinearWarmupCosineAnnealing(
        optimizer,
        warmup_epochs=min(warmup_epochs, stage1_epochs),
        total_epochs=stage1_epochs,
        warmup_start_lr=base_lr * 0.01,
        eta_min=base_lr * 0.1,
    )

    for epoch in range(1, stage1_epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        t0 = time.time()
        avg_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, device, epoch, amp_enabled,
            use_kd=False,
            grad_accum_steps=grad_accum_steps,
            max_grad_norm=max_grad_norm,
            ema=ema,
        )
        scheduler.step()
        elapsed = time.time() - t0

        if is_main:
            logger.info(
                f"Epoch {epoch}/{stage1_epochs} | "
                f"Loss: {avg_loss:.4f} | "
                f"LR: {scheduler.get_last_lr()[0]:.6f} | "
                f"Time: {elapsed:.0f}s"
            )

    # Save stage 1 checkpoint
    if is_main:
        ckpt = {
            "epoch": stage1_epochs,
            "model_state_dict": raw_model.state_dict(),
            "stage": 1,
            "config": config,
        }
        if ema is not None:
            ckpt["ema_state"] = ema.state_dict()
        torch.save(ckpt, os.path.join(output_dir, "stage1.pt"))
        logger.info(f"Stage 1 checkpoint saved to {output_dir}/stage1.pt")

    # ==========================================
    # Stage 2: Unfreeze all, fine-tune
    # ==========================================
    if is_main:
        logger.info(f"\n{'='*60}")
        logger.info(f"Stage 2: Fine-tuning all parameters ({stage2_epochs} epochs)")
        logger.info(f"{'='*60}")

    if hasattr(raw_model, "unfreeze_all"):
        raw_model.unfreeze_all()

    fine_tune_lr = base_lr * 0.1
    optimizer = optim.AdamW(
        [
            {"params": raw_model.nas_plugin.parameters(), "lr": fine_tune_lr},
            {"params": [p for n, p in raw_model.named_parameters()
                        if "nas_plugin" not in n and p.requires_grad],
             "lr": fine_tune_lr * 0.1},
        ],
        weight_decay=weight_decay,
    )
    scheduler = LinearWarmupCosineAnnealing(
        optimizer,
        warmup_epochs=min(warmup_epochs, stage2_epochs),
        total_epochs=stage2_epochs,
        warmup_start_lr=fine_tune_lr * 0.01,
        eta_min=fine_tune_lr * 0.01,
    )

    for epoch in range(1, stage2_epochs + 1):
        global_epoch = stage1_epochs + epoch
        if train_sampler is not None:
            train_sampler.set_epoch(global_epoch)

        t0 = time.time()
        avg_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, device, global_epoch, amp_enabled,
            use_kd=use_kd,
            grad_accum_steps=grad_accum_steps,
            max_grad_norm=max_grad_norm,
            ema=ema,
        )
        scheduler.step()
        elapsed = time.time() - t0

        if is_main:
            logger.info(
                f"Epoch {global_epoch}/{total_epochs} | "
                f"Loss: {avg_loss:.4f} | "
                f"LR: {scheduler.get_last_lr()[0]:.6f} | "
                f"Time: {elapsed:.0f}s"
            )

            # Save checkpoint every 10 epochs
            if epoch % 10 == 0 or epoch == stage2_epochs:
                ckpt = {
                    "epoch": global_epoch,
                    "model_state_dict": raw_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "stage": 2,
                    "config": config,
                }
                if ema is not None:
                    ckpt["ema_state"] = ema.state_dict()
                torch.save(ckpt, os.path.join(output_dir, f"epoch_{global_epoch}.pt"))

    # Save final model
    if is_main:
        final_ckpt = {
            "epoch": total_epochs,
            "model_state_dict": raw_model.state_dict(),
            "stage": "final",
            "config": config,
        }
        if ema is not None:
            final_ckpt["ema_state"] = ema.state_dict()
            # Also save EMA model separately (for evaluation)
            torch.save({
                "model_state_dict": ema.ema_model.state_dict(),
                "config": config,
            }, os.path.join(output_dir, "best_ema.pt"))
            logger.info(f"EMA model saved to {output_dir}/best_ema.pt")

        torch.save(final_ckpt, os.path.join(output_dir, "best.pt"))
        logger.info(f"Final model saved to {output_dir}/best.pt")
        logger.info("Training complete!")

    # Cleanup
    if is_distributed:
        cleanup_ddp()


if __name__ == "__main__":
    main()
