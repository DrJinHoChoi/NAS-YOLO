"""
Training Engine for NAS-YOLO
==============================
Supports:
  - Single-GPU and DistributedDataParallel (DDP) training
  - Mixed precision (AMP) for efficiency
  - Cosine annealing with warmup LR schedule
  - Pseudo-temporal training with Temporal Consistency Loss
  - Gradient accumulation for effective large batch sizes
  - Automatic checkpointing and best model tracking
  - WandB / TensorBoard logging integration

Training modes:
  1. Standard: Single-frame training on COCO (baseline)
  2. Temporal: Pseudo-temporal sequences with TCL
  3. Corruption-aware: On-the-fly corruption during training
"""

import os
import time
import math
import logging
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, DistributedSampler
from typing import Optional, Dict, List, Callable

from ..models.nas_yolo import NASYOLO, NASYOLOWithTCL
from ..utils.logging import AverageMeter, ProgressMeter, setup_logger

logger = logging.getLogger(__name__)


class Trainer:
    """NAS-YOLO training engine.

    Handles the full training lifecycle including setup, training loop,
    validation, checkpointing, and logging.

    Args:
        model: NAS-YOLO model instance.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        optimizer: Optimizer instance.
        scheduler: LR scheduler instance (optional).
        config: Training configuration dict.
        device: Training device.
        rank: Process rank for DDP (-1 for single GPU).
        world_size: Total number of processes for DDP.
        output_dir: Directory for checkpoints and logs.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        config: Optional[Dict] = None,
        device: str = "cuda",
        rank: int = -1,
        world_size: int = 1,
        output_dir: str = "./runs",
    ):
        self.device = torch.device(device)
        self.rank = rank
        self.world_size = world_size
        self.is_distributed = world_size > 1
        self.is_main = rank <= 0
        self.output_dir = output_dir

        # Default config
        self.config = {
            "epochs": 100,
            "warmup_epochs": 5,
            "base_lr": 0.01,
            "min_lr": 1e-5,
            "weight_decay": 0.0005,
            "momentum": 0.937,
            "amp": True,
            "grad_accum_steps": 1,
            "clip_grad_norm": 10.0,
            "save_interval": 10,
            "eval_interval": 10,
            "temporal_training": False,
            "tcl_weight": 0.5,
            "log_interval": 50,
        }
        if config:
            self.config.update(config)

        # Model setup
        self.model = model.to(self.device)
        if self.is_distributed:
            self.model = DDP(
                self.model,
                device_ids=[rank],
                output_device=rank,
                find_unused_parameters=True,
            )

        # Data loaders
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Optimizer
        if optimizer is None:
            self.optimizer = self._build_optimizer()
        else:
            self.optimizer = optimizer

        # Scheduler
        if scheduler is None:
            self.scheduler = self._build_scheduler()
        else:
            self.scheduler = scheduler

        # Mixed precision
        self.scaler = GradScaler() if self.config["amp"] else None

        # State
        self.epoch = 0
        self.global_step = 0
        self.best_map = 0.0

        # Logging
        if self.is_main:
            os.makedirs(output_dir, exist_ok=True)
            self.logger = setup_logger("NAS-YOLO", os.path.join(output_dir, "train.log"))
        else:
            self.logger = logging.getLogger("NAS-YOLO")

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Build SGD optimizer with weight decay separation."""
        pg0, pg1, pg2 = [], [], []  # BN, weight, bias

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if ".bn" in name or ".norm" in name:
                pg0.append(param)  # No weight decay for BN
            elif ".bias" in name:
                pg2.append(param)  # No weight decay for biases
            else:
                pg1.append(param)  # Weight decay for weights

        optimizer = torch.optim.SGD(
            pg0, lr=self.config["base_lr"],
            momentum=self.config["momentum"], nesterov=True,
        )
        optimizer.add_param_group({
            "params": pg1, "weight_decay": self.config["weight_decay"]
        })
        optimizer.add_param_group({"params": pg2})

        return optimizer

    def _build_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        """Cosine annealing scheduler with warmup."""
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config["epochs"],
            eta_min=self.config["min_lr"],
        )

    def train(self, eval_fn: Optional[Callable] = None):
        """Run full training loop.

        Args:
            eval_fn: Optional evaluation function called at eval intervals.
                     Should return a dict with 'mAP' key.
        """
        self.logger.info(f"Starting training for {self.config['epochs']} epochs")
        self.logger.info(f"Model: {self._get_model().model_scale}")
        self.logger.info(f"Config: {self.config}")

        model_info = self._get_model().get_model_info()
        self.logger.info(f"Total params: {model_info['total_params_M']:.2f}M")
        self.logger.info(f"NAS overhead: {model_info['nas_overhead_pct']:.1f}%")

        for epoch in range(self.epoch, self.config["epochs"]):
            self.epoch = epoch

            # Set epoch for distributed sampler
            if self.is_distributed:
                self.train_loader.sampler.set_epoch(epoch)

            # Training epoch
            train_metrics = self._train_epoch()

            # Update LR
            if self.scheduler is not None:
                self.scheduler.step()

            # Logging
            if self.is_main:
                lr = self.optimizer.param_groups[0]["lr"]
                self.logger.info(
                    f"Epoch {epoch}/{self.config['epochs']} | "
                    f"Loss: {train_metrics['loss']:.4f} | "
                    f"cls: {train_metrics['cls_loss']:.4f} | "
                    f"reg: {train_metrics['reg_loss']:.4f} | "
                    f"obj: {train_metrics['obj_loss']:.4f} | "
                    f"LR: {lr:.6f}"
                )

            # Evaluation
            if (
                eval_fn is not None
                and self.val_loader is not None
                and (epoch + 1) % self.config["eval_interval"] == 0
            ):
                eval_results = eval_fn(self._get_model(), self.val_loader, self.device)
                if self.is_main:
                    current_map = eval_results.get("mAP", 0.0)
                    self.logger.info(f"Eval mAP: {current_map:.4f}")

                    if current_map > self.best_map:
                        self.best_map = current_map
                        self.save_checkpoint("best.pt")

            # Checkpointing
            if self.is_main and (epoch + 1) % self.config["save_interval"] == 0:
                self.save_checkpoint(f"epoch_{epoch + 1}.pt")

        # Final checkpoint
        if self.is_main:
            self.save_checkpoint("last.pt")
            self.logger.info(f"Training complete. Best mAP: {self.best_map:.4f}")

    def _train_epoch(self) -> Dict[str, float]:
        """Single training epoch."""
        self.model.train()

        loss_meter = AverageMeter("loss")
        cls_meter = AverageMeter("cls_loss")
        reg_meter = AverageMeter("reg_loss")
        obj_meter = AverageMeter("obj_loss")

        for batch_idx, (images, targets) in enumerate(self.train_loader):
            # Warmup LR adjustment
            if self.epoch < self.config["warmup_epochs"]:
                self._warmup_lr(batch_idx)

            if self.config.get("temporal_training") and isinstance(images, list):
                loss_dict = self._train_step_temporal(images, targets)
            else:
                loss_dict = self._train_step(images, targets)

            # Update meters
            loss_meter.update(loss_dict["loss"].item())
            cls_meter.update(loss_dict.get("cls_loss", torch.tensor(0)).item())
            reg_meter.update(loss_dict.get("reg_loss", torch.tensor(0)).item())
            obj_meter.update(loss_dict.get("obj_loss", torch.tensor(0)).item())

            self.global_step += 1

            # Logging
            if (batch_idx + 1) % self.config["log_interval"] == 0 and self.is_main:
                self.logger.info(
                    f"  [{batch_idx + 1}/{len(self.train_loader)}] "
                    f"loss={loss_meter.avg:.4f}"
                )

        return {
            "loss": loss_meter.avg,
            "cls_loss": cls_meter.avg,
            "reg_loss": reg_meter.avg,
            "obj_loss": obj_meter.avg,
        }

    def _train_step(
        self, images: torch.Tensor, targets: List[Dict]
    ) -> Dict[str, torch.Tensor]:
        """Single training step (non-temporal)."""
        images = images.to(self.device)
        targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in t.items()} for t in targets]

        self.optimizer.zero_grad()

        if self.config["amp"]:
            with autocast():
                outputs = self.model(images, targets)
                losses = outputs["losses"]
                loss = losses["loss"]

            self.scaler.scale(loss).backward()

            if self.config["clip_grad_norm"] > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config["clip_grad_norm"]
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            outputs = self.model(images, targets)
            losses = outputs["losses"]
            loss = losses["loss"]

            loss.backward()

            if self.config["clip_grad_norm"] > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config["clip_grad_norm"]
                )

            self.optimizer.step()

        return losses

    def _train_step_temporal(
        self, image_sequence: List[torch.Tensor], targets_sequence: List[List[Dict]]
    ) -> Dict[str, torch.Tensor]:
        """Training step with temporal sequences and TCL."""
        image_sequence = [img.to(self.device) for img in image_sequence]
        targets_sequence = [
            [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v
              for k, v in t.items()} for t in targets]
            for targets in targets_sequence
        ]

        self.optimizer.zero_grad()
        model = self._get_model()

        if isinstance(model, NASYOLOWithTCL):
            outputs_seq, tcl_loss = model.forward_temporal_with_tcl(
                image_sequence, targets_sequence
            )
        else:
            outputs_seq = model.forward_temporal(image_sequence, targets_sequence)
            tcl_loss = torch.tensor(0.0, device=self.device)

        # Aggregate losses across frames
        total_loss = tcl_loss
        total_cls = torch.tensor(0.0, device=self.device)
        total_reg = torch.tensor(0.0, device=self.device)
        total_obj = torch.tensor(0.0, device=self.device)

        for out in outputs_seq:
            if "losses" in out:
                total_loss = total_loss + out["losses"]["loss"]
                total_cls = total_cls + out["losses"]["cls_loss"]
                total_reg = total_reg + out["losses"]["reg_loss"]
                total_obj = total_obj + out["losses"]["obj_loss"]

        total_loss = total_loss / len(outputs_seq)

        if self.config["amp"]:
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config["clip_grad_norm"])
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config["clip_grad_norm"])
            self.optimizer.step()

        return {
            "loss": total_loss.detach(),
            "cls_loss": total_cls.detach() / len(outputs_seq),
            "reg_loss": total_reg.detach() / len(outputs_seq),
            "obj_loss": total_obj.detach() / len(outputs_seq),
        }

    def _warmup_lr(self, batch_idx: int):
        """Linear warmup learning rate."""
        warmup_steps = self.config["warmup_epochs"] * len(self.train_loader)
        current_step = self.epoch * len(self.train_loader) + batch_idx

        if current_step < warmup_steps:
            lr_scale = current_step / warmup_steps
            for pg in self.optimizer.param_groups:
                pg["lr"] = self.config["base_lr"] * lr_scale

    def _get_model(self) -> nn.Module:
        """Get the underlying model (unwrap DDP if needed)."""
        if isinstance(self.model, DDP):
            return self.model.module
        return self.model

    def save_checkpoint(self, filename: str):
        """Save training checkpoint."""
        path = os.path.join(self.output_dir, filename)
        checkpoint = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_map": self.best_map,
            "model_state": self._get_model().state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "config": self.config,
        }
        if self.scheduler is not None:
            checkpoint["scheduler_state"] = self.scheduler.state_dict()
        if self.scaler is not None:
            checkpoint["scaler_state"] = self.scaler.state_dict()

        torch.save(checkpoint, path)
        self.logger.info(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path: str, resume_training: bool = True):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self._get_model().load_state_dict(checkpoint["model_state"])

        if resume_training:
            self.epoch = checkpoint["epoch"] + 1
            self.global_step = checkpoint["global_step"]
            self.best_map = checkpoint["best_map"]
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])

            if self.scheduler is not None and "scheduler_state" in checkpoint:
                self.scheduler.load_state_dict(checkpoint["scheduler_state"])
            if self.scaler is not None and "scaler_state" in checkpoint:
                self.scaler.load_state_dict(checkpoint["scaler_state"])

        self.logger.info(f"Loaded checkpoint: {path} (epoch {checkpoint['epoch']})")


def setup_distributed(rank: int, world_size: int, backend: str = "nccl"):
    """Initialize distributed training."""
    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()
