#!/usr/bin/env python3
"""
NAS-YOLO Training Script
==========================
Usage:
    # Single GPU
    python -m nas_yolo.scripts.train --config configs/default.yaml

    # Multi-GPU DDP
    torchrun --nproc_per_node=4 -m nas_yolo.scripts.train --config configs/default.yaml

    # Temporal training
    python -m nas_yolo.scripts.train --config configs/default.yaml --temporal

    # Resume from checkpoint
    python -m nas_yolo.scripts.train --config configs/default.yaml --resume runs/nas_yolo_s/last.pt

    # Ablation: no temporal
    python -m nas_yolo.scripts.train --config configs/ablation/no_temporal.yaml
"""

import os
import sys
import argparse
import yaml
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from nas_yolo.models import NASYOLO
from nas_yolo.models.nas_yolo import NASYOLOWithTCL
from nas_yolo.data.dataset import (
    COCODetectionDataset,
    PseudoTemporalDataset,
    build_dataloader,
)
from nas_yolo.data.transforms import DetectionTransform
from nas_yolo.data.corruption import CorruptionTransform
from nas_yolo.engine.trainer import Trainer, setup_distributed, cleanup_distributed
from nas_yolo.engine.evaluator import Evaluator


def parse_args():
    parser = argparse.ArgumentParser(description="NAS-YOLO Training")
    parser.add_argument("--config", type=str, default="nas_yolo/configs/default.yaml",
                        help="Path to configuration file")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint for resuming training")
    parser.add_argument("--temporal", action="store_true",
                        help="Enable temporal training mode")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Training device")
    # Config overrides (CLI takes precedence over YAML)
    parser.add_argument("--data-root", type=str, default=None,
                        help="Override COCO data root directory")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--amp", action="store_true", default=None,
                        help="Enable mixed precision training")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override base learning rate")
    parser.add_argument("--workers", type=int, default=None,
                        help="Override number of data workers")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load YAML config with default fallback."""
    default_path = os.path.join(os.path.dirname(config_path), "default.yaml")

    # Load defaults
    config = {}
    if os.path.exists(default_path) and config_path != default_path:
        with open(default_path) as f:
            config = yaml.safe_load(f) or {}

    # Override with specific config
    with open(config_path) as f:
        override = yaml.safe_load(f) or {}

    # Deep merge
    for key, value in override.items():
        if isinstance(value, dict) and key in config:
            config[key].update(value)
        else:
            config[key] = value

    return config


def build_model(config: dict) -> torch.nn.Module:
    """Build NAS-YOLO model from config."""
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})

    if train_cfg.get("temporal_training") and train_cfg.get("tcl_weight", 0) > 0:
        model = NASYOLOWithTCL(
            tcl_weight=train_cfg["tcl_weight"],
            tcl_ema=train_cfg.get("tcl_ema", 0.95),
            num_classes=model_cfg.get("num_classes", 80),
            model_scale=model_cfg.get("model_scale", "small"),
            use_noise_gate=model_cfg.get("use_noise_gate", True),
            use_temporal=model_cfg.get("use_temporal", True),
            noise_dim=model_cfg.get("noise_dim", 32),
            ema_momentum=model_cfg.get("ema_momentum", 0.9),
            pretrained_backbone=model_cfg.get("pretrained_backbone"),
        )
    else:
        model = NASYOLO(
            num_classes=model_cfg.get("num_classes", 80),
            model_scale=model_cfg.get("model_scale", "small"),
            use_noise_gate=model_cfg.get("use_noise_gate", True),
            use_temporal=model_cfg.get("use_temporal", True),
            noise_dim=model_cfg.get("noise_dim", 32),
            ema_momentum=model_cfg.get("ema_momentum", 0.9),
            pretrained_backbone=model_cfg.get("pretrained_backbone"),
        )

    return model


def build_datasets(config: dict, is_temporal: bool = False):
    """Build training and validation datasets."""
    data_cfg = config.get("data", {})
    aug_cfg = config.get("augmentation", {})
    train_cfg = config.get("training", {})

    img_size = tuple(data_cfg.get("img_size", [640, 640]))

    # Training transform
    train_transform = DetectionTransform(
        img_size=img_size,
        augment=True,
        hsv_h=aug_cfg.get("hsv_h", 0.015),
        hsv_s=aug_cfg.get("hsv_s", 0.7),
        hsv_v=aug_cfg.get("hsv_v", 0.4),
        flip_p=aug_cfg.get("flip_p", 0.5),
    )

    # Corruption augmentation (optional)
    corruption = None
    if aug_cfg.get("corruption_aug"):
        corruption = CorruptionTransform(
            corruption_type="random",
            severity=3,
            p=aug_cfg.get("corruption_p", 0.3),
        )

    # Training dataset
    train_dataset = COCODetectionDataset(
        root_dir=data_cfg.get("train_root", "data/coco/train2017"),
        ann_file=data_cfg.get("train_ann", "data/coco/annotations/instances_train2017.json"),
        transform=train_transform,
        corruption=corruption,
        img_size=img_size,
    )

    # Wrap with pseudo-temporal if enabled
    if is_temporal or train_cfg.get("temporal_training"):
        train_dataset = PseudoTemporalDataset(
            base_dataset=train_dataset,
            sequence_length=train_cfg.get("sequence_length", 5),
            corruption_type=train_cfg.get("corruption_type", "gaussian_noise"),
        )

    # Validation dataset
    val_transform = DetectionTransform(img_size=img_size, augment=False)
    val_dataset = COCODetectionDataset(
        root_dir=data_cfg.get("val_root", "data/coco/val2017"),
        ann_file=data_cfg.get("val_ann", "data/coco/annotations/instances_val2017.json"),
        transform=val_transform,
        img_size=img_size,
    )

    return train_dataset, val_dataset


def main():
    args = parse_args()
    config = load_config(args.config)

    # Distributed setup
    rank = int(os.environ.get("RANK", args.local_rank))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed = world_size > 1

    if is_distributed:
        setup_distributed(rank, world_size)

    # Apply CLI overrides to config
    if args.temporal:
        config.setdefault("training", {})["temporal_training"] = True
    if args.data_root is not None:
        config.setdefault("data", {})["train_root"] = os.path.join(args.data_root, "train2017")
        config["data"]["val_root"] = os.path.join(args.data_root, "val2017")
        config["data"]["train_ann"] = os.path.join(args.data_root, "annotations/instances_train2017.json")
        config["data"]["val_ann"] = os.path.join(args.data_root, "annotations/instances_val2017.json")
    if args.output_dir is not None:
        config.setdefault("output", {})["dir"] = args.output_dir
    if args.epochs is not None:
        config.setdefault("training", {})["epochs"] = args.epochs
    if args.batch_size is not None:
        config.setdefault("training", {})["batch_size"] = args.batch_size
    if args.amp is not None:
        config.setdefault("training", {})["amp"] = args.amp
    if args.lr is not None:
        config.setdefault("training", {})["base_lr"] = args.lr
    if args.workers is not None:
        config.setdefault("data", {})["num_workers"] = args.workers

    # Build model
    model = build_model(config)

    # Print model info
    if rank <= 0:
        info = model.get_model_info()
        print(f"\n{'=' * 60}")
        print(f"NAS-YOLO-{config['model']['model_scale']}")
        print(f"{'=' * 60}")
        print(f"Total params:     {info['total_params_M']:.2f}M")
        print(f"Backbone:         {info['backbone_params_M']:.2f}M")
        print(f"Neck:             {info['neck_params_M']:.2f}M")
        print(f"NAS Module:       {info['nas_module_params_M']:.2f}M")
        print(f"Noise Estimator:  {info['noise_estimator_params_M']:.2f}M")
        print(f"Head:             {info['head_params_M']:.2f}M")
        print(f"NAS overhead:     {info['nas_overhead_pct']:.1f}%")
        print(f"{'=' * 60}\n")

    # Build datasets
    is_temporal = config.get("training", {}).get("temporal_training", False)
    train_dataset, val_dataset = build_datasets(config, is_temporal)

    train_cfg = config.get("training", {})
    batch_size = train_cfg.get("batch_size", 16)

    if is_distributed:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        train_loader = build_dataloader(
            train_dataset, batch_size=batch_size,
            num_workers=config.get("data", {}).get("num_workers", 8),
            shuffle=False,
        )
        train_loader.sampler = train_sampler
    else:
        train_loader = build_dataloader(
            train_dataset, batch_size=batch_size,
            num_workers=config.get("data", {}).get("num_workers", 8),
        )

    val_loader = build_dataloader(
        val_dataset, batch_size=batch_size,
        num_workers=config.get("data", {}).get("num_workers", 8),
        shuffle=False,
    )

    # Build evaluator
    evaluator = Evaluator(
        num_classes=config.get("model", {}).get("num_classes", 80),
        img_size=tuple(config.get("data", {}).get("img_size", [640, 640])),
    )

    def eval_fn(model, loader, device):
        return evaluator.evaluate_map(model, loader, device)

    # Build trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_cfg,
        device=args.device if not is_distributed else f"cuda:{rank}",
        rank=rank,
        world_size=world_size,
        output_dir=config.get("output", {}).get("dir", "runs/nas_yolo"),
    )

    # Resume from checkpoint
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    trainer.train(eval_fn=eval_fn)

    # Cleanup
    if is_distributed:
        cleanup_distributed()


if __name__ == "__main__":
    main()
