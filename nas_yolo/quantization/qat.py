"""
Quantization-Aware Training (QAT) for PicoYOLO
=================================================
Fine-tune PicoYOLO with fake-quantization nodes to learn
quantization-robust weights. Typically 10-20 epochs after FP32 training.

QAT achieves better accuracy than PTQ (< 0.5% mAP drop typically)
at the cost of additional training time.

Workflow:
    1. Train PicoYOLO in FP32 (300 epochs)
    2. Insert fake-quant nodes (prepare_model)
    3. Fine-tune with QAT for 10-20 epochs
    4. Convert to actual INT8 (convert_model)
    5. Export quantized ONNX
"""

import torch
import torch.nn as nn
from typing import Optional
import copy


class PicoQATTrainer:
    """Quantization-Aware Training wrapper for PicoYOLO.

    Inserts fake quantization nodes during training so the model
    learns weights that are robust to INT8 quantization.

    Args:
        model: Pre-trained FP32 PicoYOLO model.
        qat_epochs: Number of QAT fine-tuning epochs.
        qat_lr: Learning rate for QAT (typically 10× smaller than training LR).
    """

    def __init__(
        self,
        model: nn.Module,
        qat_epochs: int = 20,
        qat_lr: float = 0.001,
    ):
        self.original_model = model
        self.qat_epochs = qat_epochs
        self.qat_lr = qat_lr
        self.qat_model = None

    def prepare_model(self) -> nn.Module:
        """Insert fake-quantization nodes for QAT.

        Returns:
            qat_model: Model with fake-quant nodes ready for QAT training.
        """
        import torch.quantization as tq

        # Deep copy to preserve original
        self.qat_model = copy.deepcopy(self.original_model)
        self.qat_model.train()

        # QAT config: fake quantization with learnable scale/zero-point
        qconfig = tq.QConfig(
            activation=tq.FakeQuantize.with_args(
                observer=tq.MovingAverageMinMaxObserver,
                dtype=torch.quint8,
                reduce_range=True,
            ),
            weight=tq.FakeQuantize.with_args(
                observer=tq.MovingAveragePerChannelMinMaxObserver,
                dtype=torch.qint8,
                qscheme=torch.per_channel_symmetric,
            ),
        )

        self.qat_model.qconfig = qconfig
        tq.prepare_qat(self.qat_model, inplace=True)

        return self.qat_model

    def convert_model(self) -> nn.Module:
        """Convert QAT model to actual INT8 model.

        Must be called after QAT training is complete.

        Returns:
            int8_model: Fully quantized INT8 model.
        """
        import torch.quantization as tq

        if self.qat_model is None:
            raise RuntimeError("Call prepare_model() first")

        self.qat_model.eval()
        int8_model = tq.convert(self.qat_model, inplace=False)

        return int8_model

    def get_qat_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer for QAT fine-tuning.

        Uses smaller LR and focuses on quantization-sensitive layers.

        Returns:
            optimizer: Configured SGD optimizer for QAT.
        """
        if self.qat_model is None:
            raise RuntimeError("Call prepare_model() first")

        # Group params: higher LR for quantization observers
        param_groups = [
            {
                "params": [p for n, p in self.qat_model.named_parameters()
                          if "fake_quant" not in n and p.requires_grad],
                "lr": self.qat_lr,
            },
        ]

        return torch.optim.SGD(
            param_groups,
            momentum=0.9,
            weight_decay=1e-4,
        )

    def get_training_config(self) -> dict:
        """Get recommended QAT training configuration.

        Returns:
            config: QAT training hyperparameters.
        """
        return {
            "epochs": self.qat_epochs,
            "lr": self.qat_lr,
            "lr_schedule": "cosine",
            "warmup_epochs": 1,
            "weight_decay": 1e-4,
            "batch_size": 64,
            "amp": False,  # QAT typically doesn't use AMP
            "ema": True,
            "freeze_bn_after": 5,  # Freeze BN stats after N epochs
            "expected_mAP_drop": "<0.5%",
        }
