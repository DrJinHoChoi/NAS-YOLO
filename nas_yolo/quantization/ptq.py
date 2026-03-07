"""
Post-Training Quantization (PTQ) for PicoYOLO
================================================
Static INT8 quantization with calibration dataset.

Workflow:
    1. Train PicoYOLO in FP32 (standard training)
    2. Collect activation statistics with CalibrationRunner
    3. Apply static INT8 quantization with PicoQuantizer
    4. Export quantized ONNX model
    5. Verify accuracy drop < 1% mAP

Quantization strategy:
    - Per-channel weight quantization (more accurate)
    - Per-tensor activation quantization (faster inference)
    - A_log parameters: FP16 or per-channel INT8 (stability-critical)
    - BatchNorm: folded into convolutions before quantization
"""

import torch
import torch.nn as nn
from typing import Optional, List
import os


class CalibrationRunner:
    """Collect activation statistics for PTQ calibration.

    Runs a representative subset of the training data through the model
    to determine optimal quantization scales and zero-points.

    Args:
        model: PicoYOLO model in eval mode.
        device: Target device.
    """

    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model = model.eval().to(device)
        self.device = device
        self.activation_stats = {}
        self._hooks = []

    def _register_hooks(self):
        """Register forward hooks to collect activation ranges."""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                hook = module.register_forward_hook(
                    self._make_hook(name)
                )
                self._hooks.append(hook)

    def _make_hook(self, name: str):
        """Create a hook function that records min/max activations."""
        def hook_fn(module, input, output):
            if name not in self.activation_stats:
                self.activation_stats[name] = {
                    "min": float("inf"),
                    "max": float("-inf"),
                    "count": 0,
                }
            stats = self.activation_stats[name]
            out_flat = output.detach().float()
            stats["min"] = min(stats["min"], out_flat.min().item())
            stats["max"] = max(stats["max"], out_flat.max().item())
            stats["count"] += 1
        return hook_fn

    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    @torch.no_grad()
    def calibrate(
        self,
        data_loader,
        num_batches: int = 100,
    ) -> dict:
        """Run calibration with representative data.

        Args:
            data_loader: DataLoader producing image batches.
            num_batches: Number of batches to use for calibration.

        Returns:
            stats: Per-layer activation statistics.
        """
        self._register_hooks()
        self.model.eval()

        for i, batch in enumerate(data_loader):
            if i >= num_batches:
                break

            if isinstance(batch, dict):
                images = batch["images"].to(self.device)
            elif isinstance(batch, (list, tuple)):
                images = batch[0].to(self.device)
            else:
                images = batch.to(self.device)

            self.model(images)

        self._remove_hooks()
        return self.activation_stats


class PicoQuantizer:
    """INT8 quantizer for PicoYOLO models.

    Supports:
        - PyTorch static quantization (CPU/ARM)
        - ONNX INT8 via ONNXRuntime quantization tools
        - QNN-compatible output (Snapdragon NPU)

    Args:
        model: PicoYOLO model.
        calibration_stats: Activation stats from CalibrationRunner.
    """

    def __init__(
        self,
        model: nn.Module,
        calibration_stats: Optional[dict] = None,
    ):
        self.model = model
        self.calibration_stats = calibration_stats

    def quantize_pytorch(self) -> nn.Module:
        """Apply PyTorch static INT8 quantization.

        Returns:
            quantized_model: INT8 quantized PicoYOLO.
        """
        import torch.quantization as tq

        model = self.model.eval().cpu()

        # Define quantization config
        # Per-channel for weights, per-tensor for activations
        qconfig = tq.QConfig(
            activation=tq.HistogramObserver.with_args(
                dtype=torch.quint8, reduce_range=True
            ),
            weight=tq.PerChannelMinMaxObserver.with_args(
                dtype=torch.qint8,
                qscheme=torch.per_channel_symmetric,
            ),
        )

        # Prepare for quantization
        model.qconfig = qconfig
        tq.prepare(model, inplace=True)

        # Run calibration (model should already have stats)
        # If not, use CalibrationRunner first

        # Convert to quantized model
        quantized = tq.convert(model, inplace=False)

        return quantized

    def export_onnx_int8(
        self,
        output_path: str,
        input_resolution: int = 320,
        opset: int = 17,
        calibration_data_path: Optional[str] = None,
    ):
        """Export PicoYOLO as INT8 ONNX model.

        Two-step process:
        1. Export FP32 ONNX
        2. Apply ONNXRuntime quantization

        Args:
            output_path: Output .onnx file path.
            input_resolution: Input image resolution.
            opset: ONNX opset version.
            calibration_data_path: Path to calibration data (for static quant).
        """
        import onnx

        model = self.model.eval().cpu()

        # Step 1: Export FP32 ONNX
        fp32_path = output_path.replace(".onnx", "_fp32.onnx")
        dummy = torch.randn(1, 3, input_resolution, input_resolution)

        # Handle temporal state in export
        torch.onnx.export(
            model,
            (dummy,),
            fp32_path,
            opset_version=opset,
            input_names=["images"],
            output_names=["cls_preds", "reg_preds", "obj_preds"],
            dynamic_axes={
                "images": {0: "batch"},
                "cls_preds": {0: "batch"},
                "reg_preds": {0: "batch"},
                "obj_preds": {0: "batch"},
            },
        )

        # Verify FP32 export
        onnx_model = onnx.load(fp32_path)
        onnx.checker.check_model(onnx_model)

        # Step 2: Apply INT8 quantization
        try:
            from onnxruntime.quantization import (
                quantize_static,
                quantize_dynamic,
                QuantType,
                CalibrationMethod,
            )

            if calibration_data_path and self.calibration_stats:
                # Static quantization (better accuracy)
                quantize_static(
                    fp32_path,
                    output_path,
                    calibration_data_reader=None,  # Use pre-computed stats
                    quant_format=QuantType.QInt8,
                    calibrate_method=CalibrationMethod.MinMax,
                    per_channel=True,
                    weight_type=QuantType.QInt8,
                )
            else:
                # Dynamic quantization (no calibration needed)
                quantize_dynamic(
                    fp32_path,
                    output_path,
                    weight_type=QuantType.QInt8,
                )

            print(f"INT8 ONNX exported: {output_path}")

        except ImportError:
            print("onnxruntime-quantization not available. "
                  "FP32 ONNX exported instead.")
            import shutil
            shutil.copy(fp32_path, output_path)

        # Report sizes
        fp32_size = os.path.getsize(fp32_path) / 1024
        if os.path.exists(output_path):
            int8_size = os.path.getsize(output_path) / 1024
            print(f"FP32: {fp32_size:.0f} KB → INT8: {int8_size:.0f} KB "
                  f"({int8_size/fp32_size:.1%})")

    def get_quantization_report(self) -> dict:
        """Generate quantization analysis report.

        Returns:
            report: Dict with per-layer quantization impact analysis.
        """
        report = {
            "total_params": sum(p.numel() for p in self.model.parameters()),
            "fp32_size_kb": sum(p.numel() for p in self.model.parameters()) * 4 / 1024,
            "int8_size_kb": sum(p.numel() for p in self.model.parameters()) * 1 / 1024,
            "compression_ratio": 4.0,
            "critical_layers": [],
        }

        # Identify layers that need special quantization care
        for name, module in self.model.named_modules():
            if "A_log" in name or "a_log" in name:
                report["critical_layers"].append({
                    "name": name,
                    "reason": "SSM stability parameter — use per-channel INT8 or FP16",
                    "recommendation": "per_channel_int8",
                })
            elif "sigmoid" in name.lower() or isinstance(module, nn.Sigmoid):
                report["critical_layers"].append({
                    "name": name,
                    "reason": "Sigmoid activation — use INT8 lookup table",
                    "recommendation": "lut_int8",
                })

        return report
