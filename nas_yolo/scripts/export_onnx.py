#!/usr/bin/env python3
"""
ONNX Export Script for NAS-YOLO Plugin
========================================
Exports the NASPlugin (or full YOLO+Plugin) to ONNX format for edge deployment.

Supports:
  - Plugin-only export (for integration with external YOLO runtimes)
  - Full model export (YOLO + Plugin, requires ultralytics)
  - Opset version 17 (compatible with ONNX Runtime, TensorRT, QNN)
  - Dynamic batch size
  - INT8/FP16 quantization hints

Usage:
    # Plugin-only export
    python -m nas_yolo.scripts.export_onnx --mode plugin-only

    # Full model export (requires ultralytics)
    python -m nas_yolo.scripts.export_onnx --mode full --weights yolov8n.pt

    # Smart glasses config
    python -m nas_yolo.scripts.export_onnx --mode plugin-only --profile ultra-lite \\
        --output smartglass_plugin.onnx
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from nas_yolo.models.nas_plugin import NASPlugin
from nas_yolo.models.mamba_attention import PROFILE_CONFIGS


class PluginONNXWrapper(nn.Module):
    """Wraps NASPlugin for clean ONNX export.

    ONNX export requires:
      - Fixed number of inputs/outputs (no Optional)
      - No in-place operations on module state
      - Deterministic control flow
    """

    def __init__(self, plugin: NASPlugin):
        super().__init__()
        self.plugin = plugin
        self.channels_list = plugin.channels_list
        self.num_scales = plugin.num_scales

    def forward(self, feat0, feat1, feat2, images):
        """Forward with explicit tensor inputs (no lists for ONNX).

        Args:
            feat0: Scale 0 features [B, C0, H0, W0]
            feat1: Scale 1 features [B, C1, H1, W1]
            feat2: Scale 2 features [B, C2, H2, W2]
            images: Raw input images [B, 3, H, W]

        Returns:
            out0, out1, out2: Enhanced features per scale
            conf0, conf1, conf2: Confidence weights per scale
        """
        features = [feat0, feat1, feat2]
        enhanced, _, conf_weights = self.plugin(features, images=images)

        return (
            enhanced[0], enhanced[1], enhanced[2],
            conf_weights[0], conf_weights[1], conf_weights[2],
        )


def export_plugin_only(
    profile: str = "ultra-lite",
    output_path: str = "nas_plugin.onnx",
    resolution: int = 640,
    opset: int = 17,
    dynamic_batch: bool = True,
):
    """Export NASPlugin standalone to ONNX."""
    channels_list = [64, 128, 256]

    # pool_size=4 for ONNX compatibility (adaptive_avg_pool2d requires
    # output_size to divide input_size; pool_size=4 divides 80,40,20)
    plugin = NASPlugin(
        channels_list=channels_list,
        profile=profile,
        mode="hybrid",
        use_noise_gate=True,
        noise_dim=32,
        pool_size=4,
    )
    plugin.eval()

    wrapper = PluginONNXWrapper(plugin)
    wrapper.eval()

    # Create dummy inputs
    B = 1
    feat_h = resolution // 8
    feat0 = torch.randn(B, channels_list[0], feat_h, feat_h)
    feat1 = torch.randn(B, channels_list[1], feat_h // 2, feat_h // 2)
    feat2 = torch.randn(B, channels_list[2], feat_h // 4, feat_h // 4)
    images = torch.randn(B, 3, resolution, resolution)

    # Input/output names
    input_names = ["feat_scale0", "feat_scale1", "feat_scale2", "images"]
    output_names = [
        "enhanced_scale0", "enhanced_scale1", "enhanced_scale2",
        "conf_scale0", "conf_scale1", "conf_scale2",
    ]

    # Dynamic axes
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {name: {0: "batch"} for name in input_names + output_names}

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"Exporting NASPlugin ({profile}) to ONNX...")
    print(f"  Output: {output_path}")
    print(f"  Opset: {opset}")
    print(f"  Resolution: {resolution}x{resolution}")
    print(f"  Dynamic batch: {dynamic_batch}")

    torch.onnx.export(
        wrapper,
        (feat0, feat1, feat2, images),
        output_path,
        opset_version=opset,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        dynamo=False,  # Use legacy exporter for wider compatibility
    )

    # Verify
    file_size = os.path.getsize(output_path) / 1024
    print(f"  File size: {file_size:.1f} KB")

    # Parameter count
    info = plugin.get_plugin_info()
    print(f"  Plugin params: {info['total_params_M']:.4f}M ({info['total_params']:,})")
    print(f"  Profile: {info['profile']}")

    # Try to verify with onnx
    try:
        import onnx
        model = onnx.load(output_path)
        onnx.checker.check_model(model)
        print(f"  ONNX verification: PASSED")
    except ImportError:
        print(f"  ONNX verification: skipped (pip install onnx)")
    except Exception as e:
        print(f"  ONNX verification: FAILED — {e}")

    # Try to verify with onnxruntime
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(output_path)
        # Auto-detect actual input names (tracer may rename them)
        ort_inputs = sess.get_inputs()
        feed = {}
        dummy_inputs = [feat0.numpy(), feat1.numpy(), feat2.numpy(), images.numpy()]
        for inp, val in zip(ort_inputs, dummy_inputs):
            feed[inp.name] = val
        outputs = sess.run(None, feed)
        print(f"  ORT inference: PASSED")
        for i, out in enumerate(outputs[:3]):
            print(f"    Output {i}: shape={out.shape}")
    except ImportError:
        print(f"  ORT inference: skipped (pip install onnxruntime)")
    except Exception as e:
        print(f"  ORT inference: FAILED — {e}")

    print(f"\nExport complete: {output_path}")
    return output_path


def export_full_model(
    weights_path: str = "yolov8n.pt",
    profile: str = "ultra-lite",
    num_classes: int = 30,
    output_path: str = "smartglass_full.onnx",
    resolution: int = 640,
    opset: int = 17,
):
    """Export full YOLO + NASPlugin model to ONNX."""
    try:
        from nas_yolo.integrations import UltralyticsWithNASPlugin
    except ImportError:
        print("Error: ultralytics required for full model export")
        print("Install with: pip install ultralytics")
        return None

    model = UltralyticsWithNASPlugin(
        weights_path=weights_path,
        plugin_cfg={
            "profile": profile,
            "mode": "hybrid",
            "use_noise_gate": True,
            "noise_dim": 32,
            "pool_size": 8,
        },
        freeze_backbone=False,
        num_classes=num_classes,
    )
    model.eval()

    # Get model info
    info = model.get_model_info()
    print(f"Exporting full model to ONNX...")
    print(f"  Base: {info['base_model']}")
    print(f"  Total params: {info['total_params_M']:.2f}M")
    print(f"  Plugin overhead: {info['overhead_pct']:.1f}%")

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    dummy = torch.randn(1, 3, resolution, resolution)

    torch.onnx.export(
        model,
        dummy,
        output_path,
        opset_version=opset,
        input_names=["images"],
        output_names=["detections"],
        dynamic_axes={"images": {0: "batch"}, "detections": {0: "batch"}},
        do_constant_folding=True,
        dynamo=False,  # Use legacy exporter for wider compatibility
    )

    file_size = os.path.getsize(output_path) / 1024
    print(f"  File size: {file_size:.1f} KB")
    print(f"  Output: {output_path}")

    return output_path


def export_pico(
    num_classes: int = 30,
    base_channels: int = 32,
    output_path: str = "pico_yolo.onnx",
    resolution: int = 320,
    opset: int = 17,
    quantize: bool = False,
    use_context: bool = False,
):
    """Export complete PicoYOLO to ONNX with optional INT8 quantization.

    Unlike export_full_model which wraps Ultralytics, this exports
    the self-contained PicoYOLO model directly.

    Target model sizes:
        FP32: ~1.2MB (0.30M params × 4 bytes)
        FP16: ~600KB
        INT8: ~350KB
    """
    from nas_yolo.models.pico_yolo import PicoYOLO

    model = PicoYOLO(
        num_classes=num_classes,
        base_channels=base_channels,
        input_resolution=resolution,
        use_context=use_context,
    )
    model.eval()

    # Print model info
    model.print_model_info()

    # Ensure output directory
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    dummy = torch.randn(1, 3, resolution, resolution)

    print(f"\nExporting PicoYOLO to ONNX...")
    torch.onnx.export(
        model,
        (dummy,),
        output_path,
        opset_version=opset,
        input_names=["images"],
        output_names=["cls_preds", "reg_preds", "obj_preds"],
        dynamic_axes={
            "images": {0: "batch"},
        },
        do_constant_folding=True,
    )

    file_size = os.path.getsize(output_path) / 1024
    print(f"  FP32 ONNX: {file_size:.0f} KB → {output_path}")

    # Verify
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print(f"  ONNX verification: PASSED")
    except ImportError:
        print(f"  ONNX verification: skipped (pip install onnx)")
    except Exception as e:
        print(f"  ONNX verification: FAILED — {e}")

    # INT8 quantization
    if quantize:
        int8_path = output_path.replace(".onnx", "_int8.onnx")
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            quantize_dynamic(
                output_path, int8_path,
                weight_type=QuantType.QInt8,
            )
            int8_size = os.path.getsize(int8_path) / 1024
            print(f"  INT8 ONNX: {int8_size:.0f} KB → {int8_path}")
            print(f"  Compression: {file_size/int8_size:.1f}×")
        except ImportError:
            print("  INT8 quantization: skipped (pip install onnxruntime)")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="NAS-YOLO ONNX Export")
    parser.add_argument("--mode", type=str, default="plugin-only",
                        choices=["plugin-only", "full", "pico"],
                        help="Export mode")
    parser.add_argument("--profile", type=str, default="ultra-lite",
                        choices=list(PROFILE_CONFIGS.keys()))
    parser.add_argument("--weights", type=str, default="yolov8n.pt",
                        help="YOLO weights (for full export)")
    parser.add_argument("--num-classes", type=int, default=30)
    parser.add_argument("--output", type=str, default=None,
                        help="Output ONNX path")
    parser.add_argument("--resolution", type=int, default=640)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--no-dynamic-batch", action="store_true")
    args = parser.parse_args()

    if args.output is None:
        if args.mode == "plugin-only":
            args.output = f"nas_plugin_{args.profile.replace('-', '_')}.onnx"
        else:
            args.output = f"smartglass_{args.num_classes}class_full.onnx"

    if args.mode == "pico":
        if args.output is None:
            args.output = f"pico_yolo_{args.num_classes}class.onnx"
        export_pico(
            num_classes=args.num_classes,
            output_path=args.output,
            resolution=args.resolution,
            opset=args.opset,
            quantize=True,
        )
    elif args.mode == "plugin-only":
        export_plugin_only(
            profile=args.profile,
            output_path=args.output,
            resolution=args.resolution,
            opset=args.opset,
            dynamic_batch=not args.no_dynamic_batch,
        )
    else:
        export_full_model(
            weights_path=args.weights,
            profile=args.profile,
            num_classes=args.num_classes,
            output_path=args.output,
            resolution=args.resolution,
            opset=args.opset,
        )


if __name__ == "__main__":
    main()
