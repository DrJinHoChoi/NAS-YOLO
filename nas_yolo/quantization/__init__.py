"""
INT8 Quantization Pipeline for PicoYOLO
=========================================
Post-Training Quantization (PTQ) and Quantization-Aware Training (QAT)
for deploying PicoYOLO on INT8 NPUs and edge processors.

Target: FP32 (1.2MB) → INT8 (~350KB) with <1% mAP degradation.

Special considerations for SA-SSM:
    - A_log parameter: Controls SSM stability (A = -exp(A_log) < 0).
      Must be quantized carefully — use per-channel INT8 or keep FP16.
    - DCT basis: Already non-learnable buffer, inherently fixed-point friendly.
    - SE sigmoid: Use INT8 lookup table for efficient inference.
    - Conv1d (d_conv=1): Quantizes cleanly — no kernel accumulation errors.
"""

from .ptq import PicoQuantizer, CalibrationRunner
from .qat import PicoQATTrainer

__all__ = [
    "PicoQuantizer",
    "CalibrationRunner",
    "PicoQATTrainer",
]
