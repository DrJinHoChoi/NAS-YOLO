# NAS-YOLO: Noise-Aware State Modeling for Robust Real-Time Object Detection

> **"We turn object detection from frame-wise perception into state-aware perception."**

## Abstract

Existing real-time object detectors (YOLO, RT-DETR) operate on single frames without temporal memory, making them vulnerable to visual corruptions (noise, fog, blur, low-light) that cause bounding box jitter, confidence oscillation, and false positive surges. We propose **NAS-YOLO**, a Noise-Aware State Modeling framework that introduces:

1. **Temporal Feature State Modeling** via Selective State Spaces (C1)
2. **Noise-Aware Gating** with spectral condition encoding (C2)
3. **Edge-Friendly Design** under 10M parameters and 5ms latency (C3)
4. **Box Stability Index (BSI)** — a novel metric for temporal detection consistency (C4)

NAS-YOLO sits between the feature pyramid neck and detection head, processing multi-scale features through input-dependent state space recurrences conditioned on estimated noise characteristics. This enables graceful degradation under corruption while preserving real-time performance.

---

## Architecture

```
Frame_t-n ... Frame_t
        ↓
  Shared Backbone (CSPDarknet)
        ↓
  Feature Pyramid Neck (PAFPN)
        ↓
  ┌─────────────────────────────────────────┐
  │  Noise-Aware State Module (our contrib) │
  │                                         │
  │  SpectralNoiseEstimator ──→ noise_desc  │
  │           ↓                             │
  │  SpatialSSM ──→ temporal features       │
  │           ↓                             │
  │  NoiseAwareGate ──→ adaptive blend      │
  │           ↓                             │
  │  ConfidenceRefiner ──→ stable scores    │
  └─────────────────────────────────────────┘
        ↓
  Detection Head (Decoupled)
        ↓
  Stable Boxes + Refined Confidence
```

## Key Contributions

| # | Contribution | Module | Impact |
|---|-------------|--------|--------|
| C1 | Temporal Feature State Modeling via Selective SSM | `nas_module.py` | Reduces box jitter, suppresses false positives |
| C2 | Noise-Aware Gating with spectral condition encoding | `noise_gate.py` | Adapts temporal reliance based on input quality |
| C3 | Edge-Friendly Design (< 10M params) | Full model | Real-time on Jetson Nano (< 5ms) |
| C4 | Box Stability Index (BSI) — novel metric | `box_stability.py` | Quantifies detection temporal consistency |

## Model Variants

| Model | Params | GFLOPs | Target Latency | Use Case |
|-------|--------|--------|----------------|----------|
| NAS-YOLO-n | ~3.5M | ~4G | < 5ms (Nano) | Edge / Embedded |
| NAS-YOLO-s | ~8.5M | ~12G | < 6ms | Balanced |
| NAS-YOLO-m | ~18M | ~26G | < 10ms | Accuracy-focused |
| NAS-YOLO-l | ~35M | ~52G | < 16ms | Maximum accuracy |

## Installation

```bash
cd nas_yolo
pip install -r requirements.txt
```

## Quick Start

### Training

```bash
# Standard training on COCO
python -m nas_yolo.scripts.train --config nas_yolo/configs/default.yaml

# Temporal training with TCL
python -m nas_yolo.scripts.train --config nas_yolo/configs/default.yaml --temporal

# Multi-GPU DDP training
torchrun --nproc_per_node=4 -m nas_yolo.scripts.train --config nas_yolo/configs/default.yaml

# Nano model for edge deployment
python -m nas_yolo.scripts.train --config nas_yolo/configs/nas_yolo_n.yaml
```

### Evaluation

```bash
# Standard mAP
python -m nas_yolo.scripts.evaluate --checkpoint runs/best.pt --config nas_yolo/configs/default.yaml

# Full evaluation (mAP + mAP-C + BSI + Latency)
python -m nas_yolo.scripts.evaluate --checkpoint runs/best.pt --full

# Corruption robustness only
python -m nas_yolo.scripts.evaluate --checkpoint runs/best.pt --corruption-only
```

### Benchmarking

```bash
# All model scales
python -m nas_yolo.scripts.benchmark --all-scales

# NAS module overhead analysis
python -m nas_yolo.scripts.benchmark --overhead-analysis
```

### Running Tests

```bash
pytest nas_yolo/tests/ -v
```

## Project Structure

```
nas_yolo/
├── models/
│   ├── nas_module.py        # C1: Selective SSM temporal state modeling
│   ├── noise_gate.py        # C2: Spectral noise estimation + adaptive gating
│   ├── temporal_buffer.py   # Temporal state management + scene change detection
│   ├── backbone.py          # CSPDarknet backbone (standard)
│   ├── neck.py              # PAFPN multi-scale fusion (standard)
│   ├── head.py              # Decoupled detection head with confidence refinement
│   └── nas_yolo.py          # Full model assembly + loss computation
├── data/
│   ├── corruption.py        # 21 corruption types (15 standard + 6 industrial)
│   ├── dataset.py           # COCO, temporal video, pseudo-temporal loaders
│   └── transforms.py        # Detection augmentations (mosaic, mixup, HSV)
├── metrics/
│   ├── box_stability.py     # C4: Novel Box Stability Index (BSI) metric
│   ├── map.py               # COCO-style mAP computation
│   └── corruption_robustness.py  # mAP-C evaluation protocol
├── engine/
│   ├── trainer.py           # Training loop with DDP + AMP + temporal support
│   └── evaluator.py         # Unified evaluation (mAP + mAP-C + BSI + latency)
├── utils/
│   ├── visualization.py     # Paper-quality figure generation
│   ├── profiling.py         # Params, FLOPs, latency benchmarking
│   └── logging.py           # Logger setup + training meters
├── configs/
│   ├── default.yaml         # Default config (NAS-YOLO-s on COCO)
│   ├── nas_yolo_n.yaml      # Nano (edge deployment)
│   ├── nas_yolo_m.yaml      # Medium (accuracy-focused)
│   └── ablation/
│       ├── no_temporal.yaml      # Ablation: disable SSM
│       ├── no_noise_gate.yaml    # Ablation: disable noise gating
│       ├── no_tcl.yaml           # Ablation: disable temporal consistency loss
│       └── temporal_window.yaml  # Ablation: vary sequence length
├── scripts/
│   ├── train.py             # Training entry point
│   ├── evaluate.py          # Evaluation entry point
│   ├── benchmark.py         # Latency/throughput benchmarking
│   └── visualize.py         # Figure generation
├── tests/
│   ├── test_nas_module.py   # SSM and NAS module tests
│   ├── test_noise_gate.py   # Noise estimation and gating tests
│   ├── test_box_stability.py # BSI metric tests
│   └── test_model.py        # Full model integration tests
└── requirements.txt
```

## Experimental Design

### Datasets

| Dataset | Purpose | Type |
|---------|---------|------|
| COCO 2017 | Clean baseline | Static images |
| COCO-C | Corruption robustness | 15 corruptions × 5 severities |
| ExDark | Low-light detection | Real low-light images |
| ACDC / Cityscapes-C | Adverse driving conditions | Fog, rain, night |
| MVTec (optional) | Industrial inspection | Manufacturing defects |

### Baselines

| Model | Category |
|-------|----------|
| YOLOv8-n/s | Single-frame YOLO |
| YOLOv9-t/s | Latest YOLO variant |
| RT-DETR-R18 | Transformer-based real-time |
| Gold-YOLO-n/s | Efficiency-focused YOLO |
| DAMO-YOLO | Industrial YOLO variant |

### Evaluation Protocol

| Metric | What it measures | Novel? |
|--------|-----------------|--------|
| mAP@50:95 | Detection accuracy (COCO-style) | No |
| mAP-C | Corruption robustness (mean across corruptions) | No |
| **BSI** | **Temporal box stability** | **Yes (ours)** |
| FPS | Real-time performance | No |
| Params (M) | Model complexity | No |

### Ablation Studies

1. **SSM removal** (`no_temporal.yaml`): Measures C1 contribution
2. **Noise gate removal** (`no_noise_gate.yaml`): Measures C2 contribution
3. **TCL removal** (`no_tcl.yaml`): Temporal consistency loss contribution
4. **Window length** (`temporal_window.yaml`): T = 1, 3, 5, 7
5. **Param/FLOPs trade-off**: NAS overhead across model scales

## Technical Details

### Selective State Space Model (SSM)

The SSM processes spatial features as 1D sequences with input-dependent discretization:

```
h_t = Ā · h_{t-1} + B̄ · x_t      (state update)
y_t = C · h_t + D · x_t            (output)

where:
  Ā = exp(Δ · A)                   (discretized transition)
  B̄ = Δ · B                        (discretized input)
  Δ = softplus(Linear(x))          (input-dependent step size)
```

This enables **selective** information propagation: the model learns to pass clean features through while suppressing noisy inputs via the learned Δ parameter.

### Noise-Aware Gating

The gate controls the temporal-spatial blend:

```
gate = σ(W_noise · noise_desc + W_feat · GAP(features))

output = gate × SSM_output + (1 - gate) × current_features
```

Under high noise: gate → 1 (trust temporal state)
Under low noise: gate → 0 (trust current frame)

### Box Stability Index (BSI)

```
BSI = α · IoU_stability + β · Conf_stability + γ · Center_stability

where:
  IoU_stability = mean(IoU(box_t, box_{t-1}))    per matched instance
  Conf_stability = 1 - mean(|score_t - score_{t-1}|)
  Center_stability = 1 - mean(||center_t - center_{t-1}|| / diag)
  α=0.5, β=0.3, γ=0.2
```

## Target Venues

| Priority | Venue | Deadline |
|----------|-------|----------|
| 1st | ICCV 2025 | March 2025 |
| 1st | CVPR 2026 | November 2025 |
| 2nd | ECCV 2026 | March 2026 |
| 3rd | NeurIPS 2025 | May 2025 |

## Citation

```bibtex
@article{nasyolo2025,
  title={NAS-YOLO: Noise-Aware State Modeling for Robust Real-Time Object Detection},
  author={},
  journal={arXiv preprint},
  year={2025}
}
```

## License

Research use only. See LICENSE for details.
