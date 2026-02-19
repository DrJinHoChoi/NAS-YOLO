#!/bin/bash
# ============================================================================
# Baseline Evaluation Script
# ============================================================================
# Evaluates pre-trained baseline models for paper comparison tables.
#
# Prerequisites:
#   - pip install ultralytics  (for YOLOv8/v10)
#   - Download baseline checkpoints
#
# Usage:
#   bash nas_yolo/experiments/run_baselines.sh [GPU_ID]
# ============================================================================

set -e

GPU=${1:-0}
OUTPUT_ROOT="runs/baselines"
export CUDA_VISIBLE_DEVICES=$GPU

mkdir -p $OUTPUT_ROOT/logs

echo "=============================================="
echo "Baseline Evaluation for NAS-YOLO Paper"
echo "=============================================="

# --- YOLOv8 ---
echo "[1/4] Evaluating YOLOv8 baselines..."

# YOLOv8-n
python -c "
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
results = model.val(data='coco.yaml', imgsz=640, batch=16)
print(f'YOLOv8-n: mAP={results.box.map:.4f}, mAP50={results.box.map50:.4f}')
" 2>&1 | tee $OUTPUT_ROOT/logs/yolov8n.log

# YOLOv8-s
python -c "
from ultralytics import YOLO
model = YOLO('yolov8s.pt')
results = model.val(data='coco.yaml', imgsz=640, batch=16)
print(f'YOLOv8-s: mAP={results.box.map:.4f}, mAP50={results.box.map50:.4f}')
" 2>&1 | tee $OUTPUT_ROOT/logs/yolov8s.log

# --- Corruption Evaluation for Baselines ---
echo ""
echo "[2/4] Evaluating baselines on corruptions..."

python -c "
import json
from nas_yolo.data.corruption import CORRUPTION_TYPES

# This script evaluates a baseline model on each corruption
# Replace with actual evaluation loop for each baseline
corruptions = list(CORRUPTION_TYPES.keys())
print(f'Will evaluate on {len(corruptions)} corruption types x 5 severities')
print('Corruption types:', corruptions)
" 2>&1 | tee $OUTPUT_ROOT/logs/corruption_plan.log

# --- BSI Evaluation for Baselines ---
echo ""
echo "[3/4] Evaluating baseline BSI..."
echo "  (Run inference on pseudo-temporal sequences and compute BSI)"
echo "  See nas_yolo/experiments/eval_baseline_bsi.py"

# --- Summary ---
echo ""
echo "[4/4] Generating summary..."
echo "  Baseline results saved to: $OUTPUT_ROOT/"
echo ""
echo "=============================================="
echo "Next: Run generate_tables.py to create LaTeX tables"
echo "=============================================="
