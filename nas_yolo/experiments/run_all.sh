#!/bin/bash
# ============================================================================
# NAS-YOLO: Complete Experiment Pipeline for ECCV 2026
# ============================================================================
# Runs ALL experiments needed for the paper in sequence:
#   1. Train NAS-YOLO variants (nano, small)
#   2. Train ablation models
#   3. Evaluate all models (clean + corruption + stability + latency)
#   4. Generate result tables and figures
#
# Usage:
#   bash nas_yolo/experiments/run_all.sh [GPU_ID] [DATA_ROOT]
#
# Prerequisites:
#   - COCO 2017 dataset at DATA_ROOT/coco/
#   - PyTorch + CUDA installed
#   - nas_yolo/requirements.txt installed
# ============================================================================

set -e

GPU=${1:-0}
DATA_ROOT=${2:-"data"}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_ROOT="runs/eccv2026_${TIMESTAMP}"

export CUDA_VISIBLE_DEVICES=$GPU

echo "=============================================="
echo "NAS-YOLO ECCV 2026 Experiment Pipeline"
echo "=============================================="
echo "GPU: $GPU"
echo "Data root: $DATA_ROOT"
echo "Output: $OUTPUT_ROOT"
echo "Started: $(date)"
echo "=============================================="

mkdir -p $OUTPUT_ROOT/logs

# ============================================================================
# Phase 1: Training (est. ~5 days on single A100)
# ============================================================================
echo ""
echo "[Phase 1/4] Training models..."

# --- 1a. NAS-YOLO-s (full model, primary result) ---
echo "  [1/6] Training NAS-YOLO-s (full)..."
python -m nas_yolo.scripts.train \
  --config nas_yolo/configs/default.yaml \
  --temporal \
  2>&1 | tee $OUTPUT_ROOT/logs/train_nasyolo_s.log

# --- 1b. NAS-YOLO-n (nano, edge result) ---
echo "  [2/6] Training NAS-YOLO-n..."
python -m nas_yolo.scripts.train \
  --config nas_yolo/configs/nas_yolo_n.yaml \
  2>&1 | tee $OUTPUT_ROOT/logs/train_nasyolo_n.log

# --- 1c. Ablation: No temporal ---
echo "  [3/6] Training ablation: no temporal..."
python -m nas_yolo.scripts.train \
  --config nas_yolo/configs/ablation/no_temporal.yaml \
  2>&1 | tee $OUTPUT_ROOT/logs/train_ablation_no_temporal.log

# --- 1d. Ablation: No noise gate ---
echo "  [4/6] Training ablation: no noise gate..."
python -m nas_yolo.scripts.train \
  --config nas_yolo/configs/ablation/no_noise_gate.yaml \
  2>&1 | tee $OUTPUT_ROOT/logs/train_ablation_no_gate.log

# --- 1e. Ablation: No TCL ---
echo "  [5/6] Training ablation: no TCL..."
python -m nas_yolo.scripts.train \
  --config nas_yolo/configs/ablation/no_tcl.yaml \
  2>&1 | tee $OUTPUT_ROOT/logs/train_ablation_no_tcl.log

# --- 1f. Ablation: Window length study ---
echo "  [6/6] Training ablation: temporal windows T=1,3,7..."
for T in 1 3 7; do
  python -m nas_yolo.scripts.train \
    --config nas_yolo/configs/ablation/temporal_window.yaml \
    2>&1 | tee $OUTPUT_ROOT/logs/train_ablation_window_T${T}.log
done

echo "[Phase 1 complete]"

# ============================================================================
# Phase 2: Evaluation (est. ~1 day)
# ============================================================================
echo ""
echo "[Phase 2/4] Evaluating all models..."

MODELS=(
  "runs/nas_yolo_s/best.pt:nasyolo_s_full"
  "runs/nas_yolo_n/best.pt:nasyolo_n"
  "runs/ablation/no_temporal/best.pt:ablation_no_temporal"
  "runs/ablation/no_noise_gate/best.pt:ablation_no_gate"
  "runs/ablation/no_tcl/best.pt:ablation_no_tcl"
)

for model_spec in "${MODELS[@]}"; do
  IFS=':' read -r ckpt name <<< "$model_spec"
  echo "  Evaluating $name..."

  if [ -f "$ckpt" ]; then
    python -m nas_yolo.scripts.evaluate \
      --checkpoint "$ckpt" \
      --full \
      --output "$OUTPUT_ROOT/${name}_results.json" \
      2>&1 | tee "$OUTPUT_ROOT/logs/eval_${name}.log"
  else
    echo "  WARNING: $ckpt not found, skipping."
  fi
done

echo "[Phase 2 complete]"

# ============================================================================
# Phase 3: Benchmarking (est. ~30 min)
# ============================================================================
echo ""
echo "[Phase 3/4] Benchmarking latency..."

python -m nas_yolo.scripts.benchmark \
  --all-scales \
  --overhead-analysis \
  2>&1 | tee $OUTPUT_ROOT/logs/benchmark.log

echo "[Phase 3 complete]"

# ============================================================================
# Phase 4: Generate paper materials (est. ~10 min)
# ============================================================================
echo ""
echo "[Phase 4/4] Generating paper materials..."

# Severity response visualization
if [ -f "$OUTPUT_ROOT/nasyolo_s_full_results.json" ]; then
  python -m nas_yolo.scripts.visualize \
    --mode severity \
    --results "$OUTPUT_ROOT/nasyolo_s_full_results.json" \
    --output-dir "$OUTPUT_ROOT/figures"
fi

# Ablation comparison
python -m nas_yolo.scripts.visualize \
  --mode ablation \
  --results "$OUTPUT_ROOT" \
  --output-dir "$OUTPUT_ROOT/figures"

# Generate result tables
python -m nas_yolo.experiments.generate_tables \
  --results-dir "$OUTPUT_ROOT" \
  --output "$OUTPUT_ROOT/tables"

echo "[Phase 4 complete]"

# ============================================================================
echo ""
echo "=============================================="
echo "ALL EXPERIMENTS COMPLETE"
echo "Results: $OUTPUT_ROOT"
echo "Finished: $(date)"
echo "=============================================="
