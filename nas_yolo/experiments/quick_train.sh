#!/bin/bash
# ============================================================
# NAS-YOLO Quick Training Pipeline
# ============================================================
# Usage:
#   ./quick_train.sh [STAGE] [GPU_IDS]
#
# Stages:
#   setup    - Install dependencies + download COCO
#   smoke    - Run smoke tests (model builds, forward, loss)
#   train    - Train all model variants (nano, small, medium)
#   ablation - Run ablation experiments
#   eval     - Evaluate on COCO-C + BSI
#   tables   - Generate LaTeX tables
#   all      - Run everything (default)
#
# Examples:
#   ./quick_train.sh all 0,1     # Full pipeline on GPU 0,1
#   ./quick_train.sh train 0     # Only training on GPU 0
#   ./quick_train.sh eval 0      # Only evaluation on GPU 0
# ============================================================

set -euo pipefail

STAGE="${1:-all}"
GPU_IDS="${2:-0}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="${PROJECT_DIR}/data/coco"
RUNS_DIR="${PROJECT_DIR}/runs"

export CUDA_VISIBLE_DEVICES="$GPU_IDS"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${GREEN}[NAS-YOLO]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
err() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }
info() { echo -e "${BLUE}[INFO]${NC} $1"; }

timestamp() { date +"%Y-%m-%d %H:%M:%S"; }

# ============================================================
# GPU Auto-Detection & Optimal Batch Size
# ============================================================
detect_gpu_config() {
    local gpu_mem_mb
    gpu_mem_mb=$(python3 -c "
import torch
if torch.cuda.is_available():
    print(int(torch.cuda.get_device_properties(0).total_mem / 1e6))
else:
    print(0)
" 2>/dev/null || echo "0")

    local gpu_name
    gpu_name=$(python3 -c "
import torch
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
" 2>/dev/null || echo "CPU")

    info "GPU: $gpu_name (${gpu_mem_mb}MB)"

    # Set batch sizes based on GPU memory
    if [ "$gpu_mem_mb" -ge 75000 ]; then
        # A100 80GB
        BATCH_NANO=64; BATCH_SMALL=32; BATCH_MEDIUM=16; WORKERS=8
        info "Tier: HIGH (A100-class) | batch: n=$BATCH_NANO s=$BATCH_SMALL m=$BATCH_MEDIUM"
    elif [ "$gpu_mem_mb" -ge 38000 ]; then
        # A100 40GB, A10G
        BATCH_NANO=48; BATCH_SMALL=24; BATCH_MEDIUM=12; WORKERS=8
        info "Tier: HIGH (40GB-class) | batch: n=$BATCH_NANO s=$BATCH_SMALL m=$BATCH_MEDIUM"
    elif [ "$gpu_mem_mb" -ge 22000 ]; then
        # L4, 3090, 4090
        BATCH_NANO=32; BATCH_SMALL=16; BATCH_MEDIUM=8; WORKERS=6
        info "Tier: MID (24GB-class) | batch: n=$BATCH_NANO s=$BATCH_SMALL m=$BATCH_MEDIUM"
    elif [ "$gpu_mem_mb" -ge 14000 ]; then
        # T4, V100 16GB
        BATCH_NANO=16; BATCH_SMALL=8; BATCH_MEDIUM=4; WORKERS=4
        info "Tier: LOW (16GB-class) | batch: n=$BATCH_NANO s=$BATCH_SMALL m=$BATCH_MEDIUM"
    else
        # <14GB or CPU
        BATCH_NANO=8; BATCH_SMALL=4; BATCH_MEDIUM=2; WORKERS=2
        warn "Tier: MINIMAL (<14GB) | batch: n=$BATCH_NANO s=$BATCH_SMALL m=$BATCH_MEDIUM"
    fi

    export BATCH_NANO BATCH_SMALL BATCH_MEDIUM WORKERS
}

# ============================================================
# Stage: Setup
# ============================================================
stage_setup() {
    log "=== SETUP === ($(timestamp))"

    # Install Python dependencies
    log "Installing dependencies..."
    cd "$PROJECT_DIR"
    pip install -e ".[all]" 2>/dev/null || pip install -r nas_yolo/requirements.txt 2>/dev/null || true

    # Verify PyTorch + CUDA
    python3 -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available!'
print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"

    # Download COCO if not present
    if [ ! -d "$DATA_DIR/train2017" ]; then
        log "Downloading COCO 2017..."
        mkdir -p "$DATA_DIR"
        cd "$DATA_DIR"

        wget -q --show-progress http://images.cocodataset.org/zips/train2017.zip
        unzip -q train2017.zip && rm train2017.zip

        wget -q --show-progress http://images.cocodataset.org/zips/val2017.zip
        unzip -q val2017.zip && rm val2017.zip

        wget -q --show-progress http://images.cocodataset.org/annotations/annotations_trainval2017.zip
        unzip -q annotations_trainval2017.zip && rm annotations_trainval2017.zip

        cd "$PROJECT_DIR"
        log "COCO 2017 downloaded to $DATA_DIR"
    else
        log "COCO 2017 already present at $DATA_DIR"
    fi

    log "Setup complete"
}

# ============================================================
# Stage: Smoke Test
# ============================================================
stage_smoke() {
    log "=== SMOKE TEST === ($(timestamp))"
    cd "$PROJECT_DIR"

    # Validate code structure
    python3 nas_yolo/validate_code.py

    # PyTorch smoke test
    python3 -c "
import torch
import sys
sys.path.insert(0, '.')

from nas_yolo.models.nas_yolo import NASYOLO, NASYOLOWithTCL

print('\n--- Model Construction ---')
for scale in ['nano', 'small', 'medium']:
    model = NASYOLO(num_classes=80, model_scale=scale).cuda().eval()
    info = model.get_model_info()

    x = torch.randn(1, 3, 640, 640).cuda()
    with torch.no_grad():
        outputs = model(x)

    print(f'NAS-YOLO-{scale[0]}: {info[\"total_params_M\"]:.2f}M, '
          f'NAS overhead: {info[\"nas_overhead_pct\"]:.1f}%')
    del model; torch.cuda.empty_cache()

print('\n--- Training (1 step) ---')
model = NASYOLOWithTCL(num_classes=80, model_scale='nano', tcl_weight=0.5).cuda().train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

from nas_yolo.models.temporal_buffer import PseudoTemporalGenerator
x = torch.randn(2, 3, 320, 320).cuda()
targets = [
    {'boxes': torch.tensor([[50., 50., 150., 150.]]).cuda(), 'labels': torch.tensor([3]).cuda()},
    {'boxes': torch.tensor([[20., 20., 100., 100.]]).cuda(), 'labels': torch.tensor([1]).cuda()},
]

gen = PseudoTemporalGenerator(sequence_length=3)
seq = gen(x)
targets_seq = [targets] * len(seq)

outputs_seq, tcl_loss = model.forward_temporal_with_tcl(seq, targets_seq)
det_loss = sum(o['losses']['loss'] for o in outputs_seq if 'losses' in o) / len(outputs_seq)
loss = det_loss + tcl_loss

optimizer.zero_grad()
loss.backward()
optimizer.step()

print(f'Loss: {loss.item():.4f} (det: {det_loss.item():.4f}, tcl: {tcl_loss.item():.4f})')
print('Training step OK')

print('\n--- Latency Benchmark ---')
import time
for scale in ['nano', 'small']:
    model = NASYOLO(num_classes=80, model_scale=scale).cuda().eval()
    x = torch.randn(1, 3, 640, 640).cuda()

    for _ in range(10): model(x)  # warmup
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(100):
        with torch.no_grad(): model(x)
    torch.cuda.synchronize()
    ms = (time.time() - t0) / 100 * 1000
    print(f'NAS-YOLO-{scale[0]}: {ms:.1f}ms ({1000/ms:.0f} FPS)')
    del model; torch.cuda.empty_cache()

print('\nAll smoke tests passed')
"

    log "Smoke tests complete"
}

# ============================================================
# Stage: Training
# ============================================================
stage_train() {
    log "=== TRAINING === ($(timestamp))"
    cd "$PROJECT_DIR"
    mkdir -p "$RUNS_DIR"

    NUM_GPUS=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)

    # scale:config:batch_size
    TRAIN_CONFIGS=(
        "nano:nas_yolo/configs/nas_yolo_n.yaml:${BATCH_NANO}"
        "small:nas_yolo/configs/default.yaml:${BATCH_SMALL}"
        "medium:nas_yolo/configs/nas_yolo_m.yaml:${BATCH_MEDIUM}"
    )

    for entry in "${TRAIN_CONFIGS[@]}"; do
        IFS=: read -r SCALE CONFIG BATCH <<< "$entry"
        RUN_DIR="$RUNS_DIR/nas_yolo_${SCALE}"

        if [ -f "$RUN_DIR/best.pt" ]; then
            log "Skipping $SCALE (checkpoint exists at $RUN_DIR/best.pt)"
            continue
        fi

        log "Training NAS-YOLO-$SCALE ($NUM_GPUS GPU(s), batch=$BATCH)..."
        mkdir -p "$RUN_DIR"

        COMMON_ARGS=(
            -m nas_yolo.scripts.train
            --config "$CONFIG"
            --data-root "$DATA_DIR"
            --output-dir "$RUN_DIR"
            --batch-size "$BATCH"
            --workers "$WORKERS"
            --amp
        )

        if [ "$NUM_GPUS" -gt 1 ]; then
            torchrun --nproc_per_node="$NUM_GPUS" "${COMMON_ARGS[@]}" \
                2>&1 | tee "$RUN_DIR/train.log"
        else
            python3 "${COMMON_ARGS[@]}" \
                2>&1 | tee "$RUN_DIR/train.log"
        fi

        log "NAS-YOLO-$SCALE training complete -> $RUN_DIR"
    done

    log "All training complete"
}

# ============================================================
# Stage: Ablation
# ============================================================
stage_ablation() {
    log "=== ABLATION === ($(timestamp))"
    cd "$PROJECT_DIR"

    ABLATION_CONFIGS=(
        "no_temporal:nas_yolo/configs/ablation/no_temporal.yaml"
        "no_noise_gate:nas_yolo/configs/ablation/no_noise_gate.yaml"
        "no_tcl:nas_yolo/configs/ablation/no_tcl.yaml"
    )

    for entry in "${ABLATION_CONFIGS[@]}"; do
        NAME="${entry%%:*}"
        CONFIG="${entry##*:}"
        RUN_DIR="$RUNS_DIR/ablation_${NAME}"

        if [ -f "$RUN_DIR/best.pt" ]; then
            log "Skipping ablation $NAME (checkpoint exists)"
            continue
        fi

        log "Training ablation: $NAME (batch=$BATCH_NANO)..."
        mkdir -p "$RUN_DIR"

        python3 -m nas_yolo.scripts.train \
            --config "$CONFIG" \
            --data-root "$DATA_DIR" \
            --output-dir "$RUN_DIR" \
            --batch-size "$BATCH_NANO" \
            --workers "$WORKERS" \
            --amp \
            2>&1 | tee "$RUN_DIR/train.log"

        log "Ablation $NAME complete -> $RUN_DIR"
    done

    log "All ablations complete"
}

# ============================================================
# Stage: Evaluation
# ============================================================
stage_eval() {
    log "=== EVALUATION === ($(timestamp))"
    cd "$PROJECT_DIR"

    MODELS=(
        "nas_yolo_nano:$RUNS_DIR/nas_yolo_nano/best.pt:nano"
        "nas_yolo_small:$RUNS_DIR/nas_yolo_small/best.pt:small"
        "nas_yolo_medium:$RUNS_DIR/nas_yolo_medium/best.pt:medium"
    )

    for entry in "${MODELS[@]}"; do
        IFS=: read -r NAME CKPT SCALE <<< "$entry"

        if [ ! -f "$CKPT" ]; then
            warn "Checkpoint not found: $CKPT (skipping $NAME)"
            continue
        fi

        EVAL_DIR="$RUNS_DIR/eval_${NAME}"
        mkdir -p "$EVAL_DIR"

        log "Evaluating $NAME (full: mAP + corruption + BSI + latency)..."

        python3 -m nas_yolo.scripts.evaluate \
            --checkpoint "$CKPT" \
            --model-scale "$SCALE" \
            --data-root "$DATA_DIR" \
            --output-dir "$EVAL_DIR" \
            --batch-size "$BATCH_SMALL" \
            --full \
            2>&1 | tee "$EVAL_DIR/eval.log"

        log "$NAME evaluation complete -> $EVAL_DIR"
    done

    # Evaluate ablation variants
    for VARIANT in no_temporal no_noise_gate no_tcl; do
        CKPT="$RUNS_DIR/ablation_${VARIANT}/best.pt"
        if [ ! -f "$CKPT" ]; then
            warn "Ablation checkpoint not found: $CKPT"
            continue
        fi

        EVAL_DIR="$RUNS_DIR/eval_ablation_${VARIANT}"
        mkdir -p "$EVAL_DIR"

        log "Evaluating ablation: $VARIANT..."

        python3 -m nas_yolo.scripts.evaluate \
            --checkpoint "$CKPT" \
            --model-scale "nano" \
            --data-root "$DATA_DIR" \
            --output-dir "$EVAL_DIR" \
            --batch-size "$BATCH_SMALL" \
            --full \
            2>&1 | tee "$EVAL_DIR/eval.log"
    done

    # Latency benchmark
    log "Running latency benchmark..."
    python3 -m nas_yolo.scripts.benchmark \
        --all-scales --overhead-analysis \
        --output "$RUNS_DIR/benchmark_results.json" \
        2>&1 | tee "$RUNS_DIR/benchmark.log"

    log "All evaluations complete"
}

# ============================================================
# Stage: Generate Tables
# ============================================================
stage_tables() {
    log "=== GENERATE TABLES === ($(timestamp))"
    cd "$PROJECT_DIR"

    python3 nas_yolo/experiments/generate_tables.py \
        --results-dir "$RUNS_DIR" \
        --output-dir "$RUNS_DIR/tables"

    log "Tables generated at $RUNS_DIR/tables/"
}

# ============================================================
# Main
# ============================================================
log "NAS-YOLO Experiment Pipeline"
log "Stage: $STAGE | GPUs: $GPU_IDS"
log "Project: $PROJECT_DIR"
log "Data: $DATA_DIR"
log "Runs: $RUNS_DIR"
echo ""

# Auto-detect GPU config
detect_gpu_config

case "$STAGE" in
    setup)    stage_setup ;;
    smoke)    stage_smoke ;;
    train)    stage_train ;;
    ablation) stage_ablation ;;
    eval)     stage_eval ;;
    tables)   stage_tables ;;
    all)
        stage_setup
        stage_smoke
        stage_train
        stage_ablation
        stage_eval
        stage_tables
        log "Full pipeline complete! ($(timestamp))"
        ;;
    *)
        err "Unknown stage: $STAGE. Use: setup|smoke|train|ablation|eval|tables|all"
        ;;
esac
