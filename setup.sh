#!/bin/bash
# ============================================================================
# NAS-YOLO One-Click Setup Script
# ============================================================================
# Sets up the complete environment for training and evaluation.
#
# Usage:
#   bash nas_yolo/setup.sh [--with-coco] [--with-baselines]
#
# Options:
#   --with-coco       Download COCO 2017 dataset (~20GB)
#   --with-baselines  Install ultralytics for baseline comparison
# ============================================================================

set -e

WITH_COCO=false
WITH_BASELINES=false

for arg in "$@"; do
    case $arg in
        --with-coco) WITH_COCO=true ;;
        --with-baselines) WITH_BASELINES=true ;;
    esac
done

echo "=============================================="
echo "NAS-YOLO Environment Setup"
echo "=============================================="

# --- Step 1: Check Python ---
echo ""
echo "[1/5] Checking Python..."
python3 --version || { echo "Python 3.9+ required"; exit 1; }

# --- Step 2: Create virtual environment ---
echo ""
echo "[2/5] Creating virtual environment..."
if [ ! -d "venv_nasyolo" ]; then
    python3 -m venv venv_nasyolo
    echo "  Created: venv_nasyolo/"
else
    echo "  Exists: venv_nasyolo/"
fi
source venv_nasyolo/bin/activate

# --- Step 3: Install dependencies ---
echo ""
echo "[3/5] Installing dependencies..."
pip install --upgrade pip
pip install -r nas_yolo/requirements.txt

if [ "$WITH_BASELINES" = true ]; then
    echo "  Installing ultralytics for baselines..."
    pip install ultralytics>=8.0.0
fi

pip install -e .

# --- Step 4: Download COCO (optional) ---
if [ "$WITH_COCO" = true ]; then
    echo ""
    echo "[4/5] Downloading COCO 2017..."
    mkdir -p data/coco

    if [ ! -d "data/coco/train2017" ]; then
        echo "  Downloading train2017 images (~18GB)..."
        wget -q http://images.cocodataset.org/zips/train2017.zip -O /tmp/train2017.zip
        unzip -q /tmp/train2017.zip -d data/coco/
        rm /tmp/train2017.zip
    fi

    if [ ! -d "data/coco/val2017" ]; then
        echo "  Downloading val2017 images (~1GB)..."
        wget -q http://images.cocodataset.org/zips/val2017.zip -O /tmp/val2017.zip
        unzip -q /tmp/val2017.zip -d data/coco/
        rm /tmp/val2017.zip
    fi

    if [ ! -d "data/coco/annotations" ]; then
        echo "  Downloading annotations..."
        wget -q http://images.cocodataset.org/annotations/annotations_trainval2017.zip -O /tmp/annotations.zip
        unzip -q /tmp/annotations.zip -d data/coco/
        rm /tmp/annotations.zip
    fi

    echo "  COCO 2017 ready at data/coco/"
else
    echo ""
    echo "[4/5] Skipping COCO download (use --with-coco to download)"
fi

# --- Step 5: Verify ---
echo ""
echo "[5/5] Verifying installation..."
python3 -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')

from nas_yolo.models import NASYOLO
m = NASYOLO(num_classes=80, model_scale='small')
info = m.get_model_info()
print(f'  NAS-YOLO-s: {info[\"total_params_M\"]:.2f}M params')
print(f'  NAS overhead: {info[\"nas_overhead_pct\"]:.1f}%')

# Quick forward pass
x = torch.randn(1, 3, 640, 640)
with torch.no_grad():
    y = m(x)
print(f'  Forward pass: OK ({len(y[\"cls_preds\"])} scales)')
print()
print('  Setup complete! Ready to train.')
"

echo ""
echo "=============================================="
echo "Setup complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "  source venv_nasyolo/bin/activate"
echo "  make -f nas_yolo/experiments/Makefile smoke-test"
echo "  make -f nas_yolo/experiments/Makefile train-full GPU=0"
echo ""
