# ============================================================================
# NAS-YOLO Docker Environment
# ============================================================================
# Reproducible environment for ECCV 2026 experiments.
#
# Build:
#   docker build -t nasyolo -f nas_yolo/Dockerfile .
#
# Run training:
#   docker run --gpus all -v $(pwd)/data:/workspace/data -v $(pwd)/runs:/workspace/runs \
#     nasyolo make -f nas_yolo/experiments/Makefile train-full GPU=0
#
# Run evaluation:
#   docker run --gpus all -v $(pwd)/data:/workspace/data -v $(pwd)/runs:/workspace/runs \
#     nasyolo make -f nas_yolo/experiments/Makefile eval-all GPU=0
#
# Interactive:
#   docker run --gpus all -it -v $(pwd)/data:/workspace/data nasyolo bash
# ============================================================================

FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel

LABEL maintainer="NAS-YOLO Authors"
LABEL description="NAS-YOLO: Noise-Aware State Modeling for Robust Real-Time Object Detection"

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    unzip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Python dependencies
COPY nas_yolo/requirements.txt /workspace/nas_yolo/requirements.txt
RUN pip install --no-cache-dir -r /workspace/nas_yolo/requirements.txt

# Install ultralytics for baseline comparison
RUN pip install --no-cache-dir ultralytics>=8.0.0

# Copy project
COPY . /workspace/

# Install nas_yolo package
RUN pip install --no-cache-dir -e .

# Download COCO (optional — mount instead for large datasets)
# RUN bash nas_yolo/experiments/download_coco.sh

# Verify installation
RUN python -c "from nas_yolo.models import NASYOLO; import torch; \
    m = NASYOLO(num_classes=10, model_scale='nano'); \
    x = torch.randn(1,3,320,320); y = m(x); \
    print('Docker build OK:', list(y.keys()))"

# Default: run smoke test
CMD ["python", "-c", "from nas_yolo.models import NASYOLO; import torch; \
    m = NASYOLO(num_classes=80, model_scale='small'); \
    info = m.get_model_info(); \
    print(f'NAS-YOLO-s ready: {info[\"total_params_M\"]:.2f}M params')"]
