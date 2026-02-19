# ============================================================================
# NAS-YOLO Experiment Makefile
# ============================================================================
# Quick commands for ECCV 2026 paper preparation
#
# Usage:
#   make train-all GPU=0          # Train all models
#   make eval-all GPU=0           # Evaluate all models
#   make tables                   # Generate LaTeX tables
#   make paper                    # Compile paper
#   make all GPU=0                # Full pipeline
# ============================================================================

GPU ?= 0
DATA_ROOT ?= data
PYTHON ?= python

# === Training ===
.PHONY: train-full train-nano train-ablations train-all

train-full:
	CUDA_VISIBLE_DEVICES=$(GPU) $(PYTHON) -m nas_yolo.scripts.train \
		--config nas_yolo/configs/default.yaml --temporal

train-nano:
	CUDA_VISIBLE_DEVICES=$(GPU) $(PYTHON) -m nas_yolo.scripts.train \
		--config nas_yolo/configs/nas_yolo_n.yaml

train-ablation-no-temporal:
	CUDA_VISIBLE_DEVICES=$(GPU) $(PYTHON) -m nas_yolo.scripts.train \
		--config nas_yolo/configs/ablation/no_temporal.yaml

train-ablation-no-gate:
	CUDA_VISIBLE_DEVICES=$(GPU) $(PYTHON) -m nas_yolo.scripts.train \
		--config nas_yolo/configs/ablation/no_noise_gate.yaml

train-ablation-no-tcl:
	CUDA_VISIBLE_DEVICES=$(GPU) $(PYTHON) -m nas_yolo.scripts.train \
		--config nas_yolo/configs/ablation/no_tcl.yaml

train-ablations: train-ablation-no-temporal train-ablation-no-gate train-ablation-no-tcl

train-all: train-full train-nano train-ablations

# === Multi-GPU Training ===
train-full-ddp:
	torchrun --nproc_per_node=4 -m nas_yolo.scripts.train \
		--config nas_yolo/configs/default.yaml --temporal

# === Evaluation ===
.PHONY: eval-full eval-nano eval-ablations eval-all benchmark

eval-full:
	CUDA_VISIBLE_DEVICES=$(GPU) $(PYTHON) -m nas_yolo.scripts.evaluate \
		--checkpoint runs/nas_yolo_s/best.pt --full \
		--output runs/nasyolo_s_full_results.json

eval-nano:
	CUDA_VISIBLE_DEVICES=$(GPU) $(PYTHON) -m nas_yolo.scripts.evaluate \
		--checkpoint runs/nas_yolo_n/best.pt --full \
		--output runs/nasyolo_n_results.json

eval-ablations:
	CUDA_VISIBLE_DEVICES=$(GPU) $(PYTHON) -m nas_yolo.scripts.evaluate \
		--checkpoint runs/ablation/no_temporal/best.pt --full \
		--output runs/ablation_no_temporal_results.json
	CUDA_VISIBLE_DEVICES=$(GPU) $(PYTHON) -m nas_yolo.scripts.evaluate \
		--checkpoint runs/ablation/no_noise_gate/best.pt --full \
		--output runs/ablation_no_gate_results.json
	CUDA_VISIBLE_DEVICES=$(GPU) $(PYTHON) -m nas_yolo.scripts.evaluate \
		--checkpoint runs/ablation/no_tcl/best.pt --full \
		--output runs/ablation_no_tcl_results.json

eval-all: eval-full eval-nano eval-ablations

benchmark:
	CUDA_VISIBLE_DEVICES=$(GPU) $(PYTHON) -m nas_yolo.scripts.benchmark \
		--all-scales --overhead-analysis

# === Paper Materials ===
.PHONY: tables figures paper

tables:
	$(PYTHON) -m nas_yolo.experiments.generate_tables \
		--results-dir runs --output nas_yolo/paper/tables

figures:
	$(PYTHON) -m nas_yolo.scripts.visualize --mode severity \
		--results runs/nasyolo_s_full_results.json \
		--output-dir nas_yolo/paper/figures
	$(PYTHON) -m nas_yolo.scripts.visualize --mode ablation \
		--results runs --output-dir nas_yolo/paper/figures

paper:
	cd nas_yolo/paper && pdflatex main && bibtex main && pdflatex main && pdflatex main

# === Tests ===
.PHONY: test

test:
	$(PYTHON) -m pytest nas_yolo/tests/ -v

# === Full Pipeline ===
.PHONY: all

all: train-all eval-all benchmark tables figures
	@echo "All experiments complete!"

# === Quick validation (smoke test) ===
.PHONY: smoke-test

smoke-test:
	$(PYTHON) -c "from nas_yolo.models import NASYOLO; \
		import torch; \
		m = NASYOLO(num_classes=10, model_scale='nano'); \
		x = torch.randn(1,3,320,320); \
		y = m(x); \
		print('Forward pass OK:', list(y.keys())); \
		info = m.get_model_info(); \
		print(f'Params: {info[\"total_params_M\"]:.2f}M, NAS overhead: {info[\"nas_overhead_pct\"]:.1f}%')"
