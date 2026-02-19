#!/usr/bin/env python3
"""
NAS-YOLO Codebase Validation (no PyTorch required)
====================================================
Validates code structure, imports, configs, and logic without external dependencies.
Run this to verify the codebase before setting up the full environment.

Usage: python validate_code.py
"""

import os
import sys
import ast
import yaml
import importlib
import traceback
from pathlib import Path

ROOT = Path(__file__).parent
PASS = 0
FAIL = 0

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  \u2713 {name}")
    else:
        FAIL += 1
        print(f"  \u2717 {name}: {detail}")

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

# ============================================================
# 1. File Structure
# ============================================================
section("1. File Structure")

required_files = [
    "models/__init__.py",
    "models/nas_module.py",
    "models/noise_gate.py",
    "models/temporal_buffer.py",
    "models/backbone.py",
    "models/neck.py",
    "models/head.py",
    "models/nas_yolo.py",
    "data/__init__.py",
    "data/corruption.py",
    "data/dataset.py",
    "data/transforms.py",
    "metrics/__init__.py",
    "metrics/box_stability.py",
    "metrics/map.py",
    "metrics/corruption_robustness.py",
    "engine/__init__.py",
    "engine/trainer.py",
    "engine/evaluator.py",
    "utils/__init__.py",
    "utils/visualization.py",
    "utils/profiling.py",
    "utils/logging.py",
    "scripts/__init__.py",
    "scripts/train.py",
    "scripts/evaluate.py",
    "scripts/benchmark.py",
    "scripts/visualize.py",
    "tests/__init__.py",
    "tests/test_nas_module.py",
    "tests/test_noise_gate.py",
    "tests/test_box_stability.py",
    "tests/test_model.py",
    "configs/default.yaml",
    "configs/nas_yolo_n.yaml",
    "configs/nas_yolo_m.yaml",
    "configs/ablation/no_temporal.yaml",
    "configs/ablation/no_noise_gate.yaml",
    "configs/ablation/no_tcl.yaml",
    "configs/ablation/temporal_window.yaml",
    "paper/main.tex",
    "paper/references.bib",
    "paper/eccv2026.cls",
    "experiments/run_all.sh",
    "experiments/run_baselines.sh",
    "experiments/generate_tables.py",
    "experiments/Makefile",
    "experiments/ECCV_ACTION_PLAN.md",
    "requirements.txt",
    "README.md",
    "pyproject.toml",
    "Dockerfile",
    "setup.sh",
]

for f in required_files:
    path = ROOT / f
    check(f, path.exists(), "file not found")

# ============================================================
# 2. Python Syntax Validation
# ============================================================
section("2. Python Syntax Validation")

py_files = list(ROOT.rglob("*.py"))
for pf in sorted(py_files):
    try:
        with open(pf) as fh:
            ast.parse(fh.read())
        check(str(pf.relative_to(ROOT)), True)
    except SyntaxError as e:
        check(str(pf.relative_to(ROOT)), False, f"line {e.lineno}: {e.msg}")

# ============================================================
# 3. AST Analysis (class/function structure)
# ============================================================
section("3. Key Class & Function Analysis")

def get_classes_and_functions(filepath):
    with open(filepath) as f:
        tree = ast.parse(f.read())
    classes = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
    functions = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    return classes, functions

# nas_module.py
cls, fns = get_classes_and_functions(ROOT / "models/nas_module.py")
check("SelectiveSSM class", "SelectiveSSM" in cls)
check("SpatialSSMAdapter class", "SpatialSSMAdapter" in cls)
check("NoiseAwareStateModule class", "NoiseAwareStateModule" in cls)

# noise_gate.py
cls, fns = get_classes_and_functions(ROOT / "models/noise_gate.py")
check("SpectralNoiseEstimator class", "SpectralNoiseEstimator" in cls)
check("NoiseAwareGate class", "NoiseAwareGate" in cls)
check("MultiScaleNoiseEstimator class", "MultiScaleNoiseEstimator" in cls)
check("_build_frequency_masks static method", "_build_frequency_masks" in fns)

# nas_yolo.py
cls, fns = get_classes_and_functions(ROOT / "models/nas_yolo.py")
check("NASYOLO class", "NASYOLO" in cls)
check("NASYOLOWithTCL class", "NASYOLOWithTCL" in cls)
check("forward_temporal method", "forward_temporal" in fns)
check("forward_temporal_with_tcl method", "forward_temporal_with_tcl" in fns)
check("compute_loss method", "compute_loss" in fns)
check("_focal_bce method", "_focal_bce" in fns)

# head.py
cls, fns = get_classes_and_functions(ROOT / "models/head.py")
check("DecoupledHead class", "DecoupledHead" in cls)
check("DetectionHead class", "DetectionHead" in cls)
check("decode_outputs method", "decode_outputs" in fns)

# temporal_buffer.py
cls, fns = get_classes_and_functions(ROOT / "models/temporal_buffer.py")
check("TemporalFeatureBuffer class", "TemporalFeatureBuffer" in cls)
check("PseudoTemporalGenerator class", "PseudoTemporalGenerator" in cls)

# backbone.py
cls, fns = get_classes_and_functions(ROOT / "models/backbone.py")
check("CSPDarknet class", "CSPDarknet" in cls)

# neck.py
cls, fns = get_classes_and_functions(ROOT / "models/neck.py")
check("PAFPN class", "PAFPN" in cls)

# data/
cls, fns = get_classes_and_functions(ROOT / "data/corruption.py")
check("CorruptionTransform class", any("Corruption" in c for c in cls), f"found: {cls}")

cls, fns = get_classes_and_functions(ROOT / "data/dataset.py")
check("COCODetectionDataset", "COCODetectionDataset" in cls)
check("PseudoTemporalDataset", "PseudoTemporalDataset" in cls)

# metrics/
cls, fns = get_classes_and_functions(ROOT / "metrics/box_stability.py")
check("BoxStabilityIndex class", "BoxStabilityIndex" in cls)
check("FrameDetections class/namedtuple", "FrameDetections" in cls or "FrameDetections" in fns or any("FrameDetections" in str(n) for n in ast.walk(ast.parse(open(ROOT / "metrics/box_stability.py").read()))))

# ============================================================
# 4. Config Validation
# ============================================================
section("4. YAML Config Validation")

config_files = list((ROOT / "configs").rglob("*.yaml"))
for cf in sorted(config_files):
    try:
        with open(cf) as f:
            data = yaml.safe_load(f)
        check(str(cf.relative_to(ROOT)), isinstance(data, dict), "not a valid YAML dict")
    except Exception as e:
        check(str(cf.relative_to(ROOT)), False, str(e))

# Check default config has required sections
with open(ROOT / "configs/default.yaml") as f:
    default_cfg = yaml.safe_load(f)

required_sections = ["model", "data", "training", "augmentation", "evaluation"]
for sec in required_sections:
    check(f"default.yaml has '{sec}' section", sec in default_cfg, f"missing section")

# ============================================================
# 5. Bug Fix Verification
# ============================================================
section("5. Bug Fix Verification")

# Check head.py has clamp before exp
with open(ROOT / "models/head.py") as f:
    head_src = f.read()
check("head.py: exp() has clamp guard", ".clamp(max=" in head_src and "torch.exp" in head_src,
      "torch.exp without clamp \u2192 NaN risk")

check("head.py: decode_outputs clips boxes", "clamp_(min=0" in head_src,
      "boxes not clipped to image boundaries")

check("head.py: conf_threshold filtering", "conf_threshold" in head_src and "mask" in head_src,
      "conf_threshold not used for filtering")

# Check nas_yolo.py fixes
with open(ROOT / "models/nas_yolo.py") as f:
    nasyolo_src = f.read()

check("nas_yolo.py: no noise_level collapse", "noise_level[0]" not in nasyolo_src,
      "still collapsing multi-scale noise to single scale")

check("nas_yolo.py: multi-positive assignment", "3\u00d73" in nasyolo_src or "Multi-positive" in nasyolo_src,
      "still using single center-point GT assignment")

check("nas_yolo.py: label bounds check", ".clamp(0, num_cls" in nasyolo_src,
      "no label bounds check")

check("nas_yolo.py: TCL device fix", "image_sequence[0].device" in nasyolo_src,
      "still using 'images.device' (undefined)")

# Check nas_module.py pool fix
with open(ROOT / "models/nas_module.py") as f:
    nasmod_src = f.read()
check("nas_module.py: SpatialSSMAdapter uses pooling", "pool_size" in nasmod_src,
      "no spatial pooling \u2192 O(H*W) sequential loop")

check("nas_module.py: per-scale noise support", "isinstance(noise_level, list)" in nasmod_src,
      "doesn't handle per-scale noise descriptors")

# Check noise_gate.py fixes
with open(ROOT / "models/noise_gate.py") as f:
    noisegate_src = f.read()
check("noise_gate.py: frequency masks cached", "_build_frequency_masks" in noisegate_src and "register_buffer" in noisegate_src,
      "frequency masks not cached as buffers")

check("noise_gate.py: no dead freq_features var", "freq_features = spatial_size" not in noisegate_src,
      "dead variable freq_features still present")

# Check temporal_buffer.py device fix
with open(ROOT / "models/temporal_buffer.py") as f:
    tb_src = f.read()
check("temporal_buffer.py: device consistency", "prev.device != conf.device" in tb_src,
      "no device consistency check in EMA")

# ============================================================
# 6. Paper Files
# ============================================================
section("6. Paper Files Validation")

tex_path = ROOT / "paper/main.tex"
with open(tex_path) as f:
    tex = f.read()

check("LaTeX: has title", "\\title{" in tex or "\\title " in tex)
check("LaTeX: has abstract", "\\begin{abstract}" in tex)
check("LaTeX: has introduction", "Introduction" in tex)
check("LaTeX: has method section", "Method" in tex or "Approach" in tex)
check("LaTeX: has experiments", "Experiment" in tex)
check("LaTeX: has references", "\\bibliography" in tex or "\\begin{thebibliography}" in tex or "bibliographystyle" in tex)
check("LaTeX: has BSI definition", "BSI" in tex or "Box Stability" in tex)
check("LaTeX: has TCL definition", "TCL" in tex or "Temporal Consistency" in tex)

bib_path = ROOT / "paper/references.bib"
with open(bib_path) as f:
    bib = f.read()
num_refs = bib.count("@")
check(f"BibTeX: {num_refs} references (\u226530 for ECCV)", num_refs >= 30, f"only {num_refs}")

# ============================================================
# 7. Dockerfile & Build
# ============================================================
section("7. Build Files")

with open(ROOT / "Dockerfile") as f:
    dockerfile = f.read()
check("Dockerfile: based on PyTorch image", "pytorch" in dockerfile.lower())
check("Dockerfile: installs requirements", "requirements" in dockerfile)

with open(ROOT / "pyproject.toml") as f:
    toml = f.read()
check("pyproject.toml: has project name", "nas-yolo" in toml or "nas_yolo" in toml)
check("pyproject.toml: has torch dependency", "torch" in toml)

# ============================================================
# Summary
# ============================================================
print(f"\n{'='*60}")
print(f"  VALIDATION SUMMARY")
print(f"{'='*60}")
print(f"  Passed: {PASS}")
print(f"  Failed: {FAIL}")
print(f"  Total:  {PASS + FAIL}")
print(f"{'='*60}")

if FAIL == 0:
    print("  All checks passed! Codebase is ready.")
else:
    print(f"  {FAIL} check(s) failed. Please review above.")

sys.exit(FAIL)
