"""
NAS-YOLO 아키텍처 다이어그램 생성
===================================
모델 구조 + 노벨티를 시각적으로 설명하는 그림 생성.

Usage:
    python -m nas_yolo.utils.architecture_diagram --output figures/
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os
import argparse


# ============================================================================
# 색상 팔레트
# ============================================================================
COLORS = {
    "bg":          "#0D1117",
    "card_base":   "#161B22",
    "card_novel":  "#1A2332",
    "border_base": "#30363D",
    "border_c1":   "#58A6FF",  # Temporal SSM (파랑)
    "border_c2":   "#F78166",  # Noise Gate (주황)
    "border_c3":   "#7EE787",  # TCL Loss (초록)
    "border_c4":   "#D2A8FF",  # BSI Metric (보라)
    "text":        "#E6EDF3",
    "text_dim":    "#8B949E",
    "arrow":       "#58A6FF",
    "arrow_dim":   "#30363D",
    "highlight":   "#FFA657",
}


def draw_box(ax, x, y, w, h, label, sublabel=None, color=None,
             border_color=None, fontsize=10, bold=False):
    """둥근 박스 + 라벨 그리기."""
    fc = color or COLORS["card_base"]
    ec = border_color or COLORS["border_base"]
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.1",
        facecolor=fc, edgecolor=ec, linewidth=2,
        zorder=2
    )
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    ax.text(x + w / 2, y + h / 2 + (0.08 if sublabel else 0),
            label, ha="center", va="center",
            fontsize=fontsize, fontweight=weight,
            color=COLORS["text"], zorder=3)
    if sublabel:
        ax.text(x + w / 2, y + h / 2 - 0.12,
                sublabel, ha="center", va="center",
                fontsize=fontsize - 2, color=COLORS["text_dim"], zorder=3)


def draw_arrow(ax, x1, y1, x2, y2, color=None, style="-|>", lw=1.5):
    """화살표 그리기."""
    c = color or COLORS["arrow"]
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=c, lw=lw),
                zorder=1)


def draw_novelty_badge(ax, x, y, label, color):
    """노벨티 배지 (별 아이콘 + 텍스트)."""
    badge = FancyBboxPatch(
        (x, y), len(label) * 0.065 + 0.15, 0.18,
        boxstyle="round,pad=0.05",
        facecolor=color, edgecolor="none", alpha=0.25, zorder=2
    )
    ax.add_patch(badge)
    ax.text(x + 0.08, y + 0.09, f"★ {label}",
            ha="left", va="center",
            fontsize=7, fontweight="bold",
            color=color, zorder=3)


# ============================================================================
# Figure 1: Overall Architecture
# ============================================================================
def draw_overall_architecture(save_dir):
    """NAS-YOLO 전체 파이프라인 다이어그램."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-1, 7)
    ax.axis("off")

    # 제목
    ax.text(5, 6.5, "NAS-YOLO: Overall Architecture",
            ha="center", va="center", fontsize=18, fontweight="bold",
            color=COLORS["text"])
    ax.text(5, 6.1,
            '"Frame-wise perception → State-aware perception"',
            ha="center", va="center", fontsize=10, style="italic",
            color=COLORS["text_dim"])

    # --- Main Pipeline (왼→오) ---
    # Input
    draw_box(ax, 0, 3.5, 1.2, 0.8, "Input", "H×W×3", fontsize=11)

    # Backbone
    draw_box(ax, 1.8, 3.5, 1.5, 0.8, "CSPDarknet", "Backbone", fontsize=11)

    # Neck
    draw_box(ax, 3.9, 3.5, 1.2, 0.8, "PAFPN", "Neck", fontsize=11)

    # NAS Module (핵심 — 크게)
    draw_box(ax, 5.7, 2.5, 2.4, 2.8, "", "",
             color=COLORS["card_novel"], border_color=COLORS["border_c1"])
    ax.text(6.9, 5.05, "NAS Module", ha="center", va="center",
            fontsize=13, fontweight="bold", color=COLORS["border_c1"])
    draw_novelty_badge(ax, 5.85, 4.7, "Core Contribution", COLORS["border_c1"])

    # NAS 내부: Selective SSM
    draw_box(ax, 5.95, 4.0, 1.9, 0.55, "Selective SSM",
             "Temporal State (C1)",
             border_color=COLORS["border_c1"], fontsize=9)

    # NAS 내부: Noise Gate
    draw_box(ax, 5.95, 3.2, 1.9, 0.55, "Noise-Aware Gate",
             "Spectral Estimation (C2)",
             border_color=COLORS["border_c2"], fontsize=9)

    # NAS 내부: Gated Update
    draw_box(ax, 5.95, 2.6, 1.9, 0.45, "Gated Feature Update",
             None, fontsize=9)

    # Detection Head
    draw_box(ax, 8.7, 3.5, 1.5, 0.8, "Det Head", "cls + reg + obj", fontsize=11)

    # --- Arrows (main flow) ---
    draw_arrow(ax, 1.2, 3.9, 1.8, 3.9)       # Input → Backbone
    draw_arrow(ax, 3.3, 3.9, 3.9, 3.9)        # Backbone → Neck
    draw_arrow(ax, 5.1, 3.9, 5.7, 3.9)        # Neck → NAS
    draw_arrow(ax, 8.1, 3.9, 8.7, 3.9)        # NAS → Head

    # NAS 내부 화살표
    draw_arrow(ax, 6.9, 3.95, 6.9, 3.75, color=COLORS["border_c1"], lw=1.2)
    draw_arrow(ax, 6.9, 3.15, 6.9, 3.05, color=COLORS["border_c2"], lw=1.2)

    # --- Temporal Buffer (아래) ---
    draw_box(ax, 5.95, 1.5, 1.9, 0.55, "Temporal Buffer",
             "State Propagation",
             border_color=COLORS["border_c1"], fontsize=9)
    # 순환 화살표 (SSM ↔ Buffer)
    draw_arrow(ax, 6.4, 2.6, 6.4, 2.05, color=COLORS["border_c1"], lw=1.2)
    draw_arrow(ax, 7.4, 2.05, 7.4, 2.6, color=COLORS["border_c1"], lw=1.2)
    ax.text(6.15, 2.3, "h_t", fontsize=8, color=COLORS["border_c1"], style="italic")
    ax.text(7.5, 2.3, "h_{t-1}", fontsize=8, color=COLORS["border_c1"], style="italic")

    # --- Noise Estimator (위에서 내려옴) ---
    draw_box(ax, 2.5, 1.3, 2.0, 0.6, "Spectral Noise",
             "Estimator (DCT)",
             border_color=COLORS["border_c2"], fontsize=9)
    # Input → Noise Estimator
    draw_arrow(ax, 0.6, 3.5, 0.6, 1.6, color=COLORS["border_c2"], lw=1.2)
    draw_arrow(ax, 0.6, 1.6, 2.5, 1.6, color=COLORS["border_c2"], lw=1.2)
    # Noise Estimator → Gate
    draw_arrow(ax, 4.5, 1.6, 5.3, 1.6, color=COLORS["border_c2"], lw=1.2)
    draw_arrow(ax, 5.3, 1.6, 5.3, 3.45, color=COLORS["border_c2"], lw=1.2)
    draw_arrow(ax, 5.3, 3.45, 5.95, 3.45, color=COLORS["border_c2"], lw=1.2)
    ax.text(4.6, 1.8, "noise descriptor d_n",
            fontsize=7, color=COLORS["border_c2"], style="italic")

    # --- TCL Loss (오른쪽 아래) ---
    draw_box(ax, 8.5, 1.5, 1.8, 0.6, "TCL Loss",
             "Temporal Consistency",
             border_color=COLORS["border_c3"], fontsize=9)
    draw_novelty_badge(ax, 8.6, 2.2, "C3: Loss", COLORS["border_c3"])
    draw_arrow(ax, 9.45, 3.5, 9.45, 2.1, color=COLORS["border_c3"], lw=1.2)

    # --- Multi-scale 표시 ---
    ax.text(5.4, 5.6, "P3 (80×80)", fontsize=7, color=COLORS["text_dim"])
    ax.text(5.4, 5.35, "P4 (40×40)", fontsize=7, color=COLORS["text_dim"])
    ax.text(5.4, 5.1, "P5 (20×20)", fontsize=7, color=COLORS["text_dim"])
    ax.text(5.1, 5.75, "Multi-scale:", fontsize=7, fontweight="bold",
            color=COLORS["text_dim"])

    # --- Legend ---
    legend_items = [
        ("C1: Selective SSM (Temporal)", COLORS["border_c1"]),
        ("C2: Noise-Aware Gating", COLORS["border_c2"]),
        ("C3: Temporal Consistency Loss", COLORS["border_c3"]),
        ("Existing (Backbone/Neck/Head)", COLORS["border_base"]),
    ]
    for i, (label, color) in enumerate(legend_items):
        y_pos = 0.6 - i * 0.25
        rect = FancyBboxPatch(
            (0, y_pos), 0.2, 0.15,
            boxstyle="round,pad=0.02",
            facecolor="none", edgecolor=color, linewidth=2
        )
        ax.add_patch(rect)
        ax.text(0.35, y_pos + 0.075, label,
                fontsize=8, va="center", color=COLORS["text"])

    plt.tight_layout(pad=0.5)
    path = os.path.join(save_dir, "fig1_overall_architecture.png")
    plt.savefig(path, dpi=200, facecolor=COLORS["bg"], bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return path


# ============================================================================
# Figure 2: NAS Module Detail
# ============================================================================
def draw_nas_module_detail(save_dir):
    """NAS 모듈 내부 상세 다이어그램."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])
    ax.set_xlim(-0.5, 10)
    ax.set_ylim(-0.5, 8.5)
    ax.axis("off")

    ax.text(5, 8, "NAS Module: Internal Architecture",
            ha="center", fontsize=16, fontweight="bold", color=COLORS["text"])

    # --- Left: Selective SSM Detail ---
    ax.text(2.5, 7.3, "C1: Selective State Space Model",
            ha="center", fontsize=12, fontweight="bold", color=COLORS["border_c1"])

    # Input projection
    draw_box(ax, 1.5, 6.4, 2, 0.5, "Input Projection",
             "x → (z, x_ssm)", fontsize=9, border_color=COLORS["border_c1"])

    # Conv1d
    draw_box(ax, 1.5, 5.6, 2, 0.5, "Depthwise Conv1d",
             f"kernel={'{d_conv}'}", fontsize=9, border_color=COLORS["border_c1"])

    # SSM Recurrence
    draw_box(ax, 1.5, 4.4, 2, 0.8, "SSM Recurrence",
             "h_t = Āh_{t-1} + B̄x_t\ny_t = Ch_t",
             fontsize=9, border_color=COLORS["border_c1"])

    # Selective params
    draw_box(ax, 0, 3.4, 1.3, 0.5, "Δ(x)", "dt_proj",
             fontsize=8, border_color=COLORS["border_c1"])
    draw_box(ax, 1.5, 3.4, 1.3, 0.5, "B(x)", "x_proj",
             fontsize=8, border_color=COLORS["border_c1"])
    draw_box(ax, 3.0, 3.4, 1.3, 0.5, "C(x)", "x_proj",
             fontsize=8, border_color=COLORS["border_c1"])

    ax.text(2.5, 3.1, "Input-dependent (Selective)",
            ha="center", fontsize=7, style="italic", color=COLORS["border_c1"])

    # Output
    draw_box(ax, 1.5, 2.4, 2, 0.5, "Output Projection",
             "y * SiLU(z)", fontsize=9, border_color=COLORS["border_c1"])

    # SSM arrows
    draw_arrow(ax, 2.5, 6.4, 2.5, 6.1, color=COLORS["border_c1"])
    draw_arrow(ax, 2.5, 5.6, 2.5, 5.2, color=COLORS["border_c1"])
    draw_arrow(ax, 2.5, 4.4, 2.5, 3.9, color=COLORS["border_c1"])
    draw_arrow(ax, 2.5, 3.4, 2.5, 2.9, color=COLORS["border_c1"])

    # --- Right: Noise-Aware Gate Detail ---
    ax.text(7.5, 7.3, "C2: Noise-Aware Gating",
            ha="center", fontsize=12, fontweight="bold", color=COLORS["border_c2"])

    # Spectral analysis
    draw_box(ax, 6.5, 6.4, 2, 0.5, "Spatial Downscale",
             "640→16×16", fontsize=9, border_color=COLORS["border_c2"])

    draw_box(ax, 6.5, 5.6, 2, 0.5, "DCT Transform",
             "Frequency Domain", fontsize=9, border_color=COLORS["border_c2"])

    draw_box(ax, 6.5, 4.7, 2, 0.5, "Band Analysis",
             "Low / Mid / High freq", fontsize=9, border_color=COLORS["border_c2"])

    draw_box(ax, 6.5, 3.8, 2, 0.5, "Noise Descriptor",
             "d_n ∈ R^32", fontsize=9, border_color=COLORS["border_c2"])

    draw_box(ax, 6.5, 2.9, 2, 0.5, "Gate Generator",
             "α = σ(W·d_n)", fontsize=9, border_color=COLORS["border_c2"])

    # Gate arrows
    draw_arrow(ax, 7.5, 6.4, 7.5, 6.1, color=COLORS["border_c2"])
    draw_arrow(ax, 7.5, 5.6, 7.5, 5.2, color=COLORS["border_c2"])
    draw_arrow(ax, 7.5, 4.7, 7.5, 4.4, color=COLORS["border_c2"])
    draw_arrow(ax, 7.5, 3.8, 7.5, 3.4, color=COLORS["border_c2"])

    # --- Gated Update (하단) ---
    ax.text(5, 2.1, "Gated Feature Update",
            ha="center", fontsize=12, fontweight="bold", color=COLORS["highlight"])

    draw_box(ax, 3, 1.2, 4, 0.65,
             "f_out = α · f_temporal + (1-α) · f_current",
             None,
             border_color=COLORS["highlight"], fontsize=10)

    # Connect SSM output → gated update
    draw_arrow(ax, 2.5, 2.4, 2.5, 1.85, color=COLORS["border_c1"])
    draw_arrow(ax, 2.5, 1.55, 3, 1.55, color=COLORS["border_c1"])
    ax.text(2.6, 2.0, "f_temporal", fontsize=7, color=COLORS["border_c1"], style="italic")

    # Connect gate → gated update
    draw_arrow(ax, 7.5, 2.9, 7.5, 1.55, color=COLORS["border_c2"])
    draw_arrow(ax, 7.0, 1.55, 7.0, 1.55, color=COLORS["border_c2"])
    ax.text(7.6, 2.4, "α (gate)", fontsize=7, color=COLORS["border_c2"], style="italic")

    # --- Novelty Insight (하단) ---
    insight_text = (
        "Key Insight: Clean frame → α≈0 (use current features)\n"
        "             Noisy frame → α≈1 (rely on temporal state)"
    )
    ax.text(5, 0.4, insight_text,
            ha="center", fontsize=9, fontfamily="monospace",
            color=COLORS["highlight"],
            bbox=dict(boxstyle="round,pad=0.3", facecolor=COLORS["card_novel"],
                      edgecolor=COLORS["highlight"], alpha=0.5))

    plt.tight_layout(pad=0.5)
    path = os.path.join(save_dir, "fig2_nas_module_detail.png")
    plt.savefig(path, dpi=200, facecolor=COLORS["bg"], bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return path


# ============================================================================
# Figure 3: Novelty Summary
# ============================================================================
def draw_novelty_summary(save_dir):
    """NAS-YOLO 4대 노벨티 요약 다이어그램."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.patch.set_facecolor(COLORS["bg"])
    fig.suptitle("NAS-YOLO: Four Key Contributions",
                 fontsize=18, fontweight="bold", color=COLORS["text"], y=0.98)

    contributions = [
        {
            "title": "C1: Selective SSM for Detection",
            "color": COLORS["border_c1"],
            "points": [
                "Mamba-inspired temporal state modeling",
                "Adapted for 2D spatial feature maps",
                "Input-dependent Δ, B, C parameters",
                "< 1ms overhead per frame",
                "State carries across frames (no raw feature storage)",
            ],
            "vs_prior": "Prior: Per-frame detection (no temporal context)\nOurs: State-aware detection with selective memory",
        },
        {
            "title": "C2: Noise-Aware Gating",
            "color": COLORS["border_c2"],
            "points": [
                "DCT-based spectral noise estimation",
                "No noise labels needed (self-supervised)",
                "Adaptive: clean→current, noisy→temporal",
                "3 frequency bands (low/mid/high)",
                "Lightweight: ~0.1M extra parameters",
            ],
            "vs_prior": "Prior: Fixed temporal fusion (all frames equal)\nOurs: Noise-conditioned adaptive gating",
        },
        {
            "title": "C3: Temporal Consistency Loss (TCL)",
            "color": COLORS["border_c3"],
            "points": [
                "Penalizes box jitter across frames",
                "EMA-smoothed target for stability",
                "Works with pseudo-temporal sequences",
                "No video data required for training",
                "Complementary to detection loss",
            ],
            "vs_prior": "Prior: Only per-frame detection loss\nOurs: + Cross-frame consistency regularization",
        },
        {
            "title": "C4: Box Stability Index (BSI)",
            "color": COLORS["border_c4"],
            "points": [
                "New evaluation metric for temporal detection",
                "Measures: position jitter, size jitter, flicker",
                "Weighted: IoU × score × id consistency",
                "Hungarian matching across frames",
                "BSI ∈ [0, 1], higher = more stable",
            ],
            "vs_prior": "Prior: Only mAP (single-frame accuracy)\nOurs: + BSI (temporal stability measurement)",
        },
    ]

    for ax, cont in zip(axes.flat, contributions):
        ax.set_facecolor(COLORS["bg"])
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 7)
        ax.axis("off")

        # 제목
        ax.text(5, 6.5, cont["title"],
                ha="center", fontsize=13, fontweight="bold",
                color=cont["color"])

        # 핵심 포인트
        for i, point in enumerate(cont["points"]):
            ax.text(0.5, 5.3 - i * 0.7, f"  •  {point}",
                    fontsize=9, color=COLORS["text"])

        # vs Prior
        box = FancyBboxPatch(
            (0.3, 0.3), 9.4, 1.4,
            boxstyle="round,pad=0.15",
            facecolor=COLORS["card_novel"],
            edgecolor=cont["color"], linewidth=1.5, alpha=0.6
        )
        ax.add_patch(box)
        ax.text(5, 1.0, cont["vs_prior"],
                ha="center", va="center",
                fontsize=8, fontfamily="monospace",
                color=cont["color"])

    plt.tight_layout(pad=1.5)
    path = os.path.join(save_dir, "fig3_novelty_summary.png")
    plt.savefig(path, dpi=200, facecolor=COLORS["bg"], bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return path


# ============================================================================
# Main
# ============================================================================
def generate_all(save_dir="figures"):
    """모든 아키텍처 다이어그램 생성."""
    os.makedirs(save_dir, exist_ok=True)
    print("NAS-YOLO 아키텍처 다이어그램 생성 중...\n")

    paths = []
    paths.append(draw_overall_architecture(save_dir))
    paths.append(draw_nas_module_detail(save_dir))
    paths.append(draw_novelty_summary(save_dir))

    print(f"\n완료! {len(paths)}개 다이어그램 → {save_dir}/")
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NAS-YOLO 아키텍처 다이어그램 생성")
    parser.add_argument("--output", default="figures", help="저장 디렉토리")
    args = parser.parse_args()
    generate_all(args.output)
