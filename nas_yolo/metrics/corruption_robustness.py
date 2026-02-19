"""
Corruption Robustness Evaluation (mAP-C)
==========================================
Following the protocol of Michaelis et al. (2019):
"Benchmarking Robustness in Object Detection: Autonomous Driving when
Winter is Coming"

mAP-C (corruption mAP) is the mean of mAP scores across all corruption
types and severity levels, providing a single robustness metric.

Additional metrics:
  - Relative Corruption Error (rCE): normalized degradation vs. baseline
  - Per-corruption breakdown
  - Severity response curves
"""

import numpy as np
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field

from .map import compute_map
from ..data.corruption import CORRUPTION_TYPES


@dataclass
class CorruptionResult:
    """Results for a single corruption type."""
    corruption_type: str
    category: str  # noise, blur, weather, digital, industrial
    severity_maps: Dict[int, float]  # severity → mAP
    mean_map: float  # mean across severities
    relative_ce: float  # relative corruption error vs. clean


@dataclass
class RobustnessReport:
    """Complete corruption robustness evaluation report."""
    clean_map: float
    map_c: float  # overall corruption mAP
    mean_rce: float  # mean relative corruption error
    per_corruption: Dict[str, CorruptionResult]
    per_category: Dict[str, float]  # category → mean mAP
    severity_curve: Dict[int, float]  # severity → mean mAP across all corruptions


class CorruptionRobustness:
    """Evaluates model robustness across visual corruptions.

    Automates the full COCO-C / ImageNet-C evaluation protocol:
      1. Evaluate on clean data → baseline mAP
      2. For each corruption type × severity level → corrupted mAP
      3. Aggregate into mAP-C, rCE, severity curves

    Args:
        corruption_types: List of corruption names to evaluate (default: all).
        severities: List of severity levels (default: [1, 2, 3, 4, 5]).
        num_classes: Number of object classes.
    """

    def __init__(
        self,
        corruption_types: Optional[List[str]] = None,
        severities: Optional[List[int]] = None,
        num_classes: int = 80,
    ):
        if corruption_types is None:
            self.corruption_types = list(CORRUPTION_TYPES.keys())
        else:
            self.corruption_types = corruption_types

        self.severities = severities or [1, 2, 3, 4, 5]
        self.num_classes = num_classes

    def evaluate(
        self,
        eval_fn: Callable,
        clean_results: Optional[Dict] = None,
    ) -> RobustnessReport:
        """Run full corruption robustness evaluation.

        Args:
            eval_fn: Function that takes (corruption_type, severity) → mAP dict.
                     Should run the model on corrupted data and return mAP metrics.
                     eval_fn(None, None) should return clean mAP.
            clean_results: Pre-computed clean evaluation results (optional).

        Returns:
            RobustnessReport with complete analysis.
        """
        # Step 1: Clean baseline
        if clean_results is None:
            clean_results = eval_fn(None, None)
        clean_map = clean_results["mAP"]

        # Step 2: Evaluate each corruption × severity
        per_corruption = {}
        all_corrupt_maps = []
        category_maps = {}

        for corruption in self.corruption_types:
            category = CORRUPTION_TYPES.get(corruption, "unknown")
            severity_maps = {}

            for severity in self.severities:
                result = eval_fn(corruption, severity)
                severity_maps[severity] = result["mAP"]
                all_corrupt_maps.append(result["mAP"])

            mean_map = np.mean(list(severity_maps.values()))

            # Relative Corruption Error
            rce = (clean_map - mean_map) / (clean_map + 1e-8) * 100

            per_corruption[corruption] = CorruptionResult(
                corruption_type=corruption,
                category=category,
                severity_maps=severity_maps,
                mean_map=float(mean_map),
                relative_ce=float(rce),
            )

            # Category aggregation
            if category not in category_maps:
                category_maps[category] = []
            category_maps[category].append(mean_map)

        # Step 3: Aggregate
        map_c = float(np.mean(all_corrupt_maps)) if all_corrupt_maps else 0.0
        mean_rce = float(np.mean([r.relative_ce for r in per_corruption.values()]))

        per_category = {
            cat: float(np.mean(maps)) for cat, maps in category_maps.items()
        }

        # Severity response curve
        severity_curve = {}
        for sev in self.severities:
            sev_maps = [
                per_corruption[c].severity_maps[sev]
                for c in self.corruption_types
                if sev in per_corruption[c].severity_maps
            ]
            severity_curve[sev] = float(np.mean(sev_maps)) if sev_maps else 0.0

        return RobustnessReport(
            clean_map=clean_map,
            map_c=map_c,
            mean_rce=mean_rce,
            per_corruption=per_corruption,
            per_category=per_category,
            severity_curve=severity_curve,
        )

    def format_report(self, report: RobustnessReport) -> str:
        """Format robustness report as a table string (for paper/logging).

        Args:
            report: RobustnessReport to format.

        Returns:
            Formatted string with tables.
        """
        lines = []
        lines.append("=" * 70)
        lines.append("Corruption Robustness Report")
        lines.append("=" * 70)
        lines.append(f"Clean mAP:     {report.clean_map:.4f}")
        lines.append(f"mAP-C:         {report.map_c:.4f}")
        lines.append(f"Mean rCE:      {report.mean_rce:.2f}%")
        lines.append("")

        # Per-category summary
        lines.append("Per-Category mAP:")
        lines.append("-" * 40)
        for cat, map_val in sorted(report.per_category.items()):
            lines.append(f"  {cat:15s}: {map_val:.4f}")
        lines.append("")

        # Severity response
        lines.append("Severity Response (mean mAP):")
        lines.append("-" * 40)
        for sev, map_val in sorted(report.severity_curve.items()):
            bar = "█" * int(map_val * 40)
            lines.append(f"  Severity {sev}: {map_val:.4f} {bar}")
        lines.append("")

        # Per-corruption detail
        lines.append("Per-Corruption Detail:")
        lines.append("-" * 70)
        header = f"{'Corruption':25s} {'Category':12s} {'mAP':>8s} {'rCE(%)':>8s}"
        lines.append(header)
        lines.append("-" * 70)

        for name in sorted(report.per_corruption.keys()):
            r = report.per_corruption[name]
            lines.append(
                f"  {r.corruption_type:23s} {r.category:12s} "
                f"{r.mean_map:8.4f} {r.relative_ce:8.2f}"
            )

        lines.append("=" * 70)
        return "\n".join(lines)


def compute_map_c(
    clean_map: float,
    corruption_maps: Dict[str, Dict[int, float]],
) -> Dict[str, float]:
    """Quick computation of mAP-C from pre-collected results.

    Args:
        clean_map: mAP on clean data.
        corruption_maps: {corruption_name: {severity: mAP}}.

    Returns:
        Dict with 'mAP_C', 'mean_rCE', and per-corruption metrics.
    """
    all_maps = []
    rces = []

    for corruption, sev_maps in corruption_maps.items():
        mean_map = np.mean(list(sev_maps.values()))
        all_maps.append(mean_map)
        rce = (clean_map - mean_map) / (clean_map + 1e-8) * 100
        rces.append(rce)

    return {
        "mAP_C": float(np.mean(all_maps)) if all_maps else 0.0,
        "mean_rCE": float(np.mean(rces)) if rces else 0.0,
        "clean_mAP": clean_map,
    }
