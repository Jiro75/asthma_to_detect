# =============================================================================
# src/visualize.py
# MEMBER 3 — Deadline: 30 April
# Role: Evaluation, SHAP Interpretability & Visualization Engineer
# =============================================================================
# Responsibility:
#   Generate the full visualization suite for the Results section of the report.
#   All figures saved to FIGURES_DIR at 300 DPI using plt.savefig(..., dpi=300,
#   bbox_inches='tight').
#
# Required output files:
#   roc_curves.png        — ROC for all 4 models overlaid, AUC in legend
#   pr_curves.png         — Precision-Recall for all 4 models (better than ROC
#                           under severe class imbalance)
#   confusion_matrix.png  — Heatmap for best model at τ*
#   threshold_sweep.png   — F1 and Recall vs τ; τ* annotated as vertical dashed
#                           line with label
#   class_distribution.png— Before/after SMOTE bar chart (coordinate counts
#                           with Member 1)
#
# ⚠️  Axes must be labelled. Legends must be included. Titles must be set.
# ⚠️  plt.tight_layout() before every savefig.
# ⚠️  300 DPI is non-negotiable for a medical ML report.
#
# Checklist:
#   [ ] Load X_test, y_test from DATA_SPLITS for ROC / PR / confusion matrix
#   [ ] ROC: plot all 4 models on one figure; use roc_curve() per model
#   [ ] PR:  plot all 4 models on one figure; use precision_recall_curve()
#   [ ] Confusion matrix: sns.heatmap with annot=True, fmt='d'
#   [ ] Threshold sweep: dual-axis or overlaid line plot with τ* dashed line
#   [ ] Print path of each saved figure
# =============================================================================

import os
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for server/script use
import matplotlib.pyplot as plt
import seaborn as sns
from config import FIGURES_DIR, DATA_SPLITS


def generate_all_plots(best_model, all_models: dict, best_params: dict,
                       cv_results: dict, tau_star: float) -> None:
    """
    Generate and save all 5 diagnostic figures to FIGURES_DIR.

    Parameters
    ----------
    best_model  : fitted pipeline (for confusion matrix)
    all_models  : dict of all 4 fitted pipelines (for ROC / PR comparison)
    best_params : dict from run_tuning()
    cv_results  : dict from run_cross_validation()
    tau_star    : float from find_best_threshold()
    """
    # TODO: implement
    raise NotImplementedError("visualize.py: generate_all_plots() not yet implemented.")
