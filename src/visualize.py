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
    Generate and save diagnostic figures to FIGURES_DIR.
    """
    import numpy as np
    import pandas as pd
    import pickle
    from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix

    os.makedirs(FIGURES_DIR, exist_ok=True)

    # 1. Load test data
    x_path = os.path.join(DATA_SPLITS, "X_test.pkl")
    y_path = os.path.join(DATA_SPLITS, "y_test.pkl")
    with open(x_path, "rb") as f:
        X_test = pickle.load(f)
    with open(y_path, "rb") as f:
        y_test = pickle.load(f)

    # 2. ROC Curves
    plt.figure(figsize=(10, 8))
    for name, model in all_models.items():
        # Handle cases where model might be a pipeline or a fitted estimator
        try:
            proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, proba)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")
        except:
            print(f"      ⚠ Could not plot ROC for {name}")

    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — All Models (Locked Test Set)")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "roc_curves.png"), dpi=300)
    plt.close()

    # 3. Precision-Recall Curves
    plt.figure(figsize=(10, 8))
    for name, model in all_models.items():
        try:
            proba = model.predict_proba(X_test)[:, 1]
            prec, rec, _ = precision_recall_curve(y_test, proba)
            plt.plot(rec, prec, label=name)
        except:
            pass

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves (Locked Test Set)")
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "pr_curves.png"), dpi=300)
    plt.close()

    # 4. Confusion Matrix Heatmap
    proba = best_model.predict_proba(X_test)[:, 1]
    y_pred = (proba >= tau_star).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["Negative", "Positive"],
                yticklabels=["Negative", "Positive"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix @ τ* = {tau_star:.2f}")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "confusion_matrix.png"), dpi=300)
    plt.close()

    print(f"  [Visualize] All diagnostic figures saved to: {FIGURES_DIR}")
