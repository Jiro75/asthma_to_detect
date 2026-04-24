# =============================================================================
# src/evaluate.py
# MEMBER 3 — Deadline: 30 April
# Role: Evaluation, SHAP Interpretability & Visualization Engineer
# =============================================================================
# Responsibility:
#   ONE-PASS final evaluation on the LOCKED TEST SET using τ*.
#
# ⚠️  WARNING: This function must be called EXACTLY ONCE, at the very end of
#     main.py. Do NOT call it during development, threshold tuning, or model
#     selection. The test set is locked — mixing it with training inflates
#     all reported metrics.
#
# Metrics to report (all required in the final report):
#   1. AUC-ROC        — threshold-independent, use roc_auc_score
#   2. F1-Score       — minority (asthma-positive) class only
#   3. Recall         — most clinically critical: proportion of true asthma caught
#   4. Precision
#   5. Accuracy       — reported but NOT the primary metric
#   6. Confusion Matrix — absolute TP, FP, TN, FN counts
#                         (gives clinicians exact missed-diagnosis count)
#
# Checklist:
#   [ ] Load X_test, y_test from DATA_SPLITS
#   [ ] proba = best_model.predict_proba(X_test)[:, 1]
#   [ ] y_pred = (proba >= tau_star).astype(int)
#   [ ] Compute all 6 metrics above
#   [ ] Print a clearly formatted metrics table (tabulate or manual f-string)
#   [ ] Return metrics as dict: {metric_name: value}
# =============================================================================

import os
import pandas as pd
from config import DATA_SPLITS


def evaluate_on_test(best_model, tau_star: float) -> dict:
    """
    Final one-pass evaluation on the locked test set.

    Parameters
    ----------
    best_model : fitted pipeline
    tau_star   : float — threshold from find_best_threshold()

    Returns
    -------
    metrics : dict with keys roc_auc, f1, recall, precision, accuracy,
              confusion_matrix (2×2 array)
    """
    # TODO: implement
    raise NotImplementedError("evaluate.py: evaluate_on_test() not yet implemented.")
