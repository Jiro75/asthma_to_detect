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
    # 1. Load test data
    x_path = os.path.join(DATA_SPLITS, "X_test.pkl")
    y_path = os.path.join(DATA_SPLITS, "y_test.pkl")

    if not os.path.exists(x_path) or not os.path.exists(y_path):
        raise FileNotFoundError(f"Test data not found in {DATA_SPLITS}.")

    import pickle
    with open(x_path, "rb") as f:
        X_test = pickle.load(f)
    with open(y_path, "rb") as f:
        y_test = pickle.load(f)

    # 2. Get probabilities & predictions
    proba = best_model.predict_proba(X_test)[:, 1]
    y_pred = (proba >= tau_star).astype(int)

    # 3. Compute metrics
    from sklearn.metrics import (roc_auc_score, f1_score, recall_score, 
                             precision_score, accuracy_score, confusion_matrix)

    metrics = {
        "roc_auc": roc_auc_score(y_test, proba),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
    }

    # 4. Print table
    print("\n  [Evaluation] Results on Locked Test Set (τ* = {:.2f}):".format(tau_star))
    print("  " + "-" * 45)
    for m, v in metrics.items():
        if m != "confusion_matrix":
            print(f"  {m:<15} : {v:.4f}")
    
    return metrics
