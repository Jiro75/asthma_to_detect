# =============================================================================
# src/threshold.py
# MEMBER 3 — Deadline: 30 April
# Role: Evaluation, SHAP Interpretability & Visualization Engineer
# =============================================================================
# Responsibility:
#   Calibrate the decision threshold on the VALIDATION SET ONLY.
#   Final test-set evaluation uses τ* found here — never re-calibrate on test.
#
# Operating points to find:
#   τ*       = argmax F1     on validation set  (primary — used in evaluate.py)
#   τ_recall = argmax Recall on validation set  (secondary — for maximum sensitivity)
#
# Sweep:
#   τ ∈ [THRESHOLD_MIN, THRESHOLD_MAX] in steps of THRESHOLD_STEP
#   At each τ: ŷ = 1[p ≥ τ], compute F1-score and Recall on X_val
#
# ⚠️  Load probabilities from the BEST MODEL only (MODEL_PATH).
# ⚠️  Do NOT use X_test at any point in this file.
#
# Checklist:
#   [ ] Load best_model from MODEL_PATH (joblib.load)
#   [ ] Load X_val, y_val from DATA_SPLITS
#   [ ] Get predict_proba scores: proba = best_model.predict_proba(X_val)[:, 1]
#   [ ] Sweep τ, record F1 and Recall at each step
#   [ ] τ*       = τ with highest F1
#   [ ] τ_recall = τ with highest Recall
#   [ ] Print: τ*, F1 at τ*, Recall at τ*, τ_recall
#   [ ] Return τ* (float) — caller passes this to evaluate.py
# =============================================================================

import joblib
import pandas as pd
import os
import pickle
import numpy as np
from config import MODEL_PATH, DATA_SPLITS, THRESHOLD_MIN, THRESHOLD_MAX, THRESHOLD_STEP


def find_best_threshold(best_model=None) -> float:
    """
    Sweep τ on validation set. Returns τ* = argmax F1.

    Parameters
    ----------
    best_model : fitted pipeline (optional — loaded from MODEL_PATH if None)

    Returns
    -------
    tau_star : float
    """
    if best_model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. "
                                    "Run training first.")
        best_model = joblib.load(MODEL_PATH)

    # 1. Load validation data
    x_path = os.path.join(DATA_SPLITS, "X_val.pkl")
    y_path = os.path.join(DATA_SPLITS, "y_val.pkl")
    
    if not os.path.exists(x_path) or not os.path.exists(y_path):
        raise FileNotFoundError(f"Validation data not found in {DATA_SPLITS}. "
                                "Run splitter first.")

    with open(x_path, "rb") as f:
        X_val = pickle.load(f)
    with open(y_path, "rb") as f:
        y_val = pickle.load(f)

    # 2. Get probabilities
    proba = best_model.predict_proba(X_val)[:, 1]

    # 3. Sweep thresholds
    thresholds = np.arange(THRESHOLD_MIN, THRESHOLD_MAX + THRESHOLD_STEP, THRESHOLD_STEP)
    best_f1 = -1.0
    tau_star = 0.5
    tau_recall = 0.5
    max_recall = -1.0

    print(f"\n  [Threshold] Sweeping tau in [{THRESHOLD_MIN}, {THRESHOLD_MAX}] ...")

    from sklearn.metrics import f1_score, recall_score

    for tau in thresholds:
        y_pred = (proba >= tau).astype(int)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        rec = recall_score(y_val, y_pred, zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            tau_star = tau
        
        if rec > max_recall:
            max_recall = rec
            tau_recall = tau

    print(f"  [Threshold] tau*       = {tau_star:.2f} (Best F1: {best_f1:.4f})")
    print(f"  [Threshold] tau_recall = {tau_recall:.2f} (Best Recall: {max_recall:.4f})")

    return float(tau_star)
