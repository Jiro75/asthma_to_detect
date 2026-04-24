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
    # TODO: implement
    raise NotImplementedError("threshold.py: find_best_threshold() not yet implemented.")
