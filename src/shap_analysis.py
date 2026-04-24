# =============================================================================
# src/shap_analysis.py
# MEMBER 3 — Deadline: 30 April
# Role: Evaluation, SHAP Interpretability & Visualization Engineer
# =============================================================================
# Responsibility:
#   Global + local SHAP explanations for the primary XGBoost model.
#   SHAP is a required deliverable under FDA SaMD guidance and EU AI Act —
#   it is NOT optional.
#
# ⚠️  CRITICAL — run SHAP on TRANSFORMED features (not raw):
#     X_test_transformed = best_model['preprocessor'].transform(X_test)
#
# ⚠️  CRITICAL — recover feature names after ColumnTransformer:
#     feature_names = best_model['preprocessor'].get_feature_names_out()
#     Pass to SHAP as: shap.Explanation(..., feature_names=feature_names)
#
# Required output files (FIGURES_DIR):
#   shap_beeswarm.png       — Global beeswarm: which features drive predictions
#                             and in which direction
#   shap_bar.png            — Mean |SHAP| per feature (global importance ranking)
#   shap_waterfall_tp.png   — Local waterfall for a True Positive sample
#   shap_waterfall_fn.png   — Local waterfall for a False Negative (missed dx)
#   shap_waterfall_fp.png   — Local waterfall for a False Positive
#
# SHAP explainer:
#   shap.TreeExplainer(best_model['clf'])  — for XGBoost/LightGBM
#   shap.KernelExplainer(...)              — fallback for LogReg/RF
#
# Checklist:
#   [ ] Load X_test, y_test from DATA_SPLITS
#   [ ] Transform: X_test_transformed = best_model['preprocessor'].transform(X_test)
#   [ ] Get feature names: best_model['preprocessor'].get_feature_names_out()
#   [ ] Instantiate shap.TreeExplainer on the clf step
#   [ ] Compute shap_values = explainer(X_test_transformed)
#   [ ] Identify TP, FN, FP sample indices from test predictions at tau_star
#   [ ] Save all 5 figures at 300 DPI
#   [ ] Print top-5 features by mean |SHAP|
# =============================================================================

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import FIGURES_DIR, DATA_SPLITS


def run_shap(best_model, tau_star: float) -> None:
    """
    Compute SHAP values and save global + local explanation plots.

    Parameters
    ----------
    best_model : fitted imblearn Pipeline (preprocessor → smote → clf)
    tau_star   : float — threshold for identifying TP/FN/FP cases
    """
    # TODO: implement
    raise NotImplementedError("shap_analysis.py: run_shap() not yet implemented.")
