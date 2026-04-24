# =============================================================================
# src/cross_validate.py
# MEMBER 2 — Deadline: 28 April
# Role: Model Training & Hyperparameter Optimization Engineer
# =============================================================================
# Responsibility:
#   Run stratified 5-fold CV for ALL 4 models using their tuned best_params.
#   This output is the MODEL SELECTION CRITERION and a required table in the
#   final report.
#
# ⚠️  Optimise on AUC-ROC (CV_SCORING = 'roc_auc'), NOT accuracy.
#     A model predicting all-negatives achieves 94.8% accuracy — useless.
# ⚠️  After CV, verify at least one fold shows non-zero True Positives per
#     model. If a model always predicts 0, the class-weight settings failed.
#
# CV settings:
#   StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
#   Metrics per fold: AUC-ROC, F1 (minority class), Recall, Precision, Accuracy
#
# Checklist:
#   [ ] Load X_train, y_train from DATA_SPLITS
#   [ ] Apply best_params to each model via pipe.set_params(clf=model, **params)
#   [ ] Compute cross_val_score or manual fold loop for all 5 metrics
#   [ ] Print table: mean ± std per metric, per model
#   [ ] Return cv_results: dict[model_name, dict[metric, float (mean)]]
#       Include 'best_model_name' key = name with highest mean AUC-ROC
# =============================================================================

import pandas as pd
import os
from config import DATA_SPLITS, CV_FOLDS, RANDOM_STATE, CV_SCORING


def run_cross_validation(cv_pipe, models: dict, best_params: dict) -> dict:
    """
    Stratified 5-fold CV on all 4 models with tuned hyperparameters.

    Parameters
    ----------
    cv_pipe     : imblearn Pipeline (preprocessor → smote → clf placeholder)
    models      : dict from get_models()
    best_params : dict from run_tuning()

    Returns
    -------
    cv_results : dict[model_name, dict[metric, mean_score]]
                 Includes 'best_model_name' key.
    """
    # TODO: implement
    raise NotImplementedError("cross_validate.py: run_cross_validation() not yet implemented.")
