# =============================================================================
# src/tuning.py
# MEMBER 2 — Deadline: 28 April
# Role: Model Training & Hyperparameter Optimization Engineer
# =============================================================================
# Responsibility:
#   Bayesian hyperparameter search via Optuna. One independent study per model.
#   Objective: MAXIMISE AUC-ROC on inner stratified 5-fold CV.
#   XGBoost and LightGBM get early stopping (50 rounds) inside each trial.
#
# Optuna settings (apply to ALL studies):
#   sampler : TPESampler(seed=RANDOM_STATE)
#   pruner  : MedianPruner()
#   trials  : OPTUNA_TRIALS (100) per model
#   direction: "maximize"
#
# ⚠️  Use a SEPARATE optuna.create_study() per model. Never share studies.
# ⚠️  Objective must use scoring='roc_auc', NOT accuracy.
#
# Search spaces:
#   XGBoost:
#     n_estimators        : int  [100, 1000]
#     learning_rate       : float [0.01, 0.3]  log=True
#     max_depth           : int  [3, 9]
#     subsample           : float [0.6, 1.0]
#     colsample_bytree    : float [0.6, 1.0]
#     reg_alpha           : float [1e-8, 10]   log=True
#     reg_lambda          : float [1e-8, 10]   log=True
#
#   LightGBM:
#     num_leaves          : int  [20, 150]
#     min_child_samples   : int  [5, 50]
#     feature_fraction    : float [0.5, 1.0]
#     learning_rate       : float [0.01, 0.3]  log=True
#     n_estimators        : int  [100, 1000]
#
#   Logistic Regression:
#     C                   : float [1e-3, 100]  log=True
#     penalty             : categorical ['l1', 'l2']
#     solver              : categorical ['liblinear', 'saga']
#
#   Random Forest:
#     n_estimators        : int  [100, 800]
#     max_depth           : int  [3, 20]
#     min_samples_leaf    : int  [1, 20]
#     max_features        : categorical ['sqrt', 'log2']
#
# Early stopping (XGB / LGBM only):
#   Pass eval_set=[(X_val_fold, y_val_fold)] to fit() inside each trial.
#   early_stopping_rounds = OPTUNA_EARLY_STOP (50)
#   Note: this is per-trial, not per-study.
#
# Checklist:
#   [ ] Suppress Optuna output: optuna.logging.set_verbosity(optuna.logging.WARNING)
#   [ ] Load X_train, y_train from DATA_SPLITS at function entry
#   [ ] Inner CV: StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
#   [ ] Print best params + best AUC-ROC score per model after each study
#   [ ] Return best_params: dict[model_name, dict[param, value]]
# =============================================================================

import pandas as pd
import os
from config import (DATA_SPLITS, OPTUNA_TRIALS, OPTUNA_EARLY_STOP,
                    CV_FOLDS, RANDOM_STATE, CV_SCORING)


def run_tuning(cv_pipe, models: dict) -> dict:
    """
    Run an independent Optuna study for each model.

    Parameters
    ----------
    cv_pipe : imblearn Pipeline (preprocessor → smote → clf placeholder)
    models  : dict from get_models()

    Returns
    -------
    best_params : dict[model_name, dict[param, value]]
    """
    # TODO: implement
    raise NotImplementedError("tuning.py: run_tuning() not yet implemented.")
