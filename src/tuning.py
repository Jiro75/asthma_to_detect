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
#   [x] Suppress Optuna output: optuna.logging.set_verbosity(optuna.logging.WARNING)
#   [x] Load X_train, y_train from DATA_SPLITS at function entry
#   [x] Inner CV: StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
#   [x] Print best params + best AUC-ROC score per model after each study
#   [x] Return best_params: dict[model_name, dict[param, value]]
# =============================================================================

import numpy as np
import pandas as pd
import pickle
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from imblearn.pipeline import Pipeline as ImbPipeline

from src.cv_pipeline import build_cv_pipeline, preprocess_for_early_stopping
from src.preprocessing import build_preprocessor_top25
from config import (
    DATA_SPLITS, OPTUNA_TRIALS, OPTUNA_EARLY_STOP,
    CV_FOLDS, RANDOM_STATE, CV_SCORING, SCALE_POS_WEIGHT,
    SMOTE_RATIO_DEFAULT,
)

# Suppress Optuna INFO spam
optuna.logging.set_verbosity(optuna.logging.WARNING)


# =============================================================================
# Data loading helper
# =============================================================================

def _load_training_data():
    """Load X_train, y_train from persisted pickle splits."""
    x_path = os.path.join(DATA_SPLITS, "X_train.pkl")
    y_path = os.path.join(DATA_SPLITS, "y_train.pkl")

    if not os.path.exists(x_path) or not os.path.exists(y_path):
        raise FileNotFoundError(
            f"Training splits not found in '{DATA_SPLITS}'. "
            "Run splitter.py first to generate splits."
        )

    with open(x_path, "rb") as f:
        X_train = pickle.load(f)
    with open(y_path, "rb") as f:
        y_train = pickle.load(f)

    return X_train, y_train


# (Redundant functions removed)


# =============================================================================
# Objective functions — one per model family
# =============================================================================

def _xgb_objective(trial, X_train, y_train, cv):
    """XGBoost objective with early stopping inside each inner CV fold."""
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 100, 1000),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth":        trial.suggest_int("max_depth", 3, 9),
        "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-8, 10, log=True),
    }

    auc_scores = []
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_tr_fold = X_train.iloc[train_idx]
        y_tr_fold = y_train.iloc[train_idx]
        X_val_fold = X_train.iloc[val_idx]
        y_val_fold = y_train.iloc[val_idx]

        # Preprocess + SMOTE on train fold; transform val fold using cv_pipeline
        X_tr_res, y_tr_res, X_val_t = preprocess_for_early_stopping(
            X_tr_fold, y_tr_fold, X_val_fold,
            smote_ratio=SMOTE_RATIO_DEFAULT,
            preprocessor=build_preprocessor_top25()
        )

        clf = XGBClassifier(
            **params,
            scale_pos_weight=SCALE_POS_WEIGHT,
            use_label_encoder=False,
            eval_metric="logloss",
            early_stopping_rounds=OPTUNA_EARLY_STOP,
            random_state=RANDOM_STATE,
        )
        clf.fit(
            X_tr_res, y_tr_res,
            eval_set=[(X_val_t, y_val_fold)],
            verbose=False,
        )

        y_proba = clf.predict_proba(X_val_t)[:, 1]
        auc_scores.append(roc_auc_score(y_val_fold, y_proba))

    return np.mean(auc_scores)


def _lgbm_objective(trial, X_train, y_train, cv):
    """LightGBM objective with early stopping via callbacks."""
    params = {
        "num_leaves":        trial.suggest_int("num_leaves", 20, 150),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "feature_fraction":  trial.suggest_float("feature_fraction", 0.5, 1.0),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators":      trial.suggest_int("n_estimators", 100, 1000),
    }

    auc_scores = []
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_tr_fold = X_train.iloc[train_idx]
        y_tr_fold = y_train.iloc[train_idx]
        X_val_fold = X_train.iloc[val_idx]
        y_val_fold = y_train.iloc[val_idx]

        X_tr_res, y_tr_res, X_val_t = preprocess_for_early_stopping(
            X_tr_fold, y_tr_fold, X_val_fold,
            smote_ratio=SMOTE_RATIO_DEFAULT,
            preprocessor=build_preprocessor_top25()
        )

        clf = LGBMClassifier(
            **params,
            is_unbalance=True,
            random_state=RANDOM_STATE,
            verbose=-1,
        )
        clf.fit(
            X_tr_res, y_tr_res,
            eval_set=[(X_val_t, y_val_fold)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=OPTUNA_EARLY_STOP, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )

        y_proba = clf.predict_proba(X_val_t)[:, 1]
        auc_scores.append(roc_auc_score(y_val_fold, y_proba))

    return np.mean(auc_scores)


def _logreg_objective(trial, X_train, y_train, cv):
    """Logistic Regression objective — uses pipeline-based cross_val_score."""
    params = {
        "C":       trial.suggest_float("C", 1e-3, 100, log=True),
        "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
        "solver":  trial.suggest_categorical("solver", ["liblinear", "saga"]),
    }

    clf = LogisticRegression(
        **params,
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_STATE,
    )

    pipe = build_cv_pipeline(
        clf, 
        smote_ratio=SMOTE_RATIO_DEFAULT,
        preprocessor=build_preprocessor_top25()
    )
    scores = cross_val_score(
        pipe, X_train, y_train,
        cv=cv, scoring="roc_auc", n_jobs=-1,
    )
    return np.mean(scores)


def _rf_objective(trial, X_train, y_train, cv):
    """Random Forest objective — uses pipeline-based cross_val_score."""
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 100, 800),
        "max_depth":        trial.suggest_int("max_depth", 3, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "max_features":     trial.suggest_categorical("max_features", ["sqrt", "log2"]),
    }

    clf = RandomForestClassifier(
        **params,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    pipe = build_cv_pipeline(
        clf, 
        smote_ratio=SMOTE_RATIO_DEFAULT,
        preprocessor=build_preprocessor_top25()
    )
    scores = cross_val_score(
        pipe, X_train, y_train,
        cv=cv, scoring="roc_auc", n_jobs=1,  # avoid nested parallelism with RF n_jobs
    )
    return np.mean(scores)


# =============================================================================
# Main entry point
# =============================================================================

def run_tuning(cv_pipe, models: dict) -> dict:
    """
    Run an independent Optuna study for each model.

    Parameters
    ----------
    cv_pipe : imblearn Pipeline (preprocessor → smote → clf placeholder)
              — not used directly; we build top-25 pipelines internally.
    models  : dict from get_models()

    Returns
    -------
    best_params : dict[model_name, dict[param, value]]
    """
    X_train, y_train = _load_training_data()
    print(f"  Loaded training data: X={X_train.shape}, y={y_train.shape}")
    print(f"  Class distribution: {dict(y_train.value_counts())}")

    # Inner CV for Optuna objective evaluation
    inner_cv = StratifiedKFold(
        n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )

    # Map model names to their objective functions
    objective_map = {
        "XGBoost":            _xgb_objective,
        "LightGBM":           _lgbm_objective,
        "LogisticRegression":  _logreg_objective,
        "RandomForest":        _rf_objective,
    }

    best_params = {}

    for model_name in models.keys():
        print(f"\n  --- Optuna: {model_name} ({OPTUNA_TRIALS} trials) ---")

        obj_fn = objective_map[model_name]

        # Create a SEPARATE study for each model (never share studies)
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=RANDOM_STATE),
            pruner=MedianPruner(),
            study_name=f"{model_name}_tuning",
        )

        # Use default_factory pattern to avoid Python closure pitfall
        def make_objective(fn):
            return lambda trial: fn(trial, X_train, y_train, inner_cv)

        study.optimize(
            make_objective(obj_fn),
            n_trials=OPTUNA_TRIALS,
            show_progress_bar=True,
        )

        best_params[model_name] = study.best_params
        print(f"  Best AUC-ROC: {study.best_value:.4f}")
        print(f"  Best params:  {study.best_params}")

    return best_params


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.models import get_models

    models = get_models()
    best_params = run_tuning(None, models)

    print("\n" + "=" * 65)
    print("  Optuna Tuning Complete — Summary")
    print("=" * 65)
    for name, params in best_params.items():
        print(f"\n  {name}:")
        for k, v in params.items():
            print(f"    {k}: {v}")
