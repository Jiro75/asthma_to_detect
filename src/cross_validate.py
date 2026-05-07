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
#   [x] Load X_train, y_train from DATA_SPLITS
#   [x] Apply best_params to each model via classifier reconstruction
#   [x] Compute cross_val_score or manual fold loop for all metrics
#   [x] Print table: mean ± std per metric, per model
#   [x] Return cv_results: dict[model_name, dict[metric, float (mean)]]
#       Include 'best_model_name' key = name with highest mean AUC-ROC
# =============================================================================

import numpy as np
import pandas as pd
import pickle
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, f1_score, recall_score, precision_score,
    accuracy_score, confusion_matrix,
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from imblearn.pipeline import Pipeline as ImbPipeline

from src.cv_pipeline import build_cv_pipeline
from src.preprocessing import build_preprocessor_top25
from config import (
    DATA_SPLITS, CV_FOLDS, RANDOM_STATE, CV_SCORING,
    SCALE_POS_WEIGHT, SMOTE_RATIO_DEFAULT,
)


def _load_training_data():
    """Load X_train, y_train from persisted pickle splits."""
    x_path = os.path.join(DATA_SPLITS, "X_train.pkl")
    y_path = os.path.join(DATA_SPLITS, "y_train.pkl")

    with open(x_path, "rb") as f:
        X_train = pickle.load(f)
    with open(y_path, "rb") as f:
        y_train = pickle.load(f)

    return X_train, y_train


def _reconstruct_classifier(model_name: str, best_params: dict):
    """
    Reconstruct a fresh classifier with base imbalance config + tuned params.
    Returns an unfitted estimator.
    """
    params = best_params.get(model_name, {})

    if model_name == "XGBoost":
        return XGBClassifier(
            **params,
            scale_pos_weight=SCALE_POS_WEIGHT,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
        )
    elif model_name == "LightGBM":
        return LGBMClassifier(
            **params,
            is_unbalance=True,
            random_state=RANDOM_STATE,
            verbose=-1,
        )
    elif model_name == "LogisticRegression":
        return LogisticRegression(
            **params,
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_STATE,
        )
    elif model_name == "RandomForest":
        return RandomForestClassifier(
            **params,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


# (Redundant _build_pipeline removed)


def run_cross_validation(cv_pipe, models: dict, best_params: dict) -> dict:
    """
    Stratified 5-fold CV on all 4 models with tuned hyperparameters.

    Parameters
    ----------
    cv_pipe     : imblearn Pipeline (preprocessor → smote → clf placeholder)
                  — not used directly; we build top-25 pipelines internally.
    models      : dict from get_models()
    best_params : dict from run_tuning()

    Returns
    -------
    cv_results : dict[model_name, dict[metric, mean_score]]
                 Includes 'best_model_name' key.
    """
    X_train, y_train = _load_training_data()
    print(f"  Loaded training data: X={X_train.shape}, y={y_train.shape}")

    cv = StratifiedKFold(
        n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )

    cv_results = {}
    best_auc = -1.0
    best_model_name = None

    for model_name in models.keys():
        print(f"\n  --- CV: {model_name} ---")

        fold_metrics = {
            "auc_roc": [], "f1": [], "recall": [],
            "precision": [], "accuracy": [],
        }
        all_tp = 0  # track True Positives across folds

        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
            X_tr_fold = X_train.iloc[train_idx]
            y_tr_fold = y_train.iloc[train_idx]
            X_val_fold = X_train.iloc[val_idx]
            y_val_fold = y_train.iloc[val_idx]

            # Reconstruct fresh classifier with tuned params
            clf = _reconstruct_classifier(model_name, best_params)
            pipe = build_cv_pipeline(
                clf,
                smote_ratio=SMOTE_RATIO_DEFAULT,
                preprocessor=build_preprocessor_top25()
            )

            # Fit on training fold
            pipe.fit(X_tr_fold, y_tr_fold)

            # Predict on validation fold
            y_pred = pipe.predict(X_val_fold)
            y_proba = pipe.predict_proba(X_val_fold)[:, 1]

            # Compute metrics
            auc = roc_auc_score(y_val_fold, y_proba)
            f1 = f1_score(y_val_fold, y_pred)
            rec = recall_score(y_val_fold, y_pred)
            prec = precision_score(y_val_fold, y_pred, zero_division=0)
            acc = accuracy_score(y_val_fold, y_pred)

            fold_metrics["auc_roc"].append(auc)
            fold_metrics["f1"].append(f1)
            fold_metrics["recall"].append(rec)
            fold_metrics["precision"].append(prec)
            fold_metrics["accuracy"].append(acc)

            # Track TP for collapsed-model detection
            cm = confusion_matrix(y_val_fold, y_pred)
            tp = cm[1, 1] if cm.shape[0] > 1 else 0
            all_tp += tp

            print(f"    Fold {fold_idx + 1}: AUC={auc:.4f}  F1={f1:.4f}  "
                  f"Recall={rec:.4f}  Prec={prec:.4f}  Acc={acc:.4f}  TP={tp}")

        # Check for collapsed model
        if all_tp == 0:
            print(f"    WARNING: {model_name} predicts all negatives across "
                  f"all folds! Class-weight settings may have failed.")

        # Store mean scores
        model_results = {}
        for metric, values in fold_metrics.items():
            mean_val = np.mean(values)
            std_val = np.std(values)
            model_results[metric] = mean_val
            model_results[f"{metric}_std"] = std_val

        cv_results[model_name] = model_results

        # Track best model by mean AUC-ROC
        mean_auc = model_results["auc_roc"]
        if mean_auc > best_auc:
            best_auc = mean_auc
            best_model_name = model_name

        print(f"    Mean:  AUC={model_results['auc_roc']:.4f}±{model_results['auc_roc_std']:.4f}  "
              f"F1={model_results['f1']:.4f}±{model_results['f1_std']:.4f}  "
              f"Recall={model_results['recall']:.4f}±{model_results['recall_std']:.4f}")

    cv_results["best_model_name"] = best_model_name

    # Print summary table
    print("\n  " + "=" * 63)
    print("  Cross-Validation Summary (mean +/- std)")
    print("  " + "=" * 63)
    print(f"  {'Model':<22s} {'AUC-ROC':>12s} {'F1':>12s} {'Recall':>12s}")
    print("  " + "-" * 60)
    for name in models.keys():
        r = cv_results[name]
        marker = " *" if name == best_model_name else ""
        print(f"  {name:<22s} "
              f"{r['auc_roc']:.4f}±{r['auc_roc_std']:.4f} "
              f"{r['f1']:.4f}±{r['f1_std']:.4f} "
              f"{r['recall']:.4f}±{r['recall_std']:.4f}{marker}")
    print("  " + "=" * 63)
    print(f"  * Best model: {best_model_name} (AUC-ROC = {best_auc:.4f})")

    return cv_results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json
    from src.models import get_models
    from config import BASE_DIR

    models = get_models()

    # Try to load cached best_params
    params_path = os.path.join(BASE_DIR, "models", "best_params.json")
    if os.path.exists(params_path):
        with open(params_path, "r") as f:
            best_params = json.load(f)
        print("Loaded cached best_params.json")
    else:
        print("No cached params found — using default model configs (empty params)")
        best_params = {name: {} for name in models.keys()}

    cv_results = run_cross_validation(None, models, best_params)
