# =============================================================================
# src/save_model.py
# MEMBER 2 — Deadline: 28 April
# Role: Model Training & Hyperparameter Optimization Engineer
# =============================================================================
# Responsibility:
#   Refit the best model on the FULL training set using its tuned best_params,
#   then serialize to MODEL_PATH with joblib. The saved file is the handoff
#   artifact to Member 3 (threshold calibration + SHAP) and Member 4 (inference).
#
# ⚠️  IMPORTANT: After saving, reload the model in a clean state and verify
#     that loaded_model.predict_proba(X_train[:5]) matches the original —
#     confirming no unpickled state dependencies.
#
# Checklist:
#   [x] Select best model name from cv_results['best_model_name']
#   [x] Load X_train, y_train from DATA_SPLITS
#   [x] Build pipeline with best classifier + best params
#   [x] Fit on (X_train, y_train)
#   [x] joblib.dump(fitted_pipe, MODEL_PATH)
#   [x] Reload and verify predictions match (assert allclose)
#   [x] Print: model name, MODEL_PATH, train-set AUC-ROC (sanity check)
#   [x] Return fitted pipeline
# =============================================================================

import os
import sys
import pickle
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.metrics import roc_auc_score

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from imblearn.pipeline import Pipeline as ImbPipeline

from src.cv_pipeline import build_cv_pipeline
from src.preprocessing import build_preprocessor_top25
from config import (
    DATA_SPLITS, MODEL_PATH, RANDOM_STATE,
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


def fit_and_save(cv_pipe, models: dict, best_params: dict, cv_results: dict):
    """
    Refit the best model on full training data and persist with joblib.

    Parameters
    ----------
    cv_pipe     : imblearn Pipeline (preprocessor → smote → clf placeholder)
                  — not used directly; we build top-25 pipelines internally.
    models      : dict from get_models()
    best_params : dict from run_tuning()
    cv_results  : dict from run_cross_validation() — used to identify best model

    Returns
    -------
    fitted imblearn Pipeline (also saved to MODEL_PATH)
    """
    # Identify best model from CV results
    best_model_name = cv_results["best_model_name"]
    print(f"  Best model: {best_model_name}")

    # Load training data
    X_train, y_train = _load_training_data()
    print(f"  Training data: X={X_train.shape}, y={y_train.shape}")

    # Reconstruct the best classifier with tuned params
    clf = _reconstruct_classifier(best_model_name, best_params)
    print(f"  Classifier config: {clf}")

    # Build and fit the pipeline
    pipeline = build_cv_pipeline(
        clf,
        smote_ratio=SMOTE_RATIO_DEFAULT,
        preprocessor=build_preprocessor_top25()
    )
    pipeline.fit(X_train, y_train)
    print("  Pipeline fitted on full training set.")

    # Sanity check: train-set AUC-ROC
    y_proba_train = pipeline.predict_proba(X_train)[:, 1]
    train_auc = roc_auc_score(y_train, y_proba_train)
    print(f"  Train-set AUC-ROC (sanity check): {train_auc:.4f}")

    # Serialize
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"  Model saved to: {MODEL_PATH}")

    # Reload and verify predictions match
    loaded_pipeline = joblib.load(MODEL_PATH)
    y_proba_original = pipeline.predict_proba(X_train[:5])
    y_proba_loaded = loaded_pipeline.predict_proba(X_train[:5])
    assert np.allclose(y_proba_original, y_proba_loaded), (
        "Loaded model predictions do NOT match original! "
        "Serialization may have failed."
    )
    print("  [OK] Reload verification passed (predictions match).")

    return pipeline


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json
    from src.models import get_models
    from config import BASE_DIR

    models = get_models()

    params_path = os.path.join(BASE_DIR, "models", "best_params.json")
    if os.path.exists(params_path):
        with open(params_path, "r") as f:
            best_params = json.load(f)
    else:
        best_params = {name: {} for name in models.keys()}

    # Fake cv_results for standalone testing
    cv_results = {"best_model_name": "XGBoost"}

    fit_and_save(None, models, best_params, cv_results)
