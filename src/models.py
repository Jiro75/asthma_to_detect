# =============================================================================
# src/models.py
# MEMBER 2 — Deadline: 28 April
# Role: Model Training & Hyperparameter Optimization Engineer
# =============================================================================
# Responsibility:
#   Define all 4 classifiers with class-imbalance-aware settings.
#   XGBoost is the PRIMARY model. LightGBM, LogReg, RF are comparison baselines.
#
# ⚠️  Class imbalance is 18.3:1 (SCALE_POS_WEIGHT). Every classifier must
#     be explicitly configured to treat imbalance as a first-class concern.
#
# Classifier specs:
#   XGBoost:
#     scale_pos_weight=SCALE_POS_WEIGHT (18.3)
#     use_label_encoder=False          ← required to suppress deprecation error
#     eval_metric='logloss'
#     random_state=RANDOM_STATE
#
#   LightGBM:
#     is_unbalance=True
#     verbose=-1
#     random_state=RANDOM_STATE
#
#   Logistic Regression:
#     class_weight='balanced'
#     max_iter=1000
#     random_state=RANDOM_STATE
#
#   Random Forest:
#     class_weight='balanced_subsample'
#     n_jobs=-1
#     random_state=RANDOM_STATE
#
# Checklist:
#   [x] Return dict: {model_name: unfitted_estimator}
#   [x] All four classifiers present with imbalance settings
#   [x] After creating models, verify no model predicts all-zeros on a tiny
#       synthetic imbalanced sample (optional sanity check)
# =============================================================================

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from config import SCALE_POS_WEIGHT, RANDOM_STATE


def get_models() -> dict:
    """
    Return {name: unfitted classifier} for all 4 models.

    Keys: "XGBoost", "LightGBM", "LogisticRegression", "RandomForest"

    Every classifier is configured with class-imbalance awareness:
    - XGBoost: scale_pos_weight=18.3 up-weights the minority class
    - LightGBM: is_unbalance=True internally adjusts weights
    - LogReg: class_weight='balanced' auto-scales inversely to freq
    - RF: class_weight='balanced_subsample' per-tree rebalancing
    """
    models = {
        "XGBoost": XGBClassifier(
            scale_pos_weight=SCALE_POS_WEIGHT,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
        ),
        "LightGBM": LGBMClassifier(
            is_unbalance=True,
            random_state=RANDOM_STATE,
            verbose=-1,
        ),
        "LogisticRegression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
        "RandomForest": RandomForestClassifier(
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    print(f"  Defined {len(models)} classifiers:")
    for name, clf in models.items():
        print(f"    • {name}: {type(clf).__name__}")
    return models


# ---------------------------------------------------------------------------
# CLI sanity check — train on a tiny synthetic imbalanced set
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import numpy as np
    from sklearn.datasets import make_classification
    from sklearn.metrics import confusion_matrix

    models = get_models()

    # Create a synthetic imbalanced dataset (similar to real data)
    X_synth, y_synth = make_classification(
        n_samples=500,
        n_features=25,
        weights=[0.95, 0.05],   # 95% majority
        random_state=RANDOM_STATE,
        flip_y=0,
    )

    print("\n--- Sanity check: training on synthetic imbalanced data ---")
    for name, clf in models.items():
        clf.fit(X_synth, y_synth)
        y_pred = clf.predict(X_synth)
        cm = confusion_matrix(y_synth, y_pred)
        tp = cm[1, 1] if cm.shape[0] > 1 else 0
        print(f"  {name:25s} → TP = {tp}  (collapsed = {tp == 0})")
        if tp == 0:
            print(f"    ⚠️  WARNING: {name} predicts all negatives!")
