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
#   [ ] Return dict: {model_name: unfitted_estimator}
#   [ ] All four classifiers present with imbalance settings
#   [ ] After creating models, verify no model predicts all-zeros on a tiny
#       synthetic imbalanced sample (optional sanity check)
# =============================================================================

from config import SCALE_POS_WEIGHT, RANDOM_STATE


def get_models() -> dict:
    """
    Return {name: unfitted classifier} for all 4 models.

    Keys: "XGBoost", "LightGBM", "LogisticRegression", "RandomForest"
    """
    # TODO: implement
    raise NotImplementedError("models.py: get_models() not yet implemented.")
