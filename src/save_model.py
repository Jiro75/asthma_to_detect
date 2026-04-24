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
#   [ ] Select best model name from cv_results['best_model_name']
#   [ ] Load X_train, y_train from DATA_SPLITS
#   [ ] Set best_params on pipeline: pipe.set_params(clf=best_clf, **params)
#   [ ] Fit on (X_train, y_train)
#   [ ] joblib.dump(fitted_pipe, MODEL_PATH)
#   [ ] Reload and verify predictions match (assert allclose)
#   [ ] Print: model name, MODEL_PATH, train-set AUC-ROC (sanity check)
#   [ ] Return fitted pipeline
# =============================================================================

import os
import joblib
import pandas as pd
from config import DATA_SPLITS, MODEL_PATH


def fit_and_save(cv_pipe, models: dict, best_params: dict, cv_results: dict):
    """
    Refit the best model on full training data and persist with joblib.

    Parameters
    ----------
    cv_pipe     : imblearn Pipeline (preprocessor → smote → clf placeholder)
    models      : dict from get_models()
    best_params : dict from run_tuning()
    cv_results  : dict from run_cross_validation() — used to identify best model

    Returns
    -------
    fitted imblearn Pipeline (also saved to MODEL_PATH)
    """
    # TODO: implement
    raise NotImplementedError("save_model.py: fit_and_save() not yet implemented.")
