# =============================================================================
# src/pipeline.py
# MEMBER 4 — Deadline: 1 May
# Role: Integration Lead & Inference Engineer
# =============================================================================
# Responsibility:
#   D1: Assemble the unified inference pipeline:
#         ColumnTransformer → SMOTE (training only) → Best Classifier
#   D2: Serialize (joblib.dump) and verify the loaded pipeline produces
#         numerically identical predictions to the pre-serialized one.
#
# This is the single fittable/predictable object used by inference.py and
# for the final submission. It must be compatible with:
#   pipeline.fit(X_train, y_train)
#   pipeline.predict(X_new)
#   pipeline.predict_proba(X_new)
#
# Serialization verification:
#   loaded = joblib.load(MODEL_PATH)
#   assert np.allclose(pipeline.predict_proba(X_val[:5]),
#                      loaded.predict_proba(X_val[:5]))
#
# Checklist:
#   [ ] Import build_preprocessor() from preprocessing.py
#   [ ] Import build_cv_pipeline() from cv_pipeline.py
#   [ ] Accept best_params and best_model_name as arguments
#   [ ] Assemble: set_params(clf=best_classifier, **best_params[best_model_name])
#   [ ] Load X_train, y_train from DATA_SPLITS; fit the pipeline
#   [ ] joblib.dump to MODEL_PATH
#   [ ] Reload and verify predictions match (np.allclose assert)
#   [ ] Print confirmation + MODEL_PATH
#   [ ] Return fitted pipeline
# =============================================================================

import os
import joblib
import numpy as np
import pandas as pd
from config import DATA_SPLITS, MODEL_PATH


def build_full_pipeline(best_params: dict, cv_results: dict, models: dict):
    """
    Assemble, fit, serialize, and verify the unified end-to-end pipeline.

    Parameters
    ----------
    best_params  : dict from run_tuning()
    cv_results   : dict from run_cross_validation() — identifies best model
    models       : dict from get_models()

    Returns
    -------
    fitted imblearn Pipeline (also saved to MODEL_PATH)
    """
    # TODO: implement
    raise NotImplementedError("pipeline.py: build_full_pipeline() not yet implemented.")
