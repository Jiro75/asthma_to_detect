# =============================================================================
# src/cv_pipeline.py
# MEMBER 1 — Deadline: 26 April
# Role: Data Pipeline & Preprocessing Engineer
# =============================================================================
# Responsibility:
#   Wrap the ColumnTransformer and SMOTE inside an imbalanced-learn Pipeline.
#   This guarantees SMOTE is applied ONLY on each CV training fold, never on
#   held-out folds — preventing minority-class leakage.
#
# ⚠️  CRITICAL: Use imblearn.pipeline.Pipeline, NOT sklearn.pipeline.Pipeline.
#     sklearn's Pipeline will apply SMOTE on validation folds during CV.
#
# Pipeline steps:
#   1. 'preprocessor' : ColumnTransformer from preprocessing.py
#   2. 'smote'        : SMOTE(sampling_strategy=SMOTE_RATIO_DEFAULT,
#                             random_state=RANDOM_STATE)
#   3. 'clf'          : placeholder (None / DummyClassifier) — swapped in
#                       by tuning.py and cross_validate.py
#
# SMOTE ratio is configurable: default SMOTE_RATIO_DEFAULT (0.4),
# can be swept across SMOTE_RATIO_MIN (0.3) – SMOTE_RATIO_MAX (0.5).
#
# Checklist:
#   [ ] Import from imblearn.pipeline, NOT sklearn.pipeline
#   [ ] Accept smote_ratio kwarg (default=SMOTE_RATIO_DEFAULT)
#   [ ] Leave step 3 as a named placeholder so callers can do:
#         pipe.set_params(clf=SomeClassifier(...))
#   [ ] Print pipeline steps on creation
# =============================================================================

from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from config import SMOTE_RATIO_DEFAULT, RANDOM_STATE


def build_cv_pipeline(
    preprocessor: ColumnTransformer,
    smote_ratio: float = SMOTE_RATIO_DEFAULT,
) -> Pipeline:
    """
    Wrap preprocessor + SMOTE into an imblearn Pipeline with a clf placeholder.

    Parameters
    ----------
    preprocessor : unfitted ColumnTransformer from build_preprocessor()
    smote_ratio  : SMOTE sampling_strategy (default SMOTE_RATIO_DEFAULT = 0.4)

    Returns
    -------
    imblearn Pipeline with steps: preprocessor → smote → clf (placeholder)
    """
    # TODO: implement
    raise NotImplementedError("cv_pipeline.py: build_cv_pipeline() not yet implemented.")
