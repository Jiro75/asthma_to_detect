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
"""
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from config import SMOTE_RATIO_DEFAULT, RANDOM_STATE


def build_cv_pipeline(
    preprocessor: ColumnTransformer,
    smote_ratio: float = SMOTE_RATIO_DEFAULT,
) -> Pipeline:
    
    Wrap preprocessor + SMOTE into an imblearn Pipeline with a clf placeholder.

    Parameters
    ----------
    preprocessor : unfitted ColumnTransformer from build_preprocessor()
    smote_ratio  : SMOTE sampling_strategy (default SMOTE_RATIO_DEFAULT = 0.4)

    Returns
    -------
    imblearn Pipeline with steps: preprocessor → smote → clf (placeholder)
    
    # TODO: implement
    raise NotImplementedError("cv_pipeline.py: build_cv_pipeline() not yet implemented.")
"""
"""
cv_pipeline.py
==============
Data Pipeline & Preprocessing Engineer — Deliverable D4

Responsibilities:
- Wrap the ColumnTransformer (preprocessing.py) and SMOTE inside an
  imbalanced-learn Pipeline — NOT a vanilla sklearn Pipeline
- This guarantees SMOTE is applied only on each training fold during
  cross-validation; it NEVER touches held-out validation folds
- SMOTE sampling_strategy is configurable (default 0.4, sweep 0.3–0.5)
- Expose build_cv_pipeline() and run_cross_validation() for downstream use

Author : Member 1
Project: Asthma Disease Detection — Phase III
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline   # ← CRITICAL: imblearn, not sklearn
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.base import BaseEstimator

from preprocessing import build_preprocessor            # D3

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("cv_pipeline")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_SMOTE_RATIO: float  = 0.4
SMOTE_RATIO_SWEEP:  list[float] = [0.3, 0.35, 0.4, 0.45, 0.5]
RANDOM_STATE:       int     = 42
CV_N_SPLITS:        int     = 5      # stratified k-fold


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_cv_pipeline(
    classifier: BaseEstimator,
    smote_ratio: float = DEFAULT_SMOTE_RATIO,
    smote_k_neighbors: int = 5,
) -> ImbPipeline:
    """
    Build an imbalanced-learn Pipeline that chains:
      1. ColumnTransformer  (preprocessing — fitted only on train fold)
      2. SMOTE              (oversampling — applied only on train fold)
      3. Classifier         (provided by caller)

    Using ``imblearn.pipeline.Pipeline`` (not sklearn's) is the only safe
    way to prevent SMOTE from leaking synthetic samples into validation folds
    during cross-validation.

    Parameters
    ----------
    classifier : sklearn-compatible estimator
        Any classifier exposing .fit() / .predict() / .predict_proba().
    smote_ratio : float, default 0.4
        Desired ratio of minority to majority class *after* SMOTE.
        Typical sweep range: [0.3, 0.35, 0.4, 0.45, 0.5].
    smote_k_neighbors : int, default 5
        Number of nearest neighbours used by SMOTE when generating synthetic
        samples. Reduce if minority class is very small.

    Returns
    -------
    imblearn.pipeline.Pipeline
        Unfitted pipeline ready for use with cross_validate() or .fit().

    Notes
    -----
    With 124 positive cases (Phase II), k_neighbors=5 is safe (124 > 5).
    If the minority count in any fold drops below k_neighbors, SMOTE raises
    a ValueError — decrease k_neighbors accordingly.
    """
    if not (0 < smote_ratio <= 1.0):
        raise ValueError(
            f"smote_ratio must be in (0, 1]. Got: {smote_ratio}"
        )

    preprocessor = build_preprocessor()

    smote = SMOTE(
        sampling_strategy=smote_ratio,
        k_neighbors=smote_k_neighbors,
        random_state=RANDOM_STATE,
       
    )

    pipeline = ImbPipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("smote",         smote),
            ("classifier",    classifier),
        ],
        memory=None,    # disable caching during development; enable for speed
    )

    logger.info(
        "imblearn Pipeline built — SMOTE ratio: %.2f | k_neighbors: %d | "
        "classifier: %s",
        smote_ratio,
        smote_k_neighbors,
        type(classifier).__name__,
    )
    return pipeline


def run_cross_validation(
    pipeline: ImbPipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scoring: list[str] | dict[str, Any] | None = None,
    n_splits: int = CV_N_SPLITS,
    return_train_score: bool = True,
) -> dict[str, np.ndarray]:
    """
    Run stratified k-fold cross-validation using the provided imblearn
    pipeline.

    SMOTE is applied only inside each training fold; it never touches the
    held-out fold — this is enforced by the imblearn pipeline mechanics.

    Parameters
    ----------
    pipeline : ImbPipeline
        Built by ``build_cv_pipeline()``.
    X_train : pd.DataFrame
        Training feature matrix (from splitter.split_dataset()).
    y_train : pd.Series
        Training labels.
    scoring : list[str] or dict or None
        Metrics to compute. Defaults to a clinically meaningful set suited
        for class-imbalanced problems.
    n_splits : int, default 5
        Number of stratified folds.
    return_train_score : bool, default True
        Whether to compute train scores (helps diagnose overfitting).

    Returns
    -------
    dict[str, np.ndarray]
        Cross-validation results from sklearn.model_selection.cross_validate.
        Keys follow the pattern ``'test_{metric}'`` and ``'train_{metric}'``.
    """
    if scoring is None:
        # Metrics appropriate for severe class imbalance (18.3:1 ratio)
        scoring = {
            "roc_auc":   "roc_auc",
            "f1":        "f1",
            "precision": "precision",
            "recall":    "recall",
            "average_precision": "average_precision",   # AUC-PR
        }

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    logger.info(
        "Starting %d-fold stratified CV — metrics: %s",
        n_splits,
        list(scoring.keys()) if isinstance(scoring, dict) else scoring,
    )

    results = cross_validate(
        estimator=pipeline,
        X=X_train,
        y=y_train,
        cv=cv,
        scoring=scoring,
        return_train_score=return_train_score,
        n_jobs=-1,
        verbose=0,
    )

    _log_cv_results(results, scoring)
    return results


def sweep_smote_ratio(
    classifier: BaseEstimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    ratios: list[float] = SMOTE_RATIO_SWEEP,
    primary_metric: str = "roc_auc",
    n_splits: int = CV_N_SPLITS,
) -> tuple[float, dict[str, dict]]:
    """
    Sweep SMOTE sampling_strategy over a range of ratios and return the best
    value according to the primary metric on held-out folds.

    Parameters
    ----------
    classifier : BaseEstimator
        Unfitted sklearn-compatible classifier.
    X_train : pd.DataFrame
    y_train : pd.Series
    ratios : list[float]
        Values of sampling_strategy to try. Default: [0.3, 0.35, 0.4, 0.45, 0.5].
    primary_metric : str
        Metric used to select the best ratio.
    n_splits : int

    Returns
    -------
    best_ratio : float
        SMOTE ratio that maximised mean ``primary_metric`` across folds.
    all_results : dict[float, dict]
        Mapping from each ratio to its full cross_validate output dict.
    """
    all_results: dict[float, dict] = {}
    best_ratio:  float = ratios[0]
    best_score:  float = -np.inf

    logger.info(
        "Sweeping SMOTE ratios %s — primary metric: '%s'",
        ratios,
        primary_metric,
    )

    for ratio in ratios:
        import copy
        clf_copy = copy.deepcopy(classifier)      # fresh estimator each time
        pipe     = build_cv_pipeline(clf_copy, smote_ratio=ratio)
        results  = run_cross_validation(
            pipe, X_train, y_train, n_splits=n_splits
        )
        all_results[ratio] = results

        mean_score = float(np.mean(results[f"test_{primary_metric}"]))
        logger.info(
            "  ratio=%.2f  mean %s = %.4f ± %.4f",
            ratio,
            primary_metric,
            mean_score,
            float(np.std(results[f"test_{primary_metric}"])),
        )

        if mean_score > best_score:
            best_score = mean_score
            best_ratio = ratio

    logger.info(
        "Best SMOTE ratio: %.2f  (mean %s = %.7f)",
        best_ratio,
        primary_metric,
        best_score,
    )
    return best_ratio, all_results


def fit_final_pipeline(
    classifier: BaseEstimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    smote_ratio: float = DEFAULT_SMOTE_RATIO,
) -> ImbPipeline:
    """
    Build and fit the full pipeline on the entire training set.

    This is called **once**, after cross-validation has confirmed the
    chosen hyperparameters, and before evaluating on the locked test set.

    Parameters
    ----------
    classifier : BaseEstimator
        Configured (but unfitted) classifier.
    X_train, y_train : training data
    smote_ratio : float
        Best ratio determined by sweep_smote_ratio().

    Returns
    -------
    ImbPipeline
        Fitted pipeline ready for .predict() / .predict_proba().
    """
    pipeline = build_cv_pipeline(classifier, smote_ratio=smote_ratio)
    logger.info("Fitting final pipeline on full training set …")
    pipeline.fit(X_train, y_train)
    logger.info("Final pipeline fitted successfully.")
    return pipeline


# ---------------------------------------------------------------------------
# Early-stopping helper  (used by Member 2 — tuning.py)
# ---------------------------------------------------------------------------

def preprocess_for_early_stopping(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    smote_ratio: float = DEFAULT_SMOTE_RATIO,
    smote_k_neighbors: int = 5,
) -> tuple[np.ndarray, pd.Series, np.ndarray]:
    """
    Prepare data for classifiers that need a raw eval_set (XGBoost, LightGBM
    early stopping). The imblearn Pipeline cannot forward ``eval_set`` through
    SMOTE to the classifier, so Member 2's tuning.py must preprocess data
    manually inside each Optuna trial instead of calling pipe.fit().

    Steps performed here:
      1. Fit ColumnTransformer on X_train only  (no leakage)
      2. Apply SMOTE to the transformed training data
      3. Transform X_val with the already-fitted ColumnTransformer

    Parameters
    ----------
    X_train : pd.DataFrame
        Training fold features (raw, before preprocessing).
    y_train : pd.Series
        Training fold labels.
    X_val : pd.DataFrame
        Validation fold features (raw).  Used only for transform, never fit.
    smote_ratio : float
        SMOTE sampling_strategy (minority / majority ratio after oversampling).
    smote_k_neighbors : int
        SMOTE k_neighbors — reduce if minority fold count is very small.

    Returns
    -------
    X_train_res : np.ndarray
        SMOTE-oversampled, preprocessed training features.
    y_train_res : pd.Series
        Corresponding labels after SMOTE.
    X_val_t : np.ndarray
        Preprocessed validation features (no SMOTE applied).

    Usage in tuning.py (inside an Optuna trial)
    --------------------------------------------
    .. code-block:: python

        from src.cv_pipeline import preprocess_for_early_stopping

        # Inside trial objective, for one inner fold:
        X_tr_res, y_tr_res, X_val_t = preprocess_for_early_stopping(
            X_inner_train, y_inner_train, X_inner_val,
            smote_ratio=smote_ratio,
        )
        clf = XGBClassifier(early_stopping_rounds=50, ...)
        clf.fit(
            X_tr_res, y_tr_res,
            eval_set=[(X_val_t, y_inner_val)],
            verbose=False,
        )
    """
    preprocessor = build_preprocessor()

    X_train_t: np.ndarray = preprocessor.fit_transform(X_train, y_train)
    X_val_t:   np.ndarray = preprocessor.transform(X_val)

    smote = SMOTE(
        sampling_strategy=smote_ratio,
        k_neighbors=smote_k_neighbors,
        random_state=RANDOM_STATE,
    )
    X_train_res, y_train_res = smote.fit_resample(X_train_t, y_train)

    logger.info(
        "preprocess_for_early_stopping: train %s → SMOTE %s | val %s",
        X_train_t.shape,
        X_train_res.shape,
        X_val_t.shape,
    )
    return X_train_res, y_train_res, X_val_t


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log_cv_results(
    results: dict[str, np.ndarray],
    scoring: list[str] | dict[str, Any],
) -> None:
    """Pretty-print mean ± std for all test-set metrics."""
    metric_names = list(scoring.keys()) if isinstance(scoring, dict) else scoring
    logger.info("Cross-validation results:")
    logger.info("  %-22s  %8s  %8s", "metric", "mean", "std")
    logger.info("  " + "-" * 42)
    for metric in metric_names:
        key = f"test_{metric}"
        if key in results:
            vals = results[key]
            logger.info(
                "  %-22s  %8.4f  %8.4f",
                metric,
                float(np.mean(vals)),
                float(np.std(vals)),
            )


# ---------------------------------------------------------------------------
# CLI demo — requires a real CSV to run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from sklearn.ensemble import RandomForestClassifier
    from src.data_loader import load_and_validate
    from src.splitter import split_dataset

    # HOW TO RUN:
    #   python src/cv_pipeline.py
    #   python src/cv_pipeline.py data/raw/asthma_disease_data.csv
    #
    # NOTE: RandomForestClassifier below is a DEMO placeholder only.
    # Member 2 will replace it with the tuned classifiers from models.py.
    # Do NOT delete this block — it is the integration smoke-test for your pipeline.
    csv = sys.argv[1] if len(sys.argv) > 1 else "data/raw/asthma_disease_data.csv"
    df  = load_and_validate(csv)
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(df)

    clf = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced_subsample",
        random_state=42,
    )
    pipe = build_cv_pipeline(clf, smote_ratio=DEFAULT_SMOTE_RATIO)

    results = run_cross_validation(pipe, X_train, y_train)

    print("\nFold-level ROC-AUC scores:")
    print(results["test_roc_auc"])