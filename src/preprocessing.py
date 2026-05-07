"""
preprocessing.py
================
Data Pipeline & Preprocessing Engineer — Deliverable D3

Responsibilities:
- Define the full ColumnTransformer with two branches:
    • Numeric  : SimpleImputer(median) → PowerTransformer(yeo-johnson) → StandardScaler
    • Categorical: SimpleImputer(most_frequent)
                   → OneHotEncoder  (nominal: Gender, Ethnicity, EducationLevel, etc.)
                   → OrdinalEncoder (ordered binary/ordinal features)
- Binary (0/1) features are routed to the ordinal branch — NOT through OHE
- Expose a build_preprocessor() factory that other modules import
- Expose a build_preprocessor_top25() factory for the top-25 feature subset

Author : Member 1
Project: Asthma Disease Detection — Phase III
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    PowerTransformer,
    StandardScaler,
)

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    NUMERIC_FEATURES, NOMINAL_FEATURES, BINARY_FEATURES,
    TOP_25_NUMERIC, TOP_25_BINARY,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("preprocessing")


# ---------------------------------------------------------------------------
# Feature catalogue
# ---------------------------------------------------------------------------
# Explicit category ordering for OrdinalEncoder (must cover all values in data)
# For binary features the order is simply [0, 1]
ORDINAL_CATEGORIES: list[list[Any]] = [
    [0, 1] for _ in BINARY_FEATURES
]

# ---------------------------------------------------------------------------
# Branch pipeline factories
# ---------------------------------------------------------------------------

def _numeric_branch() -> Pipeline:
    """
    Numeric pipeline:
      SimpleImputer(median) → PowerTransformer(yeo-johnson) → StandardScaler

    Yeo-Johnson handles zero and negative values (unlike Box-Cox) and is
    appropriate for skewed continuous clinical/behavioral measurements.
    StandardScaler is applied last so all features share the same scale
    after the non-linear transformation.
    """
    return Pipeline(steps=[
        ("imputer",    SimpleImputer(strategy="median")),
        ("power",      PowerTransformer(method="yeo-johnson", standardize=False)),
        ("scaler",     StandardScaler()),
    ])


def _nominal_branch() -> Pipeline:
    """
    Nominal categorical pipeline:
      SimpleImputer(most_frequent) → OneHotEncoder

    drop='first' is NOT used to avoid downstream ambiguity with tree-based
    models; handle_unknown='ignore' makes the pipeline robust to unseen
    categories at inference time.
    """
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe",     OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,       # return dense array
            dtype=np.float32,
        )),
    ])


def _ordinal_branch(features: list[str] | None = None) -> Pipeline:
    """
    Ordinal / binary pipeline:
      SimpleImputer(most_frequent) → OrdinalEncoder

    `categories` is explicitly supplied to guarantee a deterministic,
    reproducible encoding regardless of the order values appear in training.
    unknown_value=np.nan means unseen values are treated as missing rather
    than raising an error.
    """
    if features is None:
        features = BINARY_FEATURES
    categories = [[0, 1] for _ in features]
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder(
            categories=categories,
            handle_unknown="use_encoded_value",
            unknown_value=np.nan,
            dtype=np.float32,
        )),
    ])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_preprocessor() -> ColumnTransformer:
    """
    Construct and return the full dual-branch ColumnTransformer.

    The transformer is returned **unfitted**. It must be fitted only on
    training data (handled automatically inside the imbalanced-learn
    Pipeline in cv_pipeline.py).

    Returns
    -------
    ColumnTransformer
        With three named transformers:
          • ``'numeric'``  — continuous measurements
          • ``'nominal'``  — one-hot encoded categorical features
          • ``'ordinal'``  — binary / ordered flag features

    Notes
    -----
    remainder='drop' is intentional: PatientID must be removed before
    calling fit_transform(), which splitter.py already handles.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", _numeric_branch(),  NUMERIC_FEATURES),
            ("nominal", _nominal_branch(),  NOMINAL_FEATURES),
            ("ordinal", _ordinal_branch(),  BINARY_FEATURES),
        ],
        remainder="drop",          # drop any unlisted columns (e.g. residuals)
        n_jobs=-1,                 # run branches in parallel
        verbose_feature_names_out=True,
    )

    logger.info(
        "ColumnTransformer built — "
        "numeric: %d | nominal: %d | binary/ordinal: %d  (total input features: %d)",
        len(NUMERIC_FEATURES),
        len(NOMINAL_FEATURES),
        len(BINARY_FEATURES),
        len(NUMERIC_FEATURES) + len(NOMINAL_FEATURES) + len(BINARY_FEATURES),
    )
    return preprocessor


def build_preprocessor_top25() -> ColumnTransformer:
    """
    Construct a ColumnTransformer for the top-25 correlated features only.

    This is used by Member 2 for model training on a reduced feature set.
    Since no nominal/categorical features are in the top 25, only the
    numeric and ordinal/binary branches are included.

    Returns
    -------
    ColumnTransformer
        With two named transformers:
          • ``'numeric'``  — 22 continuous features (Yeo-Johnson + scale)
          • ``'ordinal'``  — 3 binary features (Allergy, Family_History,
                             Exercise_Induced_Symptoms)
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", _numeric_branch(),                    TOP_25_NUMERIC),
            ("ordinal", _ordinal_branch(TOP_25_BINARY),       TOP_25_BINARY),
        ],
        remainder="drop",
        n_jobs=-1,
        verbose_feature_names_out=True,
    )

    logger.info(
        "Top-25 ColumnTransformer built — "
        "numeric: %d | binary/ordinal: %d  (total: %d)",
        len(TOP_25_NUMERIC),
        len(TOP_25_BINARY),
        len(TOP_25_NUMERIC) + len(TOP_25_BINARY),
    )
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """
    Return human-readable output feature names after the transformer has
    been fitted.
    """
    return list(preprocessor.get_feature_names_out())


# ---------------------------------------------------------------------------
# Quick sanity-check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.data_loader import load_and_validate
    from src.splitter import split_dataset
    from config import DATA_RAW

    csv = sys.argv[1] if len(sys.argv) > 1 else DATA_RAW
    df  = load_and_validate(csv)

    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(df)

    preprocessor = build_preprocessor()
    X_train_t    = preprocessor.fit_transform(X_train)
    X_val_t      = preprocessor.transform(X_val)

    print(f"X_train transformed shape : {X_train_t.shape}")
    print(f"X_val   transformed shape : {X_val_t.shape}")

    # Test top-25 preprocessor
    preprocessor_25 = build_preprocessor_top25()
    X_train_25 = preprocessor_25.fit_transform(X_train)
    X_val_25   = preprocessor_25.transform(X_val)
    print(f"\nTop-25 X_train shape : {X_train_25.shape}")
    print(f"Top-25 X_val   shape : {X_val_25.shape}")