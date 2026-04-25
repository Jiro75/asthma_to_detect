# =============================================================================
# src/preprocessing.py
# MEMBER 1 — Deadline: 26 April
# Role: Data Pipeline & Preprocessing Engineer
# =============================================================================
# Responsibility:
#   Build a dual-branch ColumnTransformer. This is the most critical deliverable
#   for Member 1. The transformer is returned UNFITTED — fitting happens inside
#   the imblearn pipeline during cross-validation (never on the full dataset).
#
# Branch design:
#   Numeric branch  (NUMERIC_FEATURES, 10 cols):
#     SimpleImputer(strategy='median')
#     → PowerTransformer(method='yeo-johnson')   ← normalises skewed features
#     → StandardScaler()
#
#   Nominal branch  (NOMINAL_FEATURES, 3 cols):
#     SimpleImputer(strategy='most_frequent')
#     → OneHotEncoder(handle_unknown='ignore', sparse_output=False)
#
#   Binary branch   (BINARY_FEATURES, 13 cols):
#     Already 0/1 integers — DO NOT apply OHE (doubles column count for no gain)
#     SimpleImputer(strategy='most_frequent')
#     → OrdinalEncoder()   OR   passthrough
#
# IMPORTANT: Never call .fit() on the full dataset. The pipeline handles fitting.
#
# Checklist:
#   [ ] Use column lists from config: NUMERIC_FEATURES, NOMINAL_FEATURES, BINARY_FEATURES
#   [ ] Return unfitted ColumnTransformer (remainder='drop')
#   [ ] Print branch summary (column counts per branch)
# =============================================================================
"""
from sklearn.compose import ColumnTransformer
from config import NUMERIC_FEATURES, NOMINAL_FEATURES, BINARY_FEATURES


def build_preprocessor() -> ColumnTransformer:
   
    Build and return an unfitted dual-branch ColumnTransformer.

    Branches: numeric (impute→power→scale), nominal (impute→OHE), binary (impute→ordinal)
   
    # TODO: implement
    raise NotImplementedError("preprocessing.py: build_preprocessor() not yet implemented.")
"""
"""
preprocessing.py
================
Data Pipeline & Preprocessing Engineer — Deliverable D3

Responsibilities:
- Define the full ColumnTransformer with two branches:
    • Numeric  : SimpleImputer(median) → PowerTransformer(yeo-johnson) → StandardScaler
    • Categorical: SimpleImputer(most_frequent)
                   → OneHotEncoder  (nominal: Gender, Ethnicity, EducationLevel)
                   → OrdinalEncoder (ordered binary/ordinal features)
- Binary (0/1) features are routed to the ordinal branch — NOT through OHE
- Expose a build_preprocessor() factory that other modules import

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

# --- Continuous / truly numeric features ---
# These contain real-valued measurements and benefit from Yeo-Johnson + scaling
NUMERIC_FEATURES: list[str] = [
    "Age",
    "BMI",
    "PhysicalActivity",
    "DietQuality",
    "SleepQuality",
    "PollutionExposure",
    "PollenExposure",
    "DustExposure",
    "LungFunctionFEV1",
    "LungFunctionFVC",
]

# --- Nominal categorical features — encoded via OneHotEncoder ---
NOMINAL_FEATURES: list[str] = [
    "Gender",
    "Ethnicity",
    "EducationLevel",
]

# --- Ordered / binary features — encoded via OrdinalEncoder ---
# Binary (0/1) flags: applying OHE would double column count for zero benefit;
# route to OrdinalEncoder instead (identity transform for already-numeric 0/1)
BINARY_FEATURES: list[str] = [
    "Smoking",
    "PetAllergy",
    "FamilyHistoryAsthma",
    "HistoryOfAllergies",
    "Eczema",
    "HayFever",
    "GastroesophagealReflux",
    "Wheezing",
    "ShortnessOfBreath",
    "ChestTightness",
    "Coughing",
    "NighttimeSymptoms",
    "ExerciseInduced",
]

# Explicit category ordering for OrdinalEncoder (must cover all values in data)
# For binary features the order is simply [0, 1]
ORDINAL_CATEGORIES: list[list[Any]] = [
    [0, 1],  # Smoking
    [0, 1],  # PetAllergy
    [0, 1],  # FamilyHistoryAsthma
    [0, 1],  # HistoryOfAllergies
    [0, 1],  # Eczema
    [0, 1],  # HayFever
    [0, 1],  # GastroesophagealReflux
    [0, 1],  # Wheezing
    [0, 1],  # ShortnessOfBreath
    [0, 1],  # ChestTightness
    [0, 1],  # Coughing
    [0, 1],  # NighttimeSymptoms
    [0, 1],  # ExerciseInduced
]

assert len(BINARY_FEATURES) == len(ORDINAL_CATEGORIES), (
    "BINARY_FEATURES and ORDINAL_CATEGORIES must have the same length."
)


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


def _ordinal_branch() -> Pipeline:
    """
    Ordinal / binary pipeline:
      SimpleImputer(most_frequent) → OrdinalEncoder

    `categories` is explicitly supplied to guarantee a deterministic,
    reproducible encoding regardless of the order values appear in training.
    unknown_value=np.nan means unseen values are treated as missing rather
    than raising an error.
    """
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder(
            categories=ORDINAL_CATEGORIES,
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


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """
    Return human-readable output feature names after the transformer has
    been fitted.

    Parameters
    ----------
    preprocessor : ColumnTransformer
        A **fitted** ColumnTransformer (i.e., after calling .fit() or
        .fit_transform() on training data).

    Returns
    -------
    list[str]
        Names for every column in the transformed output matrix.

    Raises
    ------
    sklearn.exceptions.NotFittedError
        If called before the transformer is fitted.
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

    csv = sys.argv[1] if len(sys.argv) > 1 else "data/raw/asthma_disease_data.csv"
    df  = load_and_validate(csv)

    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(df)

    preprocessor = build_preprocessor()
    X_train_t    = preprocessor.fit_transform(X_train)
    X_val_t      = preprocessor.transform(X_val)

    print(f"X_train transformed shape : {X_train_t.shape}")
    print(f"X_val   transformed shape : {X_val_t.shape}")
    print(f"Output features ({X_train_t.shape[1]}):")
    for name in get_feature_names(preprocessor):
        print(f"  {name}")