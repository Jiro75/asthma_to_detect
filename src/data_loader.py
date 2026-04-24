# =============================================================================
# src/data_loader.py
# MEMBER 1 — Deadline: 26 April
# Role: Data Pipeline & Preprocessing Engineer
# =============================================================================
# Responsibility:
#   Ingest the Kaggle CSV, validate the schema (26 expected feature columns +
#   target), check dtypes, remove exact duplicate rows, and return a clean
#   (X, y) pair. Raise informative errors if validation fails.
#
# Expected output:
#   X : pd.DataFrame  — all feature columns (NUMERIC + NOMINAL + BINARY)
#   y : pd.Series     — binary target (0 = No Asthma, 1 = Asthma)
#
# Checklist:
#   [ ] Load DATA_RAW csv
#   [ ] Assert all expected columns are present (26 features + TARGET_COL)
#   [ ] Assert correct dtypes (numeric cols are float/int, binary cols are int)
#   [ ] Drop exact duplicate rows; log how many were removed
#   [ ] Separate X (features) from y (TARGET_COL)
#   [ ] Print shape and class balance summary (value_counts + ratio)
#   [ ] Raise ValueError with a descriptive message if any check fails
# =============================================================================

import pandas as pd
from config import DATA_RAW, TARGET_COL, NUMERIC_FEATURES, NOMINAL_FEATURES, BINARY_FEATURES

EXPECTED_FEATURES = NUMERIC_FEATURES + NOMINAL_FEATURES + BINARY_FEATURES  # 26 columns


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """
    Load raw CSV, validate schema and dtypes, drop duplicates.

    Returns
    -------
    X : pd.DataFrame  — feature matrix (26 columns)
    y : pd.Series     — binary target
    """
    # TODO: implement
    raise NotImplementedError("data_loader.py: load_data() not yet implemented.")
