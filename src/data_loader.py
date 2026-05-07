"""
data_loader.py
==============
Data Pipeline & Preprocessing Engineer — Deliverable D1

Responsibilities:
- Ingest the Kaggle asthma CSV
- Validate schema (expected feature columns, correct dtypes)
- Remove exact duplicate rows
- Encode target 'Diagnosis' to 1/0
- Return a clean pandas DataFrame

Author : Member 1
Project: Asthma Disease Detection — Phase III
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    NUMERIC_FEATURES,
    NOMINAL_FEATURES,
    BINARY_FEATURES,
    TARGET_COL,
    DROP_COLS
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("data_loader")

# ---------------------------------------------------------------------------
# Schema Definition
# ---------------------------------------------------------------------------
# The columns we actually need for the pipeline (features + target + drop cols)
EXPECTED_COLUMNS: list[str] = (
    NUMERIC_FEATURES + NOMINAL_FEATURES + BINARY_FEATURES + [TARGET_COL] + DROP_COLS
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_columns(df: pd.DataFrame) -> None:
    """
    Ensure all expected feature columns AND the target column are present.
    Raises ValueError with a descriptive message if any are missing.
    """
    all_expected = set(EXPECTED_COLUMNS)
    present      = set(df.columns)
    missing      = all_expected - present
    extra        = present - all_expected

    if missing:
        raise ValueError(
            f"Schema validation failed — {len(missing)} column(s) missing from CSV:\n"
            f"  Missing : {sorted(missing)}\n"
        )

    if extra:
        logger.warning(
            "Unexpected extra column(s) found and will be retained: %s", sorted(extra)
        )

    logger.info("Column validation passed — all %d expected columns present.", len(all_expected))


def _validate_dtypes(df: pd.DataFrame) -> None:
    """
    Check that each column's dtype matches its expectation.
    Numeric features should be numeric, nominal can be object/string.
    Raises TypeError listing every violating column.
    """
    violations: list[str] = []

    for col in NUMERIC_FEATURES + BINARY_FEATURES:
        if col not in df.columns: continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            violations.append(f"  '{col}': expected numeric, got {df[col].dtype}")

    if violations:
        raise TypeError(
            f"Dtype validation failed for {len(violations)} column(s):\n"
            + "\n".join(violations)
        )

    logger.info("Dtype validation passed.")


def _remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop exact duplicate rows (all columns identical).
    Logs how many duplicates were removed.
    """
    n_before = len(df)
    df = df.drop_duplicates()
    n_removed = n_before - len(df)

    if n_removed:
        logger.warning("Removed %d exact duplicate row(s). Rows remaining: %d", n_removed, len(df))
    else:
        logger.info("No duplicate rows found.")

    return df


def _report_missing_values(df: pd.DataFrame) -> None:
    """
    Log a summary of missing values per column (informational only).
    """
    missing = df.isnull().sum()
    missing = missing[missing > 0]

    if missing.empty:
        logger.info("No missing values detected in any column.")
    else:
        pct = (missing / len(df) * 100).round(2)
        report = "\n".join(
            f"  {col}: {cnt} missing ({pct[col]}%)"
            for col, cnt in missing.items()
        )
        logger.warning(
            "Missing values detected in %d column(s) — imputation handled by pipeline:\n%s",
            len(missing),
            report,
        )


def _encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode 'Diagnosis' column from 'Positive'/'Negative' to 1/0.
    """
    if TARGET_COL in df.columns:
        if df[TARGET_COL].dtype == object:
            logger.info("Encoding target '%s' ('Positive'->1, 'Negative'->0)", TARGET_COL)
            mapping = {'Positive': 1, 'Negative': 0}
            # Check for unmapped values
            unmapped = set(df[TARGET_COL].dropna().unique()) - set(mapping.keys())
            if unmapped:
                raise ValueError(f"Found unexpected values in target column: {unmapped}")
            
            df[TARGET_COL] = df[TARGET_COL].map(mapping)
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_and_validate(csv_path: str | Path) -> pd.DataFrame:
    """
    Load the CSV, validate schema, dtypes, remove duplicates, encode target.
    """
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV file not found: '{csv_path}'. "
            "Please check the path and try again."
        )

    logger.info("Loading CSV from '%s' …", csv_path)
    df = pd.read_csv(csv_path)
    logger.info("Raw shape: %d rows × %d columns.", *df.shape)

    # --- Validation ---
    _validate_columns(df)
    _validate_dtypes(df)

    # --- Cleaning & Encoding ---
    df = _remove_duplicates(df)
    df = _encode_target(df)

    # --- Informational audit ---
    _report_missing_values(df)

    # --- Class distribution ---
    if TARGET_COL in df.columns:
        class_counts = df[TARGET_COL].value_counts()
        minority = class_counts.min()
        majority = class_counts.max()
        ratio    = majority / minority if minority > 0 else 0
        logger.info(
            "Class distribution — Negative: %d | Positive: %d | Imbalance ratio: %.1f:1",
            class_counts.get(0, 0),
            class_counts.get(1, 0),
            ratio,
        )

    logger.info("Data loading complete. Final shape: %d rows × %d columns.", *df.shape)
    return df


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    from config import DATA_RAW
    path = sys.argv[1] if len(sys.argv) > 1 else DATA_RAW
    df   = load_and_validate(path)
    print("\n--- Head (first 5 rows) ---")
    print(df.head())
    print("\n--- Target distribution ---")
    print(df[TARGET_COL].value_counts())
