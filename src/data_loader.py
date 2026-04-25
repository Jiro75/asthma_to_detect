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

"""
import pandas as pd
from config import DATA_RAW, TARGET_COL, NUMERIC_FEATURES, NOMINAL_FEATURES, BINARY_FEATURES

EXPECTED_FEATURES = NUMERIC_FEATURES + NOMINAL_FEATURES + BINARY_FEATURES  # 26 columns


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    
    Load raw CSV, validate schema and dtypes, drop duplicates.

    Returns
    -------
    X : pd.DataFrame  — feature matrix (26 columns)
    y : pd.Series     — binary target
    
    #TODO: implement
    raise NotImplementedError("data_loader.py: load_data() not yet implemented.")
"""
"""
data_loader.py
==============
Data Pipeline & Preprocessing Engineer — Deliverable D1

Responsibilities:
- Ingest the Kaggle asthma CSV
- Validate schema (26 expected feature columns, correct dtypes)
- Remove exact duplicate rows
- Return a clean pandas DataFrame

Author : Member 1
Project: Asthma Disease Detection — Phase III
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path

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
# All 26 feature columns (target 'Diagnosis' excluded — handled separately)
EXPECTED_FEATURE_COLUMNS: list[str] = [
    "PatientID",
    "Age",
    "Gender",
    "Ethnicity",
    "EducationLevel",
    "BMI",
    "Smoking",
    "PhysicalActivity",
    "DietQuality",
    "SleepQuality",
    "PollutionExposure",
    "PollenExposure",
    "DustExposure",
    "PetAllergy",
    "FamilyHistoryAsthma",
    "HistoryOfAllergies",
    "Eczema",
    "HayFever",
    "GastroesophagealReflux",
    "LungFunctionFEV1",
    "LungFunctionFVC",
    "Wheezing",
    "ShortnessOfBreath",
    "ChestTightness",
    "Coughing",
    "NighttimeSymptoms",
    "ExerciseInduced",
]

TARGET_COLUMN: str = "Diagnosis"

# Expected dtypes per column (use broad categories: 'numeric' / 'object')
DTYPE_EXPECTATIONS: dict[str, str] = {
    "PatientID":              "numeric",
    "Age":                    "numeric",
    "Gender":                 "numeric",
    "Ethnicity":              "numeric",
    "EducationLevel":         "numeric",
    "BMI":                    "numeric",
    "Smoking":                "numeric",
    "PhysicalActivity":       "numeric",
    "DietQuality":            "numeric",
    "SleepQuality":           "numeric",
    "PollutionExposure":      "numeric",
    "PollenExposure":         "numeric",
    "DustExposure":           "numeric",
    "PetAllergy":             "numeric",
    "FamilyHistoryAsthma":    "numeric",
    "HistoryOfAllergies":     "numeric",
    "Eczema":                 "numeric",
    "HayFever":               "numeric",
    "GastroesophagealReflux": "numeric",
    "LungFunctionFEV1":       "numeric",
    "LungFunctionFVC":        "numeric",
    "Wheezing":               "numeric",
    "ShortnessOfBreath":      "numeric",
    "ChestTightness":         "numeric",
    "Coughing":               "numeric",
    "NighttimeSymptoms":      "numeric",
    "ExerciseInduced":        "numeric",
    "Diagnosis":              "numeric",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_columns(df: pd.DataFrame) -> None:
    """
    Ensure all expected feature columns AND the target column are present.
    Raises ValueError with a descriptive message if any are missing.
    """
    all_expected = set(EXPECTED_FEATURE_COLUMNS + [TARGET_COLUMN])
    present      = set(df.columns)
    missing      = all_expected - present
    extra        = present - all_expected

    if missing:
        raise ValueError(
            f"Schema validation failed — {len(missing)} column(s) missing from CSV:\n"
            f"  Missing : {sorted(missing)}\n"
            f"  Present : {sorted(present)}"
        )

    if extra:
        # Log a warning but do NOT raise — extra columns are harmless
        logger.warning(
            "Unexpected extra column(s) found and will be retained: %s", sorted(extra)
        )

    logger.info("Column validation passed — all %d expected columns present.", len(all_expected))


def _validate_dtypes(df: pd.DataFrame) -> None:
    """
    Check that each column's dtype matches its expectation ('numeric' or 'object').
    Raises TypeError listing every violating column.
    """
    violations: list[str] = []

    for col, expected_kind in DTYPE_EXPECTATIONS.items():
        if col not in df.columns:
            continue  # already caught by _validate_columns

        actual_kind = "numeric" if pd.api.types.is_numeric_dtype(df[col]) else "object"
        if actual_kind != expected_kind:
            violations.append(
                f"  '{col}': expected {expected_kind}, got {df[col].dtype}"
            )

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
    Log a summary of missing values per column (informational only — imputation
    is handled downstream in the ColumnTransformer pipeline).
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_and_validate(csv_path: str | Path) -> pd.DataFrame:
    """
    Load the Kaggle asthma CSV, validate its schema and dtypes, remove
    exact duplicates, and return a clean DataFrame.

    Parameters
    ----------
    csv_path : str or Path
        Path to the raw CSV file (e.g., ``data/raw/asthma_disease_data.csv``).

    Returns
    -------
    pd.DataFrame
        Validated, deduplicated DataFrame with all 26 feature columns
        and the 'Diagnosis' target column.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist at the given path.
    ValueError
        If required columns are missing.
    TypeError
        If column dtypes do not match expectations.
    """
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV file not found: '{csv_path}'. "
            "Place the raw Kaggle download in data/raw/ and try again."
        )

    logger.info("Loading CSV from '%s' …", csv_path)
    df = pd.read_csv(csv_path)
    logger.info("Raw shape: %d rows × %d columns.", *df.shape)

    # --- Validation ---
    _validate_columns(df)
    _validate_dtypes(df)

    # --- Cleaning ---
    df = _remove_duplicates(df)

    # --- Informational audit ---
    _report_missing_values(df)

    # --- Class distribution ---
    if TARGET_COLUMN in df.columns:
        class_counts = df[TARGET_COLUMN].value_counts()
        minority = class_counts.min()
        majority = class_counts.max()
        ratio    = majority / minority
        logger.info(
            "Class distribution — Negative: %d | Positive: %d | Imbalance ratio: %.1f:1",
            class_counts.get(0, 0),
            class_counts.get(1, 0),
            ratio,
        )

    logger.info("Data loading complete. Final shape: %d rows × %d columns.", *df.shape)
    return df


# ---------------------------------------------------------------------------
# CLI entry-point (quick sanity check)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    # HOW TO RUN:
    #   python src/data_loader.py
    #   python src/data_loader.py data/raw/asthma_disease_data.csv
    path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/asthma_disease_data.csv"
    df   = load_and_validate(path)
    print("\n--- Head (first 5 rows) ---")
    print(df.head())
    print("\n--- Schema info ---")
    df.info()
    print("\n--- Target distribution ---")
    print(df[TARGET_COLUMN].value_counts())
# ---------------------------------------------------------------------------
# UPDATED MAIN — HOW TO RUN:
#   python src/data_loader.py
#   python src/data_loader.py path/to/asthma_disease_data.csv
# ---------------------------------------------------------------------------
