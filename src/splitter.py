# =============================================================================
# src/splitter.py
# MEMBER 1 — Deadline: 26 April
# Role: Data Pipeline & Preprocessing Engineer
# =============================================================================
# Responsibility:
#   Stratified 70/15/15 split. Save all 6 split CSVs to DATA_SPLITS on the
#   FIRST run only — reload on subsequent runs to guarantee reproducibility.
#   The test set is written to disk and NEVER passed to any fit() call.
#
# Split strategy:
#   Step 1: train_test_split(X, y, test_size=TEST_SIZE, stratify=y)
#             → X_train_full (85%) + X_test (15%)
#   Step 2: train_test_split(X_train_full, y_train_full,
#                             test_size=VAL_SIZE, stratify=y_train_full)
#             → X_train (70%) + X_val (15%)
#
# Output files saved to DATA_SPLITS/:
#   X_train.csv, y_train.csv, X_val.csv, y_val.csv, X_test.csv, y_test.csv
#
# Checklist:
#   [ ] Skip splitting if all 6 CSVs already exist (idempotent)
#   [ ] Use stratify=y and RANDOM_STATE throughout
#   [ ] Print split sizes and class ratios per split after saving
#   [ ] Lock test set: print a warning if test split is loaded but caller
#       seems to be in a training context (optional safeguard)
# =============================================================================
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from config import DATA_SPLITS, TEST_SIZE, VAL_SIZE, RANDOM_STATE


def split_data(X: pd.DataFrame, y: pd.Series) -> None:
   
    Stratified 70/15/15 split. Saves CSVs to DATA_SPLITS on first run;
    skips if they already exist (idempotent).
   
    # TODO: implement
    raise NotImplementedError("splitter.py: split_data() not yet implemented.")
"""
"""
splitter.py
===========
Data Pipeline & Preprocessing Engineer — Deliverable D2

Responsibilities:
- Accept the clean DataFrame from data_loader.py
- Perform a stratified 70 / 15 / 15 train / validation / test split
- Immediately persist X_test and y_test to disk (never touched again until
  final evaluation)
- Return (X_train, X_val, X_test, y_train, y_val, y_test) as pandas objects

Author : Member 1
Project: Asthma Disease Detection — Phase III
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("splitter")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RANDOM_STATE: int = 42
TARGET_COLUMN: str = "Diagnosis"

# Split proportions
TRAIN_RATIO: float = 0.70
VAL_RATIO:   float = 0.15
TEST_RATIO:  float = 0.15  # 1 - TRAIN_RATIO - VAL_RATIO

DEFAULT_SPLITS_DIR = Path("data/splits")

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _log_split_stats(
    name: str,
    X: pd.DataFrame,
    y: pd.Series,
) -> None:
    """Log shape and class-balance info for a given split."""
    counts   = y.value_counts().sort_index()
    minority = counts.min()
    majority = counts.max()
    logger.info(
        "%-12s → %5d samples | features: %d | Neg: %d (%.1f%%) | Pos: %d (%.1f%%)",
        name,
        len(y),
        X.shape[1],
        counts.get(0, 0),
        counts.get(0, 0) / len(y) * 100,
        counts.get(1, 0),
        counts.get(1, 0) / len(y) * 100,
    )


def _save_split(X: pd.DataFrame, y: pd.Series, name: str, directory: Path) -> None:
    """
    Persist a feature matrix and label vector to disk using pickle.

    Naming convention: ``X_{name}.pkl`` and ``y_{name}.pkl``
    """
    directory.mkdir(parents=True, exist_ok=True)

    x_path = directory / f"X_{name}.pkl"
    y_path = directory / f"y_{name}.pkl"

    with open(x_path, "wb") as fh:
        pickle.dump(X, fh, protocol=pickle.HIGHEST_PROTOCOL)

    with open(y_path, "wb") as fh:
        pickle.dump(y, fh, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info("Saved %-12s → %s  |  %s", name, x_path, y_path)


def load_split(name: str, directory: str | Path = DEFAULT_SPLITS_DIR) -> tuple[pd.DataFrame, pd.Series]:
    """
    Reload a previously saved split from disk.

    Parameters
    ----------
    name : str
        One of ``'train'``, ``'val'``, ``'test'``.
    directory : str or Path
        Folder where splits are stored.

    Returns
    -------
    (X, y) : tuple[pd.DataFrame, pd.Series]
    """
    directory = Path(directory)

    x_path = directory / f"X_{name}.pkl"
    y_path = directory / f"y_{name}.pkl"

    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            f"Split '{name}' not found in '{directory}'. "
            "Run split_dataset() first."
        )

    with open(x_path, "rb") as fh:
        X = pickle.load(fh)
    with open(y_path, "rb") as fh:
        y = pickle.load(fh)

    logger.info("Loaded split '%s' — X: %s  y: %s", name, X.shape, y.shape)
    return X, y


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def split_dataset(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    splits_dir: str | Path = DEFAULT_SPLITS_DIR,
    save_all: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
           pd.Series,    pd.Series,    pd.Series]:
    """
    Perform a stratified 70 / 15 / 15 train / validation / test split.

    The **test set is written to disk immediately** after splitting and must
    not be used again until final model evaluation.

    Parameters
    ----------
    df : pd.DataFrame
        Clean DataFrame returned by ``data_loader.load_and_validate()``.
    target_column : str
        Name of the binary target column (default: ``'Diagnosis'``).
    splits_dir : str or Path
        Directory in which to persist the splits.
    save_all : bool
        If True, also persist train and val splits (useful for reproducibility).
        The test set is *always* persisted regardless of this flag.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test
        All returned as pandas DataFrames / Series with indices reset.

    Raises
    ------
    ValueError
        If ``target_column`` is not present in ``df``.
    """
    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in DataFrame. "
            f"Available columns: {df.columns.tolist()}"
        )

    splits_dir = Path(splits_dir)

    # Separate features from target
    # Drop PatientID — it is an identifier, not a predictive feature
    drop_cols = [target_column]
    if "PatientID" in df.columns:
        drop_cols.append("PatientID")
        logger.info("Dropping 'PatientID' — identifier column, not a feature.")

    X = df.drop(columns=drop_cols)
    y = df[target_column]

    logger.info(
        "Full dataset — %d samples | %d features | "
        "Negative: %d | Positive: %d",
        len(y),
        X.shape[1],
        (y == 0).sum(),
        (y == 1).sum(),
    )

    # -----------------------------------------------------------------------
    # Step 1: Carve out the held-out test set  (15%)
    # -----------------------------------------------------------------------
    val_test_ratio = VAL_RATIO + TEST_RATIO          # 0.30
    test_of_valtest = TEST_RATIO / val_test_ratio    # 0.50  → 15% of total

    X_train, X_valtest, y_train, y_valtest = train_test_split(
        X, y,
        test_size=val_test_ratio,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    # -----------------------------------------------------------------------
    # Step 2: Split the 30% remainder into 15% val + 15% test
    # -----------------------------------------------------------------------
    X_val, X_test, y_val, y_test = train_test_split(
        X_valtest, y_valtest,
        test_size=test_of_valtest,
        stratify=y_valtest,
        random_state=RANDOM_STATE,
    )

    # -----------------------------------------------------------------------
    # Reset indices for clean downstream indexing
    # -----------------------------------------------------------------------
    X_train = X_train.reset_index(drop=True)
    X_val   = X_val.reset_index(drop=True)
    X_test  = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_val   = y_val.reset_index(drop=True)
    y_test  = y_test.reset_index(drop=True)

    # -----------------------------------------------------------------------
    # Log split statistics
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Split summary (stratify=y, random_state=%d)", RANDOM_STATE)
    logger.info("=" * 60)
    _log_split_stats("TRAIN (70%)",      X_train, y_train)
    _log_split_stats("VALIDATION (15%)", X_val,   y_val)
    _log_split_stats("TEST (15%)",       X_test,  y_test)
    logger.info("=" * 60)

    # -----------------------------------------------------------------------
    # Verify proportions
    # -----------------------------------------------------------------------
    total = len(y_train) + len(y_val) + len(y_test)
    assert abs(len(y_train) / total - 0.70) < 0.02, "Train proportion out of range!"
    assert abs(len(y_val)   / total - 0.15) < 0.02, "Val proportion out of range!"
    assert abs(len(y_test)  / total - 0.15) < 0.02, "Test proportion out of range!"

    # -----------------------------------------------------------------------
    # Persist test set immediately — LOCKED from this point forward
    # -----------------------------------------------------------------------
    logger.info("Locking test set to disk …")
    _save_split(X_test, y_test, "test", splits_dir)
    logger.info(
        "TEST SET LOCKED. Do NOT use X_test / y_test until final evaluation."
    )

    # Optionally persist train and val splits as well
    if save_all:
        _save_split(X_train, y_train, "train", splits_dir)
        _save_split(X_val,   y_val,   "val",   splits_dir)

    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    from data_loader import load_and_validate

    csv_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/asthma_disease_data.csv"

    df = load_and_validate(csv_path)
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
        df, save_all=True
    )

    print(f"\nX_train: {X_train.shape}  y_train: {y_train.shape}")
    print(f"X_val  : {X_val.shape}  y_val  : {y_val.shape}")
    print(f"X_test : {X_test.shape}  y_test : {y_test.shape}")