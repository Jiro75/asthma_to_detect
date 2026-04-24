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

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from config import DATA_SPLITS, TEST_SIZE, VAL_SIZE, RANDOM_STATE


def split_data(X: pd.DataFrame, y: pd.Series) -> None:
    """
    Stratified 70/15/15 split. Saves CSVs to DATA_SPLITS on first run;
    skips if they already exist (idempotent).
    """
    # TODO: implement
    raise NotImplementedError("splitter.py: split_data() not yet implemented.")
