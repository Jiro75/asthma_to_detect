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

from sklearn.compose import ColumnTransformer
from config import NUMERIC_FEATURES, NOMINAL_FEATURES, BINARY_FEATURES


def build_preprocessor() -> ColumnTransformer:
    """
    Build and return an unfitted dual-branch ColumnTransformer.

    Branches: numeric (impute→power→scale), nominal (impute→OHE), binary (impute→ordinal)
    """
    # TODO: implement
    raise NotImplementedError("preprocessing.py: build_preprocessor() not yet implemented.")
