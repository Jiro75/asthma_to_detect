"""
config.py
---------
Single source of truth for all project-wide constants.
Every member imports from here — never hardcode paths or seeds elsewhere.
"""

import os

# ── Reproducibility ────────────────────────────────────────────────────────────
RANDOM_STATE = 42

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_RAW    = os.path.join(BASE_DIR, "data", "raw", "asthma_dataset.csv")
DATA_SPLITS = os.path.join(BASE_DIR, "data", "splits")
MODEL_PATH  = os.path.join(BASE_DIR, "models", "best_model.pkl")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")

# ── Split Ratios ───────────────────────────────────────────────────────────────
TEST_SIZE       = 0.15   # 15% for test
VAL_SIZE        = 0.1765 # ~15% of remaining 85% ≈ 15% of total

# ── Class Imbalance ────────────────────────────────────────────────────────────
SCALE_POS_WEIGHT = 18.3  # negative-to-positive ratio in training set
SMOTE_RATIO_MIN  = 0.3
SMOTE_RATIO_MAX  = 0.5
SMOTE_RATIO_DEFAULT = 0.4

# ── Cross-Validation ───────────────────────────────────────────────────────────
CV_FOLDS    = 5
CV_SCORING  = "roc_auc"

# ── Optuna ─────────────────────────────────────────────────────────────────────
OPTUNA_TRIALS          = 100
OPTUNA_EARLY_STOP      = 50   # early stopping rounds for XGB/LGBM inside trials

# ── Threshold Sweep ────────────────────────────────────────────────────────────
THRESHOLD_MIN  = 0.10
THRESHOLD_MAX  = 0.90
THRESHOLD_STEP = 0.01

# ── Feature Columns ────────────────────────────────────────────────────────────
TARGET_COL = "Diagnosis"

NUMERIC_FEATURES = [
    "Age", "BMI", "LungFunctionFEV1", "LungFunctionFVC",
    "PhysicalActivity", "DietQuality", "SleepQuality",
    "PollutionExposure", "PollenExposure", "DustExposure",
]

NOMINAL_FEATURES = [        # → One-Hot Encoding
    "Gender", "Ethnicity", "EducationLevel",
]

BINARY_FEATURES = [         # already 0/1, pass-through or ordinal
    "Smoking", "PetAllergy", "FamilyHistoryAsthma",
    "HistoryOfAllergies", "Eczema", "HayFever",
    "GastroesophagealReflux", "Wheezing", "ShortnessOfBreath",
    "ChestTightness", "Coughing", "NighttimeSymptoms", "ExerciseInduced",
]
