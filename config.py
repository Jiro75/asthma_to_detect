"""
config.py
---------
Single source of truth for all project-wide constants.
Every member imports from here — never hardcode paths or seeds elsewhere.

Dataset: asthma_detection_dataset.csv  (1211 rows × 113 columns)
Target : Diagnosis  ('Positive' = 1, 'Negative' = 0)
Imbalance: ~2.1 : 1  (Negative 819 / Positive 392)
"""

import os

# ── Reproducibility ────────────────────────────────────────────────────────────
RANDOM_STATE = 42

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_RAW    = os.path.join(BASE_DIR, "asthma_detection_dataset.csv")
DATA_SPLITS = os.path.join(BASE_DIR, "data", "splits")
MODEL_PATH  = os.path.join(BASE_DIR, "models", "best_model.pkl")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")

# ── Split Ratios ───────────────────────────────────────────────────────────────
TEST_SIZE = 0.15    # 15% for test
VAL_SIZE  = 0.1765  # ~15% of remaining 85% ≈ 15% of total

# ── Class Imbalance ────────────────────────────────────────────────────────────
# Negative: 819  |  Positive: 392  |  ratio ≈ 2.1 : 1
SCALE_POS_WEIGHT    = 2.1
SMOTE_RATIO_MIN     = 0.5
SMOTE_RATIO_MAX     = 0.9
SMOTE_RATIO_DEFAULT = 0.7   # gentler oversampling — classes are not severely imbalanced

# ── Cross-Validation ───────────────────────────────────────────────────────────
CV_FOLDS   = 5
CV_SCORING = "roc_auc"

# ── Optuna ─────────────────────────────────────────────────────────────────────
OPTUNA_TRIALS     = 100
OPTUNA_EARLY_STOP = 50   # early stopping rounds for XGB/LGBM inside trials

# ── Threshold Sweep ────────────────────────────────────────────────────────────
THRESHOLD_MIN  = 0.10
THRESHOLD_MAX  = 0.90
THRESHOLD_STEP = 0.01

# ── Target Column ──────────────────────────────────────────────────────────────
TARGET_COL = "Diagnosis"   # raw values: 'Positive' / 'Negative'
# Encoding: 'Positive' → 1,  'Negative' → 0

# ── Columns to drop before modelling ──────────────────────────────────────────
# Audio_File  : raw filename  (identifier)
# Patient_ID  : patient identifier
# Condition   : multi-class label — would leak the target
# Severity    : severity label — would leak the target
DROP_COLS = ["Audio_File", "Patient_ID", "Condition", "Severity"]

# ── Feature Columns (used by preprocessing.py) ────────────────────────────────

# Continuous / truly numeric features — benefit from Yeo-Johnson + scaling
NUMERIC_FEATURES = [
    # --- Clinical vitals & lab values ---
    "Age",
    "BMI",
    "Blood_Pressure_Systolic",
    "Blood_Pressure_Diastolic",
    "Heart_Rate",
    "Glucose_Level",
    "Cholesterol",
    "Oxygen_Saturation",
    "Respiratory_Rate",
    "Cough_Frequency",
    "Peak_Expiratory_Flow",
    "FEV1_FVC_Ratio",
    "Eosinophil_Count",
    "IgE_Level",
    "Wheezing_Frequency",
    "Air_Pollution_Index",
    # --- Audio / MFCC features ---
    "MFCC_1_mean",  "MFCC_1_std",
    "MFCC_2_mean",  "MFCC_2_std",
    "MFCC_3_mean",  "MFCC_3_std",
    "MFCC_4_mean",  "MFCC_4_std",
    "MFCC_5_mean",  "MFCC_5_std",
    "MFCC_6_mean",  "MFCC_6_std",
    "MFCC_7_mean",  "MFCC_7_std",
    "MFCC_8_mean",  "MFCC_8_std",
    "MFCC_9_mean",  "MFCC_9_std",
    "MFCC_10_mean", "MFCC_10_std",
    "MFCC_11_mean", "MFCC_11_std",
    "MFCC_12_mean", "MFCC_12_std",
    "MFCC_13_mean", "MFCC_13_std",
    "Delta_MFCC_1_mean",  "Delta_MFCC_2_mean",  "Delta_MFCC_3_mean",
    "Delta_MFCC_4_mean",  "Delta_MFCC_5_mean",  "Delta_MFCC_6_mean",
    "Delta_MFCC_7_mean",  "Delta_MFCC_8_mean",  "Delta_MFCC_9_mean",
    "Delta_MFCC_10_mean", "Delta_MFCC_11_mean", "Delta_MFCC_12_mean",
    "Delta_MFCC_13_mean",
    "Spectral_Centroid_mean", "Spectral_Centroid_std",
    "Spectral_Bandwidth_mean", "Spectral_Bandwidth_std",
    "Spectral_Rolloff_mean",   "Spectral_Rolloff_std",
    "Spectral_Contrast_1_mean", "Spectral_Contrast_2_mean",
    "Spectral_Contrast_3_mean", "Spectral_Contrast_4_mean",
    "Spectral_Contrast_5_mean", "Spectral_Contrast_6_mean",
    "Spectral_Contrast_7_mean",
    "Spectral_Flatness_mean",
    "ZCR_mean", "ZCR_std",
    "RMS_Energy_mean", "RMS_Energy_std",
    "Chroma_1_mean",  "Chroma_2_mean",  "Chroma_3_mean",
    "Chroma_4_mean",  "Chroma_5_mean",  "Chroma_6_mean",
    "Chroma_7_mean",  "Chroma_8_mean",  "Chroma_9_mean",
    "Chroma_10_mean", "Chroma_11_mean", "Chroma_12_mean",
    "Tonnetz_1_mean", "Tonnetz_2_mean", "Tonnetz_3_mean",
    "Tonnetz_4_mean", "Tonnetz_5_mean", "Tonnetz_6_mean",
    "Mel_Spec_mean", "Mel_Spec_std", "Mel_Spec_max", "Mel_Spec_min",
    "Tempo",
    "Duration_sec",
]

# Nominal categorical features — encoded via OneHotEncoder
NOMINAL_FEATURES = [
    "Gender",             # Male / Female
    "Smoking_Status",     # Never / Current / Former
    "Physical_Activity",  # Low / Moderate / High
    "Dust_Exposure_Level",# Low / Medium / High
    "Primary_Symptom",    # 11 categories + NaN
    "Medication_Use",     # Inhaler_Only / Inhaler_and_Oral / Steroid + NaN
]

# Binary (0/1 integer) features — pass through OrdinalEncoder, NOT OHE
BINARY_FEATURES = [
    "Family_History",
    "Allergy",
    "Exercise_Induced_Symptoms",
    "Chest_Tightness",
    "Nighttime_Symptoms",
]
