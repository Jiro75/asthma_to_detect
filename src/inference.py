# =============================================================================
# src/inference.py
# MEMBER 4 — Deadline: 1 May
# Role: Integration Lead & Inference Engineer
# =============================================================================
# Responsibility:
#   D3: Clinical inference endpoint. A physician or clinical system sends ONE
#   patient record (Python dict) and receives a risk score, binary decision,
#   and SHAP-based feature attribution — the full clinical deployment layer.
#
# ⚠️  Single patient only: accept a dict, convert to 1-row DataFrame with
#     pd.DataFrame([patient_dict]) BEFORE passing to the pipeline.
#     Do NOT design this for batches.
#
# Return schema:
#   {
#     "risk_score"      : float,       # predicted P(asthma)
#     "label"           : str,         # "Asthma Positive" | "Asthma Negative"
#     "threshold_used"  : float,       # tau_star value
#     "top_features"    : [            # SHAP attribution, top 5
#         {"feature": "LungFunctionFEV1", "shap_value": +0.42},
#         ...
#     ]
#   }
#
# SHAP for single patient:
#   X_transformed = loaded_model['preprocessor'].transform(patient_df)
#   explainer = shap.TreeExplainer(loaded_model['clf'])
#   shap_vals = explainer(X_transformed)
#   feature_names = loaded_model['preprocessor'].get_feature_names_out()
#
# Checklist:
#   [ ] Load model from model_path with joblib.load (fresh load, no session state)
#   [ ] Convert patient_dict → pd.DataFrame([patient_dict])
#   [ ] predict_proba → apply threshold → derive label string
#   [ ] Compute SHAP for single sample; extract top 5 by |shap_value|
#   [ ] Return structured dict (schema above)
#   [ ] __main__ block: demo call with a sample patient dict
# =============================================================================

import joblib
import pandas as pd
from config import MODEL_PATH


def predict_patient(patient_dict: dict,
                    model_path: str = MODEL_PATH,
                    threshold: float = 0.5) -> dict:
    """
    Run clinical inference for a single patient.

    Parameters
    ----------
    patient_dict : dict  — {feature_name: value} for one patient
    model_path   : str   — path to serialized pipeline (.pkl)
    threshold    : float — classification threshold τ* from threshold.py

    Returns
    -------
    dict with keys: risk_score, label, threshold_used, top_features
    """
    # TODO: implement
    raise NotImplementedError("inference.py: predict_patient() not yet implemented.")


if __name__ == "__main__":
    # Smoke-test with a sample patient — update feature values to match dataset
    sample = {
        "Age": 45,
        "BMI": 27.5,
        "LungFunctionFEV1": 72.0,
        "LungFunctionFVC": 85.0,
        "PhysicalActivity": 4.0,
        "DietQuality": 5.0,
        "SleepQuality": 6.0,
        "PollutionExposure": 3.0,
        "PollenExposure": 2.0,
        "DustExposure": 4.0,
        "Gender": 1,
        "Ethnicity": 0,
        "EducationLevel": 2,
        "Smoking": 1,
        "PetAllergy": 0,
        "FamilyHistoryAsthma": 1,
        "HistoryOfAllergies": 1,
        "Eczema": 0,
        "HayFever": 1,
        "GastroesophagealReflux": 0,
        "Wheezing": 1,
        "ShortnessOfBreath": 1,
        "ChestTightness": 0,
        "Coughing": 1,
        "NighttimeSymptoms": 0,
        "ExerciseInduced": 1,
    }
    result = predict_patient(sample, threshold=0.35)
    print(result)
