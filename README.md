# Asthma Detection — ML Pipeline

A modular machine learning pipeline for asthma risk prediction using clinical tabular data.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

## Project Structure

```
asthma_detection/
├── config.py           # Shared constants
├── main.py             # Master script — runs entire pipeline
├── src/                # Module code (members write here)
├── data/               # Raw CSV + train/val/test splits
├── models/             # Serialized best model (.pkl)
├── figures/            # All output plots (300 DPI)
├── notebooks/          # Optional EDA notebooks
└── reports/            # Final report PDF
```

## Pipeline

1. Load & validate data → `src/data_loader.py`
2. Stratified 70/15/15 split → `src/splitter.py`
3. ColumnTransformer preprocessing → `src/preprocessing.py`
4. SMOTE imbalance handling → `src/cv_pipeline.py`
5. Define 4 classifiers (XGB, LGBM, LogReg, RF) → `src/models.py`
6. Optuna Bayesian tuning (100 trials) → `src/tuning.py`
7. Stratified 5-fold CV → `src/cross_validate.py`
8. Fit + serialize best model → `src/save_model.py`
9. Threshold sweep on val set → `src/threshold.py`
10. Final locked test evaluation → `src/evaluate.py`
11. Plots (ROC, PR, CM, threshold) → `src/visualize.py`
12. SHAP global + local explanations → `src/shap_analysis.py`

## Team Deadlines

| Member | Files | Deadline |
|--------|-------|----------|
| 1 | data_loader, splitter, preprocessing, cv_pipeline | 26 April |
| 2 | models, tuning, cross_validate, save_model | 28 April |
| 3 | threshold, evaluate, visualize, shap_analysis | 30 April |
| 4 | pipeline, inference | 1 May |
