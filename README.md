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
asthma_detection/                                    # Project root
│
├── README.md                                         # Project overview 
├── config.py                                         # Shared constants (ALREADY CREATED)
├── main.py                                           # Master script — runs entire pipeline
├── requirements.txt                                  # Dependencies: pip install -r requirements.txt
│
├── data/                                             
│   ├── raw/
│   │   └── asthma_dataset.csv                        # Original Kaggle CSV (download manually)
│   │
│   └── splits/                                       # Train/Val/Test splits (created by splitter.py)
│       ├── X_train.csv
│       ├── y_train.csv
│       ├── X_val.csv
│       ├── y_val.csv
│       ├── X_test.csv
│       └── y_test.csv
│
├── src/                                              # MEMBERS WRITE THEIR CODE HERE (.py files)
│   │
│   ├── [MEMBER 1 - Deadline: 26 April]
│   ├── data_loader.py                                # Load, validate, clean dataset
│   ├── splitter.py                                   # Stratified 70/15/15 split, save to disk
│   ├── preprocessing.py                              # Build ColumnTransformer (dual-branch)
│   ├── cv_pipeline.py                                # Wrap with SMOTE inside imblearn Pipeline
│   │
│   ├── [MEMBER 2 - Deadline: 28 April]
│   ├── models.py                                     # Define 4 classifiers (XGB, LGBM, LogReg, RF)
│   ├── tuning.py                                     # Optuna Bayesian search (100 trials per model)
│   ├── cross_validate.py                             # Stratified 5-fold CV, report metrics
│   ├── save_model.py                                 # Fit best model, serialize with joblib
│   │
│   ├── [MEMBER 3 - Deadline: 30 April]
│   ├── threshold.py                                  # Sweep τ on validation set, find τ*
│   ├── evaluate.py                                   # Final eval on locked test set (ONE PASS ONLY)
│   ├── visualize.py                                  # ROC, PR, confusion matrix, threshold plots
│   ├── shap_analysis.py                              # Global + local SHAP explanations
│   │
│   ├── [MEMBER 4 - Deadline: 1 May]
│   ├── pipeline.py                                   # Assemble unified ColumnTransformer→SMOTE→Clf
│   └── inference.py                                  # Clinical inference endpoint (one patient → risk + explanation)
│
├── models/                                           # Serialized models
│   └── best_model.pkl                                # Fitted pipeline saved by save_model.py (joblib)
│
├── figures/                                          # All output visualizations (300 DPI)
│   ├── roc_curves.png                                # ROC curves for all 4 models
│   ├── pr_curves.png                                 # Precision-Recall curves
│   ├── confusion_matrix.png                          # Confusion matrix heatmap (best model)
│   ├── threshold_sweep.png                           # F1 and Recall vs threshold τ
│   ├── class_distribution.png                        # Before/after SMOTE bar chart
│   ├── shap_beeswarm.png                             # Global SHAP beeswarm
│   ├── shap_bar.png                                  # Global SHAP feature importance
│   ├── shap_waterfall_tp.png                         # Local SHAP for true positive
│   ├── shap_waterfall_fn.png                         # Local SHAP for false negative
│   └── shap_waterfall_fp.png                         # Local SHAP for false positive 
│
├── notebooks/                                        # For EXPLORATION ONLY (optional, not submitted)
│   └── 01_EDA.ipynb                            
│
│
└── reports/                                          # Final report and documentation
    └── Final_Report.pdf                              # Comprehensive report 
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
