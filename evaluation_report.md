# Model Evaluation Report

## 1. Metrics & Confusion Matrix

| Model | AUC-ROC | F1-Score | Recall | Precision | $\tau^*$ | Confusion Matrix (TN, FP, FN, TP) |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **XGBoost** | `0.9963` | `0.9573` | `0.9492` | `0.9655` | `0.80` | TN=121, FP=2, FN=3, TP=56 |
| **LightGBM** | `0.9963` | `0.9573` | `0.9492` | `0.9655` | `0.15` | TN=121, FP=2, FN=3, TP=56 |
| **Logistic Regression** | `0.9822` | `0.9016` | `0.9322` | `0.8730` | `0.58` | TN=115, FP=8, FN=4, TP=55 |
| **Random Forest** | `0.9950` | `0.9194` | `0.9661` | `0.8769` | `0.47` | TN=115, FP=8, FN=2, TP=57 |

## 2. Comparative Plots

### ROC Curves (All 4 Models)
![ROC Curves](figures/roc_curves_all.png)

### Precision-Recall Curves (All 4 Models)
![Precision Recall Curves](figures/pr_curves_all.png)
