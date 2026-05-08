import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (roc_curve, auc, precision_recall_curve, f1_score,
                             recall_score, precision_score, roc_auc_score, confusion_matrix)

from config import DATA_SPLITS, BASE_DIR, SMOTE_RATIO_DEFAULT, FIGURES_DIR
from src.save_model import _reconstruct_classifier
from src.cv_pipeline import build_cv_pipeline
from src.preprocessing import build_preprocessor_top25
from src.threshold import find_best_threshold
from src.models import get_models
from src.data_loader import load_data
from src.splitter import split_data

def main():
    # Ensure splits exist
    if not os.path.exists(os.path.join(DATA_SPLITS, "X_train.pkl")):
        print("Splits not found. Loading and splitting data...")
        X, y = load_data()
        split_data(X, y)

    models_dict = get_models()
    params_path = os.path.join(BASE_DIR, "models", "best_params.json")
    with open(params_path, "r") as f:
        best_params = json.load(f)

    # Load splits
    with open(os.path.join(DATA_SPLITS, "X_train.pkl"), "rb") as f: X_train = pickle.load(f)
    with open(os.path.join(DATA_SPLITS, "y_train.pkl"), "rb") as f: y_train = pickle.load(f)
    with open(os.path.join(DATA_SPLITS, "X_test.pkl"), "rb") as f: X_test = pickle.load(f)
    with open(os.path.join(DATA_SPLITS, "y_test.pkl"), "rb") as f: y_test = pickle.load(f)

    results = {}
    fitted_pipelines = {}

    plt.figure(figsize=(10, 8))
    roc_fig = plt.gca()
    plt.figure(figsize=(10, 8))
    pr_fig = plt.gca()

    for name in models_dict.keys():
        print(f"\\n--- Evaluating {name} ---")
        clf = _reconstruct_classifier(name, best_params)
        pipe = build_cv_pipeline(clf, smote_ratio=SMOTE_RATIO_DEFAULT, preprocessor=build_preprocessor_top25())
        pipe.fit(X_train, y_train)
        fitted_pipelines[name] = pipe
        
        tau_star = find_best_threshold(pipe)
        
        proba = pipe.predict_proba(X_test)[:, 1]
        y_pred = (proba >= tau_star).astype(int)
        
        auc_score = roc_auc_score(y_test, proba)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        prec = precision_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        
        results[name] = {
            "AUC-ROC": auc_score,
            "F1": f1,
            "Recall": rec,
            "Precision": prec,
            "Tau Star": tau_star,
            "Confusion Matrix": cm.tolist()
        }
        print(f"AUC: {auc_score:.4f}, F1: {f1:.4f}, Recall: {rec:.4f}, Precision: {prec:.4f}, Tau*: {tau_star:.2f}")
        print(f"CM:\\n{cm}")

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, proba)
        roc_fig.plot(fpr, tpr, label=f"{name} (AUC = {auc_score:.3f})")

        # PR Curve
        prec_curve, rec_curve, _ = precision_recall_curve(y_test, proba)
        pr_fig.plot(rec_curve, prec_curve, label=name)

    # Finalize ROC
    roc_fig.plot([0, 1], [0, 1], "k--", alpha=0.5)
    roc_fig.set_xlabel("False Positive Rate")
    roc_fig.set_ylabel("True Positive Rate")
    roc_fig.set_title("ROC Curves — All Models (Locked Test Set)")
    roc_fig.legend(loc="lower right")
    roc_fig.grid(alpha=0.3)
    roc_fig.figure.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    roc_fig.figure.savefig(os.path.join(FIGURES_DIR, "roc_curves_all.png"), dpi=300)

    # Finalize PR
    pr_fig.set_xlabel("Recall")
    pr_fig.set_ylabel("Precision")
    pr_fig.set_title("Precision-Recall Curves (Locked Test Set)")
    pr_fig.legend(loc="lower left")
    pr_fig.grid(alpha=0.3)
    pr_fig.figure.tight_layout()
    pr_fig.figure.savefig(os.path.join(FIGURES_DIR, "pr_curves_all.png"), dpi=300)

    # --- Combined Figure ---
    fig_comb, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Re-plot on combined figure
    for name in models_dict.keys():
        proba = fitted_pipelines[name].predict_proba(X_test)[:, 1]
        
        # ROC
        fpr, tpr, _ = roc_curve(y_test, proba)
        roc_auc = roc_auc_score(y_test, proba)
        ax_roc.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")
        
        # PR
        prec_curve, rec_curve, _ = precision_recall_curve(y_test, proba)
        ax_pr.plot(rec_curve, prec_curve, label=name)

    # Style combined ROC
    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curves (Locked Test Set)")
    ax_roc.legend(loc="lower right")
    ax_roc.grid(alpha=0.3)

    # Style combined PR
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-Recall Curves (Locked Test Set)")
    ax_pr.legend(loc="lower left")
    ax_pr.grid(alpha=0.3)

    fig_comb.tight_layout()
    fig_comb.savefig(os.path.join(FIGURES_DIR, "combined_curves.png"), dpi=300)
    plt.close(fig_comb)

    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\\nAll done. evaluation_results.json and plots saved.")

if __name__ == "__main__":
    main()
