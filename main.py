# =============================================================================
# main.py — Master reproducibility script
# Member 4 deliverable D4
# Usage:
#   python main.py           # full run (100-trial Optuna search)
#   python main.py --fast    # skip Optuna, load cached best_params.json
# =============================================================================
# Execution order:
#   1.  Load & validate data           (data_loader)
#   2.  Split 70/15/15 — idempotent    (splitter)     ← skipped if splits exist
#   3.  Build ColumnTransformer        (preprocessing)
#   4.  Wrap with SMOTE pipeline       (cv_pipeline)
#   5.  Define 4 classifiers           (models)
#   6.  Optuna tuning OR load cache    (tuning)        ← skipped with --fast
#   7.  Stratified 5-fold CV           (cross_validate)
#   8.  Fit + save best model          (save_model / pipeline)
#   9.  Threshold sweep on val set     (threshold)
#   10. ONE-PASS test evaluation       (evaluate)
#   11. Generate all plots             (visualize)
#   12. SHAP analysis                  (shap_analysis)
#   13. Print final summary table
# =============================================================================

import sys
import os
import argparse
import json

sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader    import load_data
from src.splitter       import split_data
from src.preprocessing  import build_preprocessor
from src.cv_pipeline    import build_cv_pipeline
from src.models         import get_models
from src.tuning         import run_tuning
from src.cross_validate import run_cross_validation
from src.save_model     import fit_and_save
from src.threshold      import find_best_threshold
from src.evaluate       import evaluate_on_test
from src.visualize      import generate_all_plots
from src.shap_analysis  import run_shap
from config             import BASE_DIR


CACHED_PARAMS_PATH = os.path.join(BASE_DIR, "models", "best_params.json")


def parse_args():
    parser = argparse.ArgumentParser(description="Asthma Detection Pipeline")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Skip Optuna search and load cached best_params.json instead.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 65)
    print("  Asthma Detection Pipeline")
    print(f"  Mode: {'FAST (cached params)' if args.fast else 'FULL (Optuna search)'}")
    print("=" * 65)

    # ── Step 1: Load data ─────────────────────────────────────────────────────
    print("\n[1/12] Loading & validating data...")
    X, y = load_data()

    # ── Step 2: Split (idempotent — skips if CSVs already exist) ─────────────
    print("[2/12] Stratified 70/15/15 split...")
    split_data(X, y)

    # ── Step 3: Preprocessor ──────────────────────────────────────────────────
    print("[3/12] Building ColumnTransformer (numeric / nominal / binary)...")
    preprocessor = build_preprocessor()

    # ── Step 4: SMOTE pipeline ────────────────────────────────────────────────
    print("[4/12] Wrapping with SMOTE (imblearn Pipeline)...")
    cv_pipe = build_cv_pipeline(preprocessor)

    # ── Step 5: Define models ─────────────────────────────────────────────────
    print("[5/12] Defining 4 classifiers (XGB, LGBM, LogReg, RF)...")
    models = get_models()

    # ── Step 6: Hyperparameter tuning (or load cache) ─────────────────────────
    if args.fast and os.path.exists(CACHED_PARAMS_PATH):
        print("[6/12] --fast flag: loading cached best_params.json...")
        with open(CACHED_PARAMS_PATH, "r") as f:
            best_params = json.load(f)
    else:
        print("[6/12] Running Optuna Bayesian search (100 trials × 4 models)...")
        best_params = run_tuning(cv_pipe, models)
        os.makedirs(os.path.dirname(CACHED_PARAMS_PATH), exist_ok=True)
        with open(CACHED_PARAMS_PATH, "w") as f:
            json.dump(best_params, f, indent=2)
        print(f"       Cached to {CACHED_PARAMS_PATH}")

    # ── Step 7: Cross-validation ──────────────────────────────────────────────
    print("[7/12] Stratified 5-fold CV on all 4 models...")
    cv_results = run_cross_validation(cv_pipe, models, best_params)

    # ── Step 8: Fit + save best model ─────────────────────────────────────────
    print("[8/12] Fitting best model and serializing to disk...")
    best_model = fit_and_save(cv_pipe, models, best_params, cv_results)

    # ── Step 9: Threshold calibration ─────────────────────────────────────────
    print("[9/12] Sweeping threshold on validation set...")
    tau_star = find_best_threshold(best_model)

    # ── Step 10: Locked test evaluation (ONE PASS) ────────────────────────────
    print("[10/12] Final evaluation on locked test set (one pass only)...")
    test_metrics = evaluate_on_test(best_model, tau_star)

    # ── Step 11: Visualizations ───────────────────────────────────────────────
    print("[11/12] Generating figures...")
    generate_all_plots(best_model, models, best_params, cv_results, tau_star)

    # ── Step 12: SHAP analysis ────────────────────────────────────────────────
    print("[12/12] Running SHAP analysis...")
    run_shap(best_model, tau_star)

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  Pipeline complete. Final Test Metrics:")
    print("=" * 65)
    for metric, value in test_metrics.items():
        if metric != "confusion_matrix":
            print(f"  {metric:<20} {value:.4f}")
    if "confusion_matrix" in test_metrics:
        cm = test_metrics["confusion_matrix"]
        print(f"\n  Confusion Matrix:")
        print(f"    TN={cm[0][0]}  FP={cm[0][1]}")
        print(f"    FN={cm[1][0]}  TP={cm[1][1]}")
    print(f"\n  τ* (threshold) = {tau_star:.2f}")
    print("=" * 65)


if __name__ == "__main__":
    main()
