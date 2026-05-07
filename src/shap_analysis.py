# =============================================================================
# src/shap_analysis.py
# MEMBER 3 — Deadline: 30 April
# Role: Evaluation, SHAP Interpretability & Visualization Engineer
# =============================================================================

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from config import FIGURES_DIR, DATA_SPLITS, MODEL_PATH, RANDOM_STATE


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_test_data():
    """Load X_test, y_test from persisted pickle splits."""
    x_path = os.path.join(DATA_SPLITS, "X_test.pkl")
    y_path = os.path.join(DATA_SPLITS, "y_test.pkl")
    with open(x_path, "rb") as f:
        X_test = pickle.load(f)
    with open(y_path, "rb") as f:
        y_test = pickle.load(f)
    return X_test, y_test


def _get_preprocessor_and_clf(pipeline):
    """
    Extract the fitted ColumnTransformer and the raw classifier from the
    imblearn pipeline.  Handles two common step-name conventions.
    """
    # Step names used by build_cv_pipeline(): 'preprocessor', 'smote', 'classifier'
    # Fallback names: 'preprocessor', 'smote', 'clf'
    step_names = [name for name, _ in pipeline.steps]

    preproc_name = "preprocessor" if "preprocessor" in step_names else step_names[0]
    clf_name     = "classifier"   if "classifier"   in step_names else step_names[-1]

    preprocessor = pipeline.named_steps[preproc_name]
    clf          = pipeline.named_steps[clf_name]
    return preprocessor, clf


def _get_feature_names(preprocessor) -> list[str]:
    """Return cleaned output feature names from the fitted ColumnTransformer."""
    raw_names = list(preprocessor.get_feature_names_out())
    # Strip sklearn branch prefix (e.g. 'numeric__Age' → 'Age')
    return [n.split("__")[-1] for n in raw_names]


def _build_explainer(clf, X_transformed: np.ndarray):
    """
    Instantiate the most appropriate SHAP explainer for the given classifier.
    Tree-based models (XGB, LGBM, RF) use TreeExplainer; others fall back to
    LinearExplainer or KernelExplainer.
    """
    clf_type = type(clf).__name__

    if clf_type in ("XGBClassifier", "LGBMClassifier", "RandomForestClassifier",
                    "GradientBoostingClassifier", "ExtraTreesClassifier"):
        explainer = shap.TreeExplainer(clf)
    elif clf_type in ("LogisticRegression", "LinearSVC", "SGDClassifier"):
        explainer = shap.LinearExplainer(clf, X_transformed)
    else:
        # Generic fallback — slow but universally compatible
        background = shap.kmeans(X_transformed, k=50)
        explainer  = shap.KernelExplainer(clf.predict_proba, background)

    return explainer


def _find_case_indices(y_test: pd.Series, y_pred: np.ndarray,
                       seed: int = RANDOM_STATE):
    """
    Return (tp_idx, fn_idx, fp_idx) — one representative sample each.
    Falls back to -1 if a category has no samples.
    """
    rng = np.random.default_rng(seed)

    tp_mask = (y_test.values == 1) & (y_pred == 1)
    fn_mask = (y_test.values == 1) & (y_pred == 0)
    fp_mask = (y_test.values == 0) & (y_pred == 1)

    def _pick(mask):
        indices = np.where(mask)[0]
        return int(rng.choice(indices)) if len(indices) > 0 else -1

    return _pick(tp_mask), _pick(fn_mask), _pick(fp_mask)


def _save_waterfall(shap_explanation, idx: int, title: str,
                    filename: str, feature_names: list[str]) -> None:
    """Save a SHAP waterfall plot for a single sample."""
    if idx == -1:
        print(f"    [WARN]  No sample found for {title} — skipping waterfall plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 7))
    shap.plots.waterfall(shap_explanation[idx], max_display=15, show=False)
    plt.title(title, fontsize=13, pad=10)
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {out_path}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_shap(best_model, tau_star: float) -> None:
    """
    Compute SHAP values for the best model and save all 5 explanation plots.

    Outputs (saved to FIGURES_DIR at 300 DPI):
        shap_beeswarm.png        — Global beeswarm
        shap_bar.png             — Mean |SHAP| bar chart
        shap_waterfall_tp.png    — Local waterfall — True Positive
        shap_waterfall_fn.png    — Local waterfall — False Negative
        shap_waterfall_fp.png    — Local waterfall — False Positive

    Parameters
    ----------
    best_model : fitted imblearn Pipeline  (preprocessor → smote → clf)
    tau_star   : float — decision threshold from find_best_threshold()
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)
    print("\n  [SHAP] Starting analysis ...")

    # ------------------------------------------------------------------
    # 1. Load test data & extract pipeline components
    # ------------------------------------------------------------------
    X_test, y_test = _load_test_data()
    print(f"  [SHAP] Loaded test set: X={X_test.shape}, y={y_test.shape}")

    preprocessor, clf = _get_preprocessor_and_clf(best_model)
    feature_names     = _get_feature_names(preprocessor)

    # Transform test features (no SMOTE on test data)
    X_test_t = preprocessor.transform(X_test)
    print(f"  [SHAP] Transformed test features: {X_test_t.shape}")

    # ------------------------------------------------------------------
    # 2. Generate predictions at tau_star (for waterfall case selection)
    # ------------------------------------------------------------------
    proba  = best_model.predict_proba(X_test)[:, 1]
    y_pred = (proba >= tau_star).astype(int)

    # ------------------------------------------------------------------
    # 3. Build SHAP explainer & compute values
    # ------------------------------------------------------------------
    print("  [SHAP] Building explainer ...")
    explainer = _build_explainer(clf, X_test_t)

    print("  [SHAP] Computing SHAP values (may take ~30 s for tree models) ...")
    shap_values = explainer(X_test_t)

    # Attach human-readable feature names to the Explanation object
    shap_values.feature_names = feature_names

    # For multi-output explainers (e.g. XGBoost multi-class), keep class-1 slice
    if shap_values.values.ndim == 3:
        shap_values_cls1 = shap.Explanation(
            values         = shap_values.values[:, :, 1],
            base_values    = shap_values.base_values[:, 1]
                             if shap_values.base_values.ndim == 2
                             else shap_values.base_values,
            data           = shap_values.data,
            feature_names  = feature_names,
        )
    else:
        shap_values_cls1 = shap_values

    # ------------------------------------------------------------------
    # 4. GLOBAL — Beeswarm plot
    # ------------------------------------------------------------------
    print("  [SHAP] Generating global beeswarm plot ...")
    fig, ax = plt.subplots(figsize=(12, 9))
    shap.plots.beeswarm(shap_values_cls1, max_display=20, show=False)
    plt.title("SHAP Beeswarm — Global Feature Impact on Asthma Prediction",
              fontsize=13, pad=10)
    plt.tight_layout()
    beeswarm_path = os.path.join(FIGURES_DIR, "shap_beeswarm.png")
    plt.savefig(beeswarm_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {beeswarm_path}")

    # ------------------------------------------------------------------
    # 5. GLOBAL — Bar plot (mean |SHAP|)
    # ------------------------------------------------------------------
    print("  [SHAP] Generating global bar plot ...")
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.bar(shap_values_cls1, max_display=20, show=False)
    plt.title("SHAP Feature Importance — Mean |SHAP Value| (Top 20)",
              fontsize=13, pad=10)
    plt.tight_layout()
    bar_path = os.path.join(FIGURES_DIR, "shap_bar.png")
    plt.savefig(bar_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {bar_path}")

    # ------------------------------------------------------------------
    # 6. LOCAL — Waterfall plots (TP, FN, FP)
    # ------------------------------------------------------------------
    tp_idx, fn_idx, fp_idx = _find_case_indices(y_test, y_pred)
    print(f"  [SHAP] Case indices -> TP={tp_idx}, FN={fn_idx}, FP={fp_idx}")

    _save_waterfall(shap_values_cls1, tp_idx,
                    "SHAP Waterfall — True Positive (Correctly Detected Asthma)",
                    "shap_waterfall_tp.png", feature_names)

    _save_waterfall(shap_values_cls1, fn_idx,
                    "SHAP Waterfall — False Negative (Missed Asthma Case)",
                    "shap_waterfall_fn.png", feature_names)

    _save_waterfall(shap_values_cls1, fp_idx,
                    "SHAP Waterfall — False Positive (Healthy Flagged as Asthma)",
                    "shap_waterfall_fp.png", feature_names)

    # ------------------------------------------------------------------
    # 7. Print top-5 features by mean |SHAP|
    # ------------------------------------------------------------------
    mean_abs_shap = np.abs(shap_values_cls1.values).mean(axis=0)
    top5_idx      = np.argsort(mean_abs_shap)[::-1][:5]

    print("\n  [SHAP] Top-5 features by mean |SHAP value|:")
    print(f"  {'Rank':<6} {'Feature':<35} {'Mean |SHAP|':>12}")
    print("  " + "-" * 55)
    for rank, i in enumerate(top5_idx, 1):
        fname = feature_names[i] if i < len(feature_names) else f"feature_{i}"
        print(f"  {rank:<6} {fname:<35} {mean_abs_shap[i]:>12.5f}")

    print("\n  [SHAP] Analysis complete. All figures saved to:", FIGURES_DIR)
