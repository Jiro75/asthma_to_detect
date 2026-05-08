import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure we can import from src and config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_and_validate
from src.splitter import split_dataset
from src.preprocessing import build_preprocessor, get_feature_names
from src.cv_pipeline import preprocess_for_early_stopping
from config import DATA_RAW, FIGURES_DIR, SMOTE_RATIO_DEFAULT, TARGET_COL

def generate_transformation_plot(X_train, df_transformed):
    print("Generating Figure 1: Yeo-Johnson Transformation Effect...")
    plt.figure(figsize=(12, 5))
    
    # Before
    plt.subplot(1, 2, 1)
    sns.histplot(X_train['IgE_Level'], kde=True, color='red', bins=30)
    plt.title("Original 'IgE_Level' (Highly Skewed)")
    plt.xlabel("Raw IgE Level")
    
    # After
    plt.subplot(1, 2, 2)
    sns.histplot(df_transformed['numeric__IgE_Level'], kde=True, color='green', bins=30)
    plt.title("Transformed 'IgE_Level' (Gaussian & Scaled)")
    plt.xlabel("Standardized Yeo-Johnson Value")
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "preprocessing_transformation.png"), dpi=300)
    plt.close()

def generate_class_distribution_plot(X_train, y_train, X_val):
    print("Generating Figure 2: Class Imbalance (Before vs After SMOTE)...")
    
    counts_before = y_train.value_counts().sort_index()
    
    X_train_res, y_train_res, _ = preprocess_for_early_stopping(
        X_train, y_train, X_val, smote_ratio=SMOTE_RATIO_DEFAULT
    )
    counts_after = y_train_res.value_counts().sort_index()
    
    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot Before
    labels_before = [f"Negative\n({counts_before.get(0, 0)})", f"Positive\n({counts_before.get(1, 0)})"]
    sns.barplot(x=labels_before, y=counts_before.values, ax=axes[0], color="#D32F2F")  # specific color instead of palette
    axes[0].set_title("Training Fold (Before SMOTE)", pad=15)
    axes[0].set_ylabel("Number of Samples")
    axes[0].set_ylim(0, max(counts_after.values) * 1.1)
    
    for i, v in enumerate(counts_before.values):
        axes[0].text(i, v + max(counts_after.values)*0.02, str(v), ha='center', va='bottom', fontweight='bold')
        
    # Plot After
    labels_after = [f"Negative\n({counts_after.get(0, 0)})", f"Positive\n({counts_after.get(1, 0)})"]
    sns.barplot(x=labels_after, y=counts_after.values, ax=axes[1], color="#1976D2")  # specific color instead of palette
    axes[1].set_title(f"Training Fold (After SMOTE, ratio={SMOTE_RATIO_DEFAULT})", pad=15)
    axes[1].set_ylabel("")
    axes[1].set_ylim(0, max(counts_after.values) * 1.1)
    
    for i, v in enumerate(counts_after.values):
        axes[1].text(i, v + max(counts_after.values)*0.02, str(v), ha='center', va='bottom', fontweight='bold')
        
    plt.suptitle("Class Imbalance Resolution via SMOTE (Training Data)", fontsize=16, fontweight='bold', y=1.05)
    plt.tight_layout()
    
    plt.savefig(os.path.join(FIGURES_DIR, "class_distribution_before_after_smote.png"), dpi=300, bbox_inches='tight')
    plt.close()

def generate_correlation_bar_chart(y_train, df_transformed):
    print("Generating Figure 3: Correlation Bar Chart...")
    
    correlations = df_transformed.apply(lambda col: col.corr(y_train))
    correlations = correlations.dropna()
    
    # Select Top Features by Absolute Correlation
    top_features = correlations.abs().sort_values(ascending=False).head(25).index
    top_corrs = correlations[top_features]
    
    # Sort them so highest absolute is at top of barh
    top_corrs = top_corrs.reindex(top_corrs.abs().sort_values(ascending=True).index)
    
    clean_names = [name.split('__')[-1] for name in top_corrs.index]
    
    # Reset seaborn theme to default for this specific chart
    sns.reset_defaults()
    fig, ax = plt.subplots(figsize=(12, 10))
    
    colors = ['#2ca02c' if val >= 0 else '#d62728' for val in top_corrs.values]
    bars = ax.barh(clean_names, top_corrs.values, color=colors, height=0.6)
    
    ax.axvline(x=0, color='black', linewidth=1)
    
    max_corr = top_corrs.abs().max()
    threshold = 0.1 if max_corr < 0.3 else 0.2
    
    ax.axvline(x=-threshold, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.axvline(x=threshold, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    
    for bar in bars:
        width = bar.get_width()
        x_offset = 0.01 if width >= 0 else -0.01
        ha = 'left' if width >= 0 else 'right'
        ax.text(width + x_offset, bar.get_y() + bar.get_height() / 2, f'{width:.3f}', ha=ha, va='center', fontsize=9)
    
    ax.set_xlabel('Pearson correlation with Diagnosis', fontsize=11)
    ax.set_title('Feature → Diagnosis correlation\nGreen = positive, Red = negative | Dashed lines = signal thresholds', fontsize=13, pad=15)
    
    padding = 0.05
    ax.set_xlim(min(0, top_corrs.min()) - padding, max(0, top_corrs.max()) + padding)
    plt.tight_layout()
    
    plt.savefig(os.path.join(FIGURES_DIR, "correlation_bar_chart.png"), dpi=300)
    plt.close()

def generate_correlation_heatmap(df_transformed):
    print("Generating Figure 4: Top Feature Correlations Heatmap...")
    sns.reset_defaults()
    
    correlations = df_transformed.corr()[TARGET_COL].drop(TARGET_COL).dropna().sort_values(ascending=False)
    top_pos = correlations.head(10).index.tolist()
    top_neg = correlations.tail(10).index.tolist()
    top_features = top_pos + top_neg + [TARGET_COL]
    
    corr_matrix = df_transformed[top_features].corr()
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, vmin=-1, vmax=1, square=True, linewidths=.5)
    plt.title("Correlation Heatmap: Top 20 Features vs Diagnosis", fontsize=16)
    plt.tight_layout()
    
    plt.savefig(os.path.join(FIGURES_DIR, "preprocessing_correlation_heatmap.png"), dpi=300)
    plt.close()

def main():
    print("="*60)
    print("  Unified Preprocessing Visualization Script")
    print("="*60)

    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("1. Loading and splitting data...")
    df = load_and_validate(DATA_RAW)
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(df)

    print("2. Applying ColumnTransformer (Yeo-Johnson, Scaling, OHE)...")
    preprocessor = build_preprocessor()
    X_train_transformed = preprocessor.fit_transform(X_train, y_train)
    feature_names = get_feature_names(preprocessor)
    
    df_transformed = pd.DataFrame(X_train_transformed, columns=feature_names)
    df_transformed[TARGET_COL] = y_train.values

    # Generate all plots
    generate_transformation_plot(X_train, df_transformed)
    generate_class_distribution_plot(X_train, y_train, X_val)
    generate_correlation_bar_chart(y_train, df_transformed)
    generate_correlation_heatmap(df_transformed)

    print("\n" + "="*60)
    print("Success! All visualizations saved to the 'figures/' directory:")
    print("  - preprocessing_transformation.png")
    print("  - class_distribution_before_after_smote.png")
    print("  - correlation_bar_chart.png")
    print("  - preprocessing_correlation_heatmap.png")
    print("="*60)

if __name__ == "__main__":
    main()
