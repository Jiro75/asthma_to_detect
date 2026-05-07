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
from imblearn.over_sampling import SMOTE
from config import DATA_RAW, FIGURES_DIR, SMOTE_RATIO_DEFAULT, TARGET_COL

def main():
    print("="*60)
    print("  Preprocessing Visualization Script")
    print("="*60)

    # 1. Ensure figures directory exists
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # 2. Load and Split Data
    print("Loading and splitting data...")
    df = load_and_validate(DATA_RAW)
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(df)

    # 3. Fit Preprocessor
    print("Applying ColumnTransformer (Yeo-Johnson, Scaling, OHE)...")
    preprocessor = build_preprocessor()
    X_train_transformed = preprocessor.fit_transform(X_train)
    feature_names = get_feature_names(preprocessor)
    
    # Create DataFrame for easy plotting
    df_transformed = pd.DataFrame(X_train_transformed, columns=feature_names)
    df_transformed[TARGET_COL] = y_train.values

    # ---------------------------------------------------------
    # PLOT 1: Feature Distribution (Before vs After Yeo-Johnson)
    # ---------------------------------------------------------
    print("Generating Figure 1: Yeo-Johnson Transformation Effect...")
    plt.figure(figsize=(12, 5))
    
    # Before
    plt.subplot(1, 2, 1)
    sns.histplot(X_train['IgE_Level'], kde=True, color='red', bins=30)
    plt.title("Original 'IgE_Level' (Highly Skewed)")
    plt.xlabel("Raw IgE Level")
    
    # After
    plt.subplot(1, 2, 2)
    # Feature name is prefixed with branch name
    sns.histplot(df_transformed['numeric__IgE_Level'], kde=True, color='green', bins=30)
    plt.title("Transformed 'IgE_Level' (Gaussian & Scaled)")
    plt.xlabel("Standardized Yeo-Johnson Value")
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "preprocessing_transformation.png"), dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # PLOT 2: Class Imbalance (Before vs After SMOTE)
    # ---------------------------------------------------------
    print("Generating Figure 2: SMOTE Oversampling Effect...")
    
    # Apply SMOTE standalone for visualization
    smote = SMOTE(sampling_strategy=SMOTE_RATIO_DEFAULT, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train_transformed, y_train)
    
    counts_before = y_train.value_counts().sort_index()
    counts_after = y_resampled.value_counts().sort_index()

    labels = ['Negative (0)', 'Positive (1)']
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(x - width/2, counts_before.values, width, label='Before SMOTE', color='royalblue')
    ax.bar(x + width/2, counts_after.values, width, label=f'After SMOTE (Ratio {SMOTE_RATIO_DEFAULT})', color='darkorange')
    
    ax.set_ylabel('Number of Patients')
    ax.set_title('Training Set Class Balance (Before vs After SMOTE)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "preprocessing_smote_balance.png"), dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # PLOT 3: Top Feature Correlations Heatmap
    # ---------------------------------------------------------
    print("Generating Figure 3: Top Feature Correlations Heatmap...")
    
    # Get top 10 positive and top 10 negative correlations
    correlations = df_transformed.corr()[TARGET_COL].drop(TARGET_COL).dropna().sort_values(ascending=False)
    top_pos = correlations.head(10).index.tolist()
    top_neg = correlations.tail(10).index.tolist()
    top_features = top_pos + top_neg + [TARGET_COL]
    
    # Compute correlation matrix just for top features
    corr_matrix = df_transformed[top_features].corr()
    
    plt.figure(figsize=(14, 12))
    # Draw heatmap
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, 
                vmin=-1, vmax=1, square=True, linewidths=.5)
    plt.title("Correlation Heatmap: Top 20 Features vs Diagnosis", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "preprocessing_correlation_heatmap.png"), dpi=300)
    plt.close()

    print("\n" + "="*60)
    print("Success! Visualizations saved to the 'figures/' directory:")
    print("  1. figures/preprocessing_transformation.png")
    print("  2. figures/preprocessing_smote_balance.png")
    print("  3. figures/preprocessing_correlation_heatmap.png")
    print("="*60)

if __name__ == "__main__":
    main()
