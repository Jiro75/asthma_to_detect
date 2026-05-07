import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_and_validate
from src.splitter import split_dataset
from src.preprocessing import build_preprocessor, get_feature_names
from config import DATA_RAW, FIGURES_DIR, TARGET_COL

def main():
    print("="*60)
    print("  Correlation Bar Chart Generator")
    print("="*60)

    os.makedirs(FIGURES_DIR, exist_ok=True)

    # 1. Load and Transform Data
    print("Loading and transforming data...")
    df = load_and_validate(DATA_RAW)
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(df)

    preprocessor = build_preprocessor()
    X_train_transformed = preprocessor.fit_transform(X_train)
    feature_names = get_feature_names(preprocessor)
    
    # 2. Compute Correlations
    print("Computing correlations...")
    df_transformed = pd.DataFrame(X_train_transformed, columns=feature_names)
    
    # Calculate correlation with y_train
    correlations = df_transformed.apply(lambda col: col.corr(y_train))
    correlations = correlations.dropna()
    
    # 3. Select Top Features by Absolute Correlation
    # We select the top 25 features to keep the chart readable
    top_features = correlations.abs().sort_values(ascending=False).head(25).index
    top_corrs = correlations[top_features]
    
    # Sort them so the highest absolute correlation is at the top of the bar chart
    # To have highest at the top in barh, we need highest at the end of the series
    top_corrs = top_corrs.reindex(top_corrs.abs().sort_values(ascending=True).index)
    
    # Clean up feature names for display (e.g., 'numeric__IgE_Level' -> 'IgE_Level')
    clean_names = [name.split('__')[-1] for name in top_corrs.index]
    
    # 4. Create the Bar Chart
    print("Generating the chart...")
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Colors: Green for positive, Red for negative
    # Use hex colors similar to the image
    colors = ['#2ca02c' if val >= 0 else '#d62728' for val in top_corrs.values]
    
    bars = ax.barh(clean_names, top_corrs.values, color=colors, height=0.6)
    
    # Add vertical dashed lines at -0.1, 0, 0.1 (or -0.2, 0.2 depending on max values)
    ax.axvline(x=0, color='black', linewidth=1)
    
    # Determine appropriate thresholds based on max correlation
    max_corr = top_corrs.abs().max()
    threshold = 0.1 if max_corr < 0.3 else 0.2
    
    ax.axvline(x=-threshold, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.axvline(x=threshold, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    
    # Add data labels next to bars
    for bar in bars:
        width = bar.get_width()
        # Offset the text slightly from the end of the bar
        x_offset = 0.01 if width >= 0 else -0.01
        ha = 'left' if width >= 0 else 'right'
        ax.text(width + x_offset, 
                bar.get_y() + bar.get_height() / 2, 
                f'{width:.3f}', 
                ha=ha, va='center', fontsize=9)
    
    # Formatting
    ax.set_xlabel('Pearson correlation with Diagnosis', fontsize=11)
    ax.set_title('Feature → Diagnosis correlation\nGreen = positive, Red = negative | Dashed lines = signal thresholds', 
                 fontsize=13, pad=15)
    
    # Set dynamic x-limits to ensure labels fit
    padding = 0.05
    ax.set_xlim(min(0, top_corrs.min()) - padding, max(0, top_corrs.max()) + padding)
    
    plt.tight_layout()
    
    output_path = os.path.join(FIGURES_DIR, "correlation_bar_chart.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Success! Saved chart to {output_path}")

if __name__ == "__main__":
    main()
