"""
EDA and feature engineering script for yt_dataset_v4_merged.csv
Saves plots to analysis/plots and writes analysis/EDA_REPORT.md
"""
import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
except Exception as e:
    print("Missing required packages. Please install pandas, numpy, matplotlib, seaborn, scikit-learn.")
    raise

# Paths
script_dir = Path(__file__).resolve().parent
workspace_root = script_dir.parent
data_path = workspace_root / 'feature_engineering_v2' / 'yt_dataset_v4_merged.csv'
plots_dir = script_dir / 'plots'
plots_dir.mkdir(parents=True, exist_ok=True)
report_path = script_dir / 'EDA_REPORT.md'
output_csv = script_dir / 'yt_dataset_feature_engineered.csv'

print(f"Loading data from: {data_path}")
if not data_path.exists():
    raise FileNotFoundError(f"Data file not found: {data_path}")

# Load
df = pd.read_csv(data_path, low_memory=False)
print(f"Rows: {len(df)}, Columns: {len(df.columns)}")

# Identify target column (dislike(s))
lower_cols = [c.lower() for c in df.columns]
candidate = None
for i, c in enumerate(lower_cols):
    if 'dislike' in c:
        candidate = df.columns[i]
        break
if candidate is None:
    raise RuntimeError('Could not find a target column containing "dislike" in the header names')

target_col = candidate
print(f"Detected target column: {target_col}")

# Numeric features
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if target_col not in numeric_cols:
    # try to coerce
    try:
        df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
        if df[target_col].dtype.kind in 'iuf':
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    except Exception:
        pass

numeric_cols = [c for c in numeric_cols if c != target_col]
print(f"Numeric feature count (excluding target): {len(numeric_cols)}")

# Basic missing value report
missing = df.isna().mean().sort_values(ascending=False)
missing_report = missing[missing > 0]

# Correlation matrix
corr_cols = numeric_cols + [target_col]
corr_df = df[corr_cols].corr()

# Save correlation heatmap
plt.figure(figsize=(12,10))
sns.heatmap(corr_df, cmap='vlag', center=0, robust=True)
plt.title('Correlation matrix (numeric features + target)')
plt.tight_layout()
heatmap_path = plots_dir / 'correlation_heatmap.png'
plt.savefig(heatmap_path, dpi=150)
plt.close()
print(f"Saved correlation heatmap to {heatmap_path}")

# Top features correlated with target
target_corr = corr_df[target_col].drop(index=target_col).abs().sort_values(ascending=False)
top_corr = target_corr.head(20)

# PCA on numeric features (fillna)
X = df[numeric_cols].copy()
X_filled = X.fillna(X.median())
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_filled)

pca = PCA()
pca.fit(X_scaled)
explained = pca.explained_variance_ratio_
cum_explained = np.cumsum(explained)

# Plot explained variance
plt.figure(figsize=(8,5))
plt.plot(np.arange(1, len(explained)+1), explained, marker='o', label='per-component')
plt.plot(np.arange(1, len(explained)+1), cum_explained, marker='s', label='cumulative')
plt.xlabel('Principal component')
plt.ylabel('Explained variance ratio')
plt.title('PCA explained variance')
plt.legend()
plt.grid(True)
plt.tight_layout()
pca_path = plots_dir / 'pca_explained_variance.png'
plt.savefig(pca_path, dpi=150)
plt.close()
print(f"Saved PCA explained variance plot to {pca_path}")

# PCA scatter of first 2 components (colored by log target)
if X_scaled.shape[1] >= 2:
    comps = pca.transform(X_scaled)[:, :2]
    log_target = np.log1p(df[target_col].fillna(0))
    plt.figure(figsize=(8,6))
    sc = plt.scatter(comps[:,0], comps[:,1], c=log_target, cmap='viridis', s=8, alpha=0.7)
    plt.colorbar(sc, label='log1p(target)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('PCA (PC1 vs PC2) colored by log(target)')
    plt.tight_layout()
    pca_scatter_path = plots_dir / 'pca_scatter_pc1_pc2.png'
    plt.savefig(pca_scatter_path, dpi=150)
    plt.close()
    print(f"Saved PCA scatter to {pca_scatter_path}")

# Histograms and log1p histograms for numeric features
hist_sample_limit = 100
for col in numeric_cols:
    series = df[col]
    if series.dropna().shape[0] == 0:
        continue
    fig, axes = plt.subplots(1,2, figsize=(10,3))
    sns.histplot(series.dropna(), bins=50, ax=axes[0], kde=False)
    axes[0].set_title(col)
    # log transform
    try:
        transformed = np.log1p(series.fillna(0))
        sns.histplot(transformed, bins=50, ax=axes[1], kde=False)
        axes[1].set_title(f'log1p({col})')
    except Exception:
        axes[1].text(0.5,0.5,'log transform failed', horizontalalignment='center')
    plt.tight_layout()
    safe_col = col.replace('/', '_').replace(' ', '_')[:120]
    hist_path = plots_dir / f'hist_{safe_col}.png'
    fig.savefig(hist_path, dpi=120)
    plt.close(fig)

# Create some simple engineered features if available
engineered = df.copy()
# log target
engineered['log_' + target_col] = np.log1p(engineered[target_col].fillna(0))

# conditional ratios
if 'views' in engineered.columns and 'likes' in engineered.columns:
    engineered['likes_per_view'] = engineered['likes'] / engineered['views'].replace(0, np.nan)
if 'views' in engineered.columns and 'comment_count' in engineered.columns:
    engineered['comments_per_view'] = engineered['comment_count'] / engineered['views'].replace(0, np.nan)

# title length if title exists
for title_col in ['title', 'video_title']:
    if title_col in engineered.columns:
        engineered['title_length'] = engineered[title_col].fillna('').astype(str).str.len()
        break

engineered.to_csv(output_csv, index=False)
print(f"Saved engineered dataset to {output_csv}")

# Write Markdown report
with open(report_path, 'w', encoding='utf8') as f:
    f.write('# EDA and Feature Engineering Report\n\n')
    f.write(f'Input file: `{data_path}`\n\n')
    f.write(f'Rows: {len(df)}, Columns: {len(df.columns)}\n\n')
    f.write(f'Detected target column: `{target_col}`\n\n')
    f.write('## Missing values (columns with >0% missing)\n\n')
    if missing_report.empty:
        f.write('No missing values detected in any column.\n\n')
    else:
        f.write('| column | percent_missing |\n')
        f.write('|---|---:|\n')
        for col, val in missing_report.items():
            f.write(f'| `{col}` | {val:.3f} |\n')
        f.write('\n')

    f.write('## Top features by absolute Pearson correlation with target\n\n')
    f.write('| feature | abs_corr_with_target |\n')
    f.write('|---|---:|\n')
    for col, val in top_corr.items():
        f.write(f'| `{col}` | {val:.4f} |\n')
    f.write('\n')

    f.write('Correlation heatmap: `analysis/plots/correlation_heatmap.png`\n\n')
    f.write('PCA explained variance plot: `analysis/plots/pca_explained_variance.png`\n\n')
    if X_scaled.shape[1] >= 2:
        f.write('PCA scatter (PC1 vs PC2): `analysis/plots/pca_scatter_pc1_pc2.png`\n\n')

    f.write('## PCA summary\n\n')
    # number components to reach thresholds
    thresholds = [0.75, 0.9, 0.95]
    for t in thresholds:
        comps_needed = int(np.searchsorted(cum_explained, t) + 1)
        f.write(f'- {t*100:.0f}% variance reached with {comps_needed} components\n')
    f.write('\n')

    f.write('## Feature distributions\n\n')
    f.write('Individual histograms and log-transformed histograms saved to `analysis/plots/` (files prefixed with `hist_`).\n\n')

    f.write('## Engineered dataset\n\n')
    f.write(f'Wrote engineered dataset to `{output_csv}` containing added column `log_{target_col}` and conditional ratio features when input columns were present.\n\n')

    f.write('## Recommendations\n\n')
    f.write('- Use `log_{}` as the regression target for models that expect gaussian-like residuals.\n'.format(target_col))
    f.write('- Start modeling with the top correlated features listed above.\n')
    f.write('- Consider dimensionality reduction (PCA) or feature selection if you have many numeric features or multicollinearity.\n')
    f.write('\n')

print(f"Wrote report to {report_path}")
print('Done')
