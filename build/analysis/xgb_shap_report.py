"""
Compute SHAP values for saved XGBoost model and produce a feature importance report.

Outputs:
 - analysis/plots/shap_summary_bar.png
 - analysis/plots/shap_summary_dot.png
 - analysis/shap_importance.csv
 - analysis/SHAP_REPORT.md

Run:
& .venv/Scripts/python.exe analysis/xgb_shap_report.py
"""
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / 'yt_dataset_feature_engineered.csv'
MODEL_PATH = ROOT / 'xgb_pipeline.pkl'
PLOTS_DIR = ROOT / 'plots'
PLOTS_DIR.mkdir(exist_ok=True)

# Provide a top-level _to_str function so preprocessor FunctionTransformer can be unpickled
def _to_str(X):
    return X.astype(str)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f'Model not found: {MODEL_PATH}. Train XGBoost first.')
if not DATA_PATH.exists():
    raise FileNotFoundError(f'Data not found: {DATA_PATH}.')

print('Loading model and preprocessor...')
obj = joblib.load(MODEL_PATH)
preprocessor = obj.get('preprocessor') if isinstance(obj, dict) else None
model = obj.get('model') if isinstance(obj, dict) else None
if model is None:
    raise RuntimeError('Saved model content not as expected (expected dict with model and preprocessor)')

print('Loading data...')
df = pd.read_csv(DATA_PATH, low_memory=False)
target = 'log_dislikes' if 'log_dislikes' in df.columns else ('dislikes' if 'dislikes' in df.columns else None)
if target is None:
    raise RuntimeError('Target column not found in data')

# If a leakage list exists, note it but don't drop yet; we'll align to the preprocessor's expected inputs
leak_file = ROOT / 'leakage_dropped_xgb.txt'
leaked_cols = []
if leak_file.exists():
    leaked_cols = [l.strip() for l in leak_file.read_text().splitlines() if l.strip()]
    if leaked_cols:
        print('Note: leakage-dropped columns recorded from training:', leaked_cols)

X = df.drop(columns=[target])
y = df[target]

# sample for SHAP to speed up computation
sample_n = min(2000, len(X))
X_sample = X.sample(n=sample_n, random_state=42)

print('Transforming features with preprocessor...')
if preprocessor is None:
    raise RuntimeError('Preprocessor not found in saved model')

# Ensure the DataFrame has the same input columns the preprocessor was fitted with
expected_cols = None
if hasattr(preprocessor, 'feature_names_in_'):
    try:
        expected_cols = list(preprocessor.feature_names_in_)
    except Exception:
        expected_cols = None
if expected_cols is None:
    # try to extract columns from the ColumnTransformer definition
    try:
        expected_cols = []
        for name, trans, cols in preprocessor.transformers_:
            if cols in ('drop', 'passthrough'):
                continue
            if isinstance(cols, (list, tuple, pd.Index, np.ndarray)):
                expected_cols.extend(list(cols))
            else:
                expected_cols = None
                break
    except Exception:
        expected_cols = None

if expected_cols is not None:
    # Reindex to expected columns; fill missing with zeros
    X_sample = X_sample.reindex(columns=expected_cols, fill_value=0)
else:
    # If we couldn't determine expected columns, add any leakage-dropped columns back as zeros
    for c in leaked_cols:
        if c not in X_sample.columns:
            X_sample[c] = 0

X_sample_trans = preprocessor.transform(X_sample)

# try to get feature names
try:
    feature_names = preprocessor.get_feature_names_out(X_sample.columns)
except Exception:
    # fallback: try to compose names from numeric + categorical transformers
    feature_names = None
    try:
        num_names = preprocessor.named_transformers_['num'].named_steps.get('scaler', None)
    except Exception:
        pass

if feature_names is None:
    # last resort: generate generic names
    feature_names = [f'f{i}' for i in range(X_sample_trans.shape[1])]

print(f'Feature matrix shape after transform: {X_sample_trans.shape}')

print('Importing shap...')
try:
    import shap
except Exception as e:
    raise RuntimeError('shap is required. Install with: pip install shap') from e

print('Creating SHAP explainer...')
try:
    explainer = shap.Explainer(model)
    shap_vals = explainer(X_sample_trans)
except Exception:
    # fallback for older shap/xgboost combinations
    try:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_sample_trans)
    except Exception as e:
        raise RuntimeError('Failed to create SHAP explainer') from e

print('Calculating mean absolute SHAP per feature...')
if hasattr(shap_vals, 'values'):
    arr = np.array(shap_vals.values)
else:
    arr = np.array(shap_vals)

mean_abs_shap = np.mean(np.abs(arr), axis=0)
df_shap = pd.DataFrame({'feature': feature_names, 'mean_abs_shap': mean_abs_shap})
df_shap = df_shap.sort_values('mean_abs_shap', ascending=False)
shap_csv = ROOT / 'shap_importance.csv'
df_shap.to_csv(shap_csv, index=False)
print('Wrote SHAP importance CSV to', shap_csv)

# summary bar plot
plt.figure(figsize=(8,6))
topN = min(30, df_shap.shape[0])
plt.barh(df_shap['feature'].head(topN)[::-1], df_shap['mean_abs_shap'].head(topN)[::-1])
plt.xlabel('mean |SHAP value|')
plt.title('SHAP feature importance (mean |value|)')
plt.tight_layout()
bar_path = PLOTS_DIR / 'shap_summary_bar.png'
plt.savefig(bar_path, dpi=150)
plt.close()
print('Saved SHAP bar plot to', bar_path)

# dot summary (be careful with older shap API)
try:
    plt.figure(figsize=(8,6))
    shap.summary_plot(shap_vals, X_sample_trans, feature_names=feature_names, show=False)
    dot_path = PLOTS_DIR / 'shap_summary_dot.png'
    plt.tight_layout()
    plt.savefig(dot_path, dpi=150)
    plt.close()
    print('Saved SHAP dot summary to', dot_path)
except Exception:
    print('Could not generate SHAP dot summary plot (incompatible shap version)')

# write a markdown report
report = ROOT / 'SHAP_REPORT.md'
with report.open('w', encoding='utf8') as f:
    f.write('# SHAP Feature Importance Report\n\n')
    f.write(f'Data sample size for SHAP: {sample_n}\n\n')
    f.write('Top features by mean absolute SHAP:\n\n')
    f.write('| feature | mean_abs_shap |\n')
    f.write('|---|---:|\n')
    for _, row in df_shap.head(50).iterrows():
        f.write(f'| `{row.feature}` | {row.mean_abs_shap:.6f} |\n')
    f.write('\n')
    f.write(f'Bar plot: `analysis/plots/shap_summary_bar.png`\n\n')
    f.write(f'Dot plot: `analysis/plots/shap_summary_dot.png` (if generated)\n')

print('Wrote SHAP report to', report)
print('Done')
