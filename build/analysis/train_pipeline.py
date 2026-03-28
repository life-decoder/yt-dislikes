"""
Train a reproducible scikit-learn pipeline for predicting dislikes.

Pipeline steps:
 - load engineered CSV (detects `log_dislikes` or `dislikes` target)
 - column discovery: numeric + small-cardinality categoricals (includes `genre_id`)
 - preprocessing: median imputation + StandardScaler for numerics; OneHotEncoder for categoricals
 - SelectKBest (f_regression) feature selection
 - Ridge regression model

Outputs:
 - prints train/test RMSE and CV RMSE
 - saves fitted pipeline to `analysis/pipeline_model.pkl`

Run from workspace root (PowerShell):
& .venv/Scripts/python.exe analysis/train_pipeline.py
"""
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / 'yt_dataset_feature_engineered.csv'
MODEL_OUT = ROOT / 'pipeline_model.pkl'

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Engineered dataset not found: {DATA_PATH}\nRun analysis/eda_and_feature_engineering.py first to create it.")

print(f"Loading data from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH, low_memory=False)
print(f"Rows: {len(df)}, Columns: {len(df.columns)}")

# detect target: prefer log target if present
if 'log_dislikes' in df.columns:
    target_col = 'log_dislikes'
elif 'dislikes' in df.columns:
    target_col = 'dislikes'
else:
    # fallback: search for column name containing 'dislike'
    target_col = None
    for c in df.columns:
        if 'dislike' in c.lower():
            target_col = c
            break
    if target_col is None:
        raise RuntimeError('Could not locate a dislike target column')

print(f"Using target: {target_col}")

# Prepare X, y
y = df[target_col].copy()
X = df.drop(columns=[target_col])

def detect_and_remove_target_leakage(X_df, y_series, corr_threshold=0.995):
    """Detect columns that are equal/near-equal to the target or have extremely high correlation.
    Returns cleaned X and list of dropped columns."""
    to_drop = []
    for col in X_df.columns:
        s = X_df[col]
        # exact equality (works for ints/strings)
        try:
            if s.equals(y_series):
                to_drop.append(col)
                continue
        except Exception:
            pass

        # numeric correlation checks
        if pd.api.types.is_numeric_dtype(s) and pd.api.types.is_numeric_dtype(y_series):
            s_f = s.fillna(0).astype(float)
            y_f = y_series.fillna(0).astype(float)
            # exact numeric equality
            if (s_f == y_f).all():
                to_drop.append(col)
                continue
            # log equality checks
            try:
                if np.allclose(np.log1p(s_f), y_f, rtol=1e-6, atol=1e-6):
                    to_drop.append(col)
                    continue
                if np.allclose(np.log1p(y_f), s_f, rtol=1e-6, atol=1e-6):
                    to_drop.append(col)
                    continue
            except Exception:
                pass
            # Pearson correlation
            try:
                corr = np.corrcoef(s_f, y_f)[0,1]
                if not np.isnan(corr) and abs(corr) >= corr_threshold:
                    to_drop.append(col)
                    continue
            except Exception:
                pass

    if to_drop:
        print('Detected potential leakage columns relative to target; dropping:', to_drop)
        X_clean = X_df.drop(columns=to_drop)
    else:
        print('No obvious leakage detected.')
        X_clean = X_df
    return X_clean, to_drop

# detect and remove leakage
X, dropped = detect_and_remove_target_leakage(X, y)
if dropped:
    (ROOT / 'leakage_dropped_pipeline.txt').write_text('\n'.join(dropped))

# Drop obvious non-feature textual columns to keep pipeline clean (title, description, etc.)
text_cols = [c for c in X.columns if c.lower() in ('title', 'description', 'video_title', 'tags')]
if text_cols:
    print('Dropping text columns from features:', text_cols)
    X = X.drop(columns=text_cols)

# numeric and categorical discovery
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
# treat small-cardinality object/integers as categorical
categorical_cols = []
for c in X.columns:
    if c in numeric_cols:
        continue
    nunq = X[c].nunique(dropna=False)
    if nunq <= 50:
        categorical_cols.append(c)

# ensure genre_id is present in categorical list if it exists
if 'genre_id' in X.columns and 'genre_id' not in categorical_cols:
    categorical_cols.append('genre_id')

print(f"Numeric features: {len(numeric_cols)}; Categorical features: {len(categorical_cols)}")

# Column transformer
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
])

def _to_str(X):
    # top-level function so FunctionTransformer is picklable
    return X.astype(str)


categorical_pipeline = Pipeline([
    # convert categorical inputs (including numeric codes like genre_id) to string
    ('tostr', FunctionTransformer(_to_str, validate=False)),
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    # sklearn >=1.2 uses 'sparse_output' instead of deprecated 'sparse'
    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
])

preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, numeric_cols),
    ('cat', categorical_pipeline, categorical_cols),
], remainder='drop')

# Full pipeline: preprocess -> feature selection -> model
pipeline = Pipeline([
    ('preproc', preprocessor),
    ('select', SelectKBest(score_func=f_regression, k=20)),
    ('model', Ridge(alpha=1.0))
])

print('Splitting data (80/20)')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print('Fitting pipeline...')
pipeline.fit(X_train, y_train)

print('Predicting on test set...')
y_pred = pipeline.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f'Test RMSE ({target_col}): {rmse:.4f}')
print(f'Test R2   ({target_col}): {r2:.4f}')

print('5-fold CV (neg_mean_squared_error) on training set...')
# some sklearn versions don't support 'neg_root_mean_squared_error'; use neg_mean_squared_error and convert
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
cv_rmse = np.sqrt(-cv_scores)
print(f'CV RMSEs: {cv_rmse.round(4)}; mean={cv_rmse.mean():.4f}, std={cv_rmse.std():.4f}')

print(f'Saving fitted pipeline to {MODEL_OUT}')
joblib.dump(pipeline, MODEL_OUT)

print('Done')
