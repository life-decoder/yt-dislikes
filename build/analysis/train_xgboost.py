"""
Train an XGBoost regressor using a scikit-learn Pipeline.

Requirements: xgboost installed in the active environment.

Run:
& .venv/Scripts/python.exe analysis/train_xgboost.py
"""
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
except Exception as e:
    raise ImportError('xgboost is required. Install with: pip install xgboost') from e


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / 'yt_dataset_feature_engineered.csv'
OUT_PATH = ROOT / 'xgb_pipeline.pkl'

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Engineered dataset not found: {DATA_PATH}. Run the EDA script first.")

print(f"Loading data from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH, low_memory=False)

# target preference
if 'log_dislikes' in df.columns:
    target_col = 'log_dislikes'
elif 'dislikes' in df.columns:
    target_col = 'dislikes'
else:
    target_col = None
    for c in df.columns:
        if 'dislike' in c.lower():
            target_col = c
            break
    if target_col is None:
        raise RuntimeError('Could not find a dislikes target column')

print(f'Using target: {target_col}')

y = df[target_col]
X = df.drop(columns=[target_col])

# drop large text columns if present
for text_col in ['title', 'description', 'video_title', 'tags']:
    if text_col in X.columns:
        X = X.drop(columns=[text_col])

numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = []
for c in X.columns:
    if c in numeric_cols:
        continue
    if X[c].nunique(dropna=False) <= 50:
        categorical_cols.append(c)

if 'genre_id' in X.columns and 'genre_id' not in categorical_cols:
    categorical_cols.append('genre_id')

print(f'Numeric features: {len(numeric_cols)}; Categorical features: {len(categorical_cols)}')


def _to_str(X):
    return X.astype(str)


def detect_and_remove_target_leakage(X_df, y_series, corr_threshold=0.995):
    to_drop = []
    for col in X_df.columns:
        s = X_df[col]
        try:
            if s.equals(y_series):
                to_drop.append(col)
                continue
        except Exception:
            pass
        if pd.api.types.is_numeric_dtype(s) and pd.api.types.is_numeric_dtype(y_series):
            s_f = s.fillna(0).astype(float)
            y_f = y_series.fillna(0).astype(float)
            if (s_f == y_f).all():
                to_drop.append(col)
                continue
            try:
                if np.allclose(np.log1p(s_f), y_f, rtol=1e-6, atol=1e-6):
                    to_drop.append(col)
                    continue
                if np.allclose(np.log1p(y_f), s_f, rtol=1e-6, atol=1e-6):
                    to_drop.append(col)
                    continue
            except Exception:
                pass
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

# detect leakage and remove
X, dropped = detect_and_remove_target_leakage(X, y)
if dropped:
    (ROOT / 'leakage_dropped_xgb.txt').write_text('\n'.join(dropped))


numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
])

categorical_pipeline = Pipeline([
    ('tostr', FunctionTransformer(_to_str, validate=False)),
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
])

preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, numeric_cols),
    ('cat', categorical_pipeline, categorical_cols),
], remainder='drop')

pipeline = Pipeline([
    ('preproc', preprocessor),
    ('model', XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        tree_method='auto',
        n_jobs=-1,
        verbosity=0
    ))
])

print('Splitting data (train/test 80/20 and internal validation set for early stopping)')
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.125, random_state=42)
# this gives roughly 70% train, 10% val, 20% test

print('Fitting preprocessing on training data...')
preprocessor.fit(X_train)

X_train_trans = preprocessor.transform(X_train)
X_val_trans = preprocessor.transform(X_val)
X_test_trans = preprocessor.transform(X_test)

print('Training XGBoost (attempt early stopping compatible with installed xgboost)')
import xgboost as xgb_pkg

# create XGBRegressor instance from pipeline
xgb = pipeline.named_steps['model']
eval_set = [(X_val_trans, y_val)]

trained = False
# Try callback-based API (newer xgboost)
try:
    es_callback = xgb_pkg.callback.EarlyStopping(rounds=50, save_best=True)
    xgb.fit(X_train_trans, y_train, eval_set=eval_set, callbacks=[es_callback], verbose=False)
    trained = True
except Exception:
    # try older API with early_stopping_rounds
    try:
        xgb.fit(X_train_trans, y_train, eval_set=eval_set, early_stopping_rounds=50, verbose=False)
        trained = True
    except Exception:
        # fallback: train without early stopping
        xgb.fit(X_train_trans, y_train)
        trained = True

if not trained:
    raise RuntimeError('Could not train xgboost with available API')

# attach trained model
pipeline.named_steps['model'] = xgb
print('Predicting on test set...')
y_pred = xgb.predict(X_test_trans)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f'Test RMSE ({target_col}): {rmse:.4f}')
print(f'Test R2   ({target_col}): {r2:.4f}')

print(f'Saving pipeline (preprocessor + model) to {OUT_PATH}')
# save preprocessor + trained xgb as a tuple to ensure pipeline is reproducible
joblib.dump({'preprocessor': preprocessor, 'model': xgb}, OUT_PATH)

print('Done')
