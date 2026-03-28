"""XGBoost model selection script
- Loads `yt_dataset_v5.csv` from repo root
- Splits into train/val/test = 75/10/15
- Trains XGBRegressor on `log_dislikes` target using early stopping on validation set
- Produces and saves plots into `analysis/plots/`
- Writes a markdown report `analysis/XGBoost_selection_report.md`
"""
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from xgboost.callback import EarlyStopping as EarlyStoppingCallback
import joblib

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / 'yt_dataset_v5.csv'
PLOTS_DIR = Path(__file__).resolve().parents[0] / 'plots'
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = Path(__file__).resolve().parents[0] / 'XGBoost_selection_report.md'
MODEL_PATH = Path(__file__).resolve().parents[0] / 'xgb_model_selection_joblib.pkl'

RANDOM_STATE = 42


def load_and_prepare(path):
    df = pd.read_csv(path)
    # Drop obvious ID / text columns
    drop_cols = ['video_id', 'channel_id', 'published_at']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    # Drop rows with missing values
    df = df.dropna()
    return df


def split_data(df, target='log_dislikes'):
    X = df.drop(columns=[target, 'dislikes']) if 'dislikes' in df.columns else df.drop(columns=[target])
    y = df[target].values
    # First split off test set (15%)
    X_rem, X_test, y_rem, y_test = train_test_split(X, y, test_size=0.15, random_state=RANDOM_STATE)
    # Then split remaining into train (75%) and val (10%). Since remaining is 85% of original, val fraction = 10/85
    val_fraction = 10.0 / 85.0
    X_train, X_val, y_train, y_val = train_test_split(X_rem, y_rem, test_size=val_fraction, random_state=RANDOM_STATE)
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_xgb(X_train, y_train, X_val, y_val):
    # Use the core xgboost.train API with DMatrix and callbacks for reliable early stopping
    dtrain = xgb.DMatrix(X_train.values, label=y_train, feature_names=list(X_train.columns))
    dval = xgb.DMatrix(X_val.values, label=y_val, feature_names=list(X_val.columns))
    params = {
        'objective': 'reg:squarederror',
        'eta': 0.05,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'verbosity': 1,
        'seed': RANDOM_STATE,
        'eval_metric': 'rmse',
    }
    callbacks = [EarlyStoppingCallback(rounds=50, save_best=True)]
    evals = [(dtrain, 'train'), (dval, 'validation')]
    evals_result = {}
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=evals,
        evals_result=evals_result,
        callbacks=callbacks,
        verbose_eval=50,
    )
    return booster, evals_result


def evaluate(model, X, y, label='validation'):
    # model is an xgboost.Booster trained with xgb.train
    dmat = xgb.DMatrix(X.values, feature_names=list(X.columns))
    ntree_limit = getattr(model, 'best_ntree_limit', None)
    if ntree_limit:
        preds = model.predict(dmat, ntree_limit=ntree_limit)
    else:
        preds = model.predict(dmat)
        # mean_squared_error in some sklearn versions doesn't accept 'squared' kwarg; compute MSE then sqrt
        mse = mean_squared_error(y, preds)
        rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y, preds)
    r2 = r2_score(y, preds)
    return {'rmse': rmse, 'mae': mae, 'r2': r2, 'preds': preds}


def plot_pred_vs_true(y_true, y_pred, outpath):
    plt.figure(figsize=(7,7))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.3, s=20)
    minv = min(y_true.min(), y_pred.min())
    maxv = max(y_true.max(), y_pred.max())
    plt.plot([minv, maxv], [minv, maxv], color='red', linestyle='--')
    plt.xlabel('True log_dislikes')
    plt.ylabel('Predicted log_dislikes')
    plt.title('Predicted vs True (validation)')
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def plot_residuals(y_true, y_pred, outpath):
    res = y_true - y_pred
    plt.figure(figsize=(8,4))
    sns.histplot(res, bins=50, kde=True)
    plt.xlabel('Residual (true - pred)')
    plt.title('Residual distribution (validation)')
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def plot_feature_importance(booster, feature_names, outpath, top_n=25):
    score = booster.get_score(importance_type='weight')
    items = []
    for k, v in score.items():
        if k.startswith('f'):
            idx = int(k[1:])
            name = feature_names[idx] if idx < len(feature_names) else k
        else:
            name = k
        items.append((name, v))
    if not items:
        df = pd.DataFrame({'feature': feature_names, 'importance': np.zeros(len(feature_names))})
    else:
        df = pd.DataFrame(items, columns=['feature', 'importance'])
        df = df.sort_values('importance', ascending=False).head(top_n)
    plt.figure(figsize=(8, max(4, 0.25 * len(df))))
    sns.barplot(data=df, x='importance', y='feature')
    plt.title('XGBoost feature importance (weight)')
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def plot_learning_curve(model_or_evals, outpath):
    try:
        if isinstance(model_or_evals, dict):
            evals = model_or_evals
        else:
            # try attribute attached to booster
            evals = getattr(model_or_evals, '_evals_result', None) or {}
        train_rmse = evals.get('train', {}).get('rmse', [])
        val_rmse = evals.get('validation', {}).get('rmse', [])
        if not train_rmse or not val_rmse:
            raise ValueError('No evals history available')
        rounds = range(1, len(train_rmse) + 1)
        plt.figure(figsize=(8,4))
        plt.plot(rounds, train_rmse, label='train rmse')
        plt.plot(rounds, val_rmse, label='val rmse')
        plt.xlabel('Boosting round')
        plt.ylabel('RMSE')
        plt.legend()
        plt.title('Training vs Validation RMSE')
        plt.tight_layout()
        plt.savefig(outpath)
        plt.close()
    except Exception as e:
        print('Could not plot learning curve:', e)


def write_report(report_path, df, feature_names, train_shape, val_shape, test_shape, val_metrics, best_iteration, model_path):
    lines = []
    lines.append('# XGBoost model selection report')
    lines.append('')
    lines.append('This report documents a model selection run using XGBoost to predict `log_dislikes` from the available features in `yt_dataset_v5.csv`.')
    lines.append('')
    lines.append('## Data')
    lines.append(f'- Original dataset shape (after dropna): {df.shape}')
    lines.append(f'- Features used ({len(feature_names)}): {feature_names}')
    lines.append(f'- Train / Val / Test shapes: {train_shape} / {val_shape} / {test_shape}')
    lines.append('')
    lines.append('## Model & training')
    lines.append('- Model: XGBoost XGBRegressor')
    lines.append(f'- Best iteration (early stopping): {best_iteration}')
    lines.append('')
    lines.append('## Validation performance (used for model selection)')
    lines.append(f'- RMSE: {val_metrics["rmse"]:.4f}')
    lines.append(f'- MAE: {val_metrics["mae"]:.4f}')
    lines.append(f'- R^2: {val_metrics["r2"]:.4f}')
    lines.append('')
    lines.append('## Visualizations')
    lines.append('Predicted vs True (validation):')
    lines.append(f'![](plots/xgb_selection_pred_vs_true.png)')
    lines.append('')
    lines.append('Residual distribution (validation):')
    lines.append(f'![](plots/xgb_selection_residuals.png)')
    lines.append('')
    lines.append('Feature importance:')
    lines.append(f'![](plots/xgb_selection_feature_importance.png)')
    lines.append('')
    lines.append('Learning curve (RMSE per boosting round):')
    lines.append(f'![](plots/xgb_selection_learning_curve.png)')
    lines.append('')
    lines.append('## Notes and insights')
    lines.append('- The model was evaluated on the validation set (10% of data); the test set (15%) is held out and not used for selection as requested.')
    lines.append('- Feature importance indicates which numeric features the tree-based model relies on most; consider domain-informed feature engineering for further improvement.')
    lines.append('- Residual distribution and scatterplot show how predictions deviate; consider log-transforming other skewed inputs or adding interaction features.')
    lines.append('')
    lines.append('## Artifacts')
    lines.append(f'- Saved model: {model_path}')
    lines.append(f'- Plots directory: {PLOTS_DIR}')

    report_path.write_text('\n'.join(lines))


if __name__ == '__main__':
    print('Loading data...')
    df = load_and_prepare(DATA_PATH)
    print('Splitting...')
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, target='log_dislikes')
    print(f'Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}')

    feature_names = list(X_train.columns)

    print('Training XGBoost...')
    model, evals_result = train_xgb(X_train, y_train, X_val, y_val)
    print('Training complete. Best iteration:', getattr(model, 'best_iteration', None))

    print('Evaluating on validation set...')
    val_res = evaluate(model, X_val, y_val, label='val')

    # Save model (Booster)
    try:
        # model is a Booster
        model.save_model(str(MODEL_PATH))
        print('Saved booster model to', MODEL_PATH)
    except Exception:
        joblib.dump(model, MODEL_PATH)
        print('Saved model (joblib fallback) to', MODEL_PATH)

    # plots
    plot_pred_vs_true(y_val, val_res['preds'], PLOTS_DIR / 'xgb_selection_pred_vs_true.png')
    plot_residuals(y_val, val_res['preds'], PLOTS_DIR / 'xgb_selection_residuals.png')
    plot_feature_importance(model, feature_names, PLOTS_DIR / 'xgb_selection_feature_importance.png')
    plot_learning_curve(evals_result, PLOTS_DIR / 'xgb_selection_learning_curve.png')

    # write report
    write_report(REPORT_PATH, df, feature_names, X_train.shape, X_val.shape, X_test.shape, val_res, getattr(model, 'best_iteration', None), MODEL_PATH)
    print('Report written to', REPORT_PATH)
