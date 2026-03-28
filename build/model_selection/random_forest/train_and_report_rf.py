"""Train and validate a Random Forest regressor on yt_dataset_v5.csv.

Saves:
- plots to model_selection/random_forest/plots/
- trained model to model_selection/random_forest/rf_model.pkl
- markdown report to model_selection/random_forest/RANDOM_FOREST_REPORT.md

Usage: run this file with the project's Python environment.
"""
from __future__ import annotations
import os
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT.parent / "yt_dataset_v5.csv"
OUT_DIR = ROOT
PLOTS_DIR = OUT_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    # Target
    if 'log_dislikes' not in df.columns:
        raise KeyError('log_dislikes must be present in the dataset')
    y = df['log_dislikes'].astype(float)

    # Drop raw dislikes to prevent leakage
    X = df.drop(columns=['dislikes', 'log_dislikes'], errors='ignore')

    # Drop identifier columns that shouldn't be used as features
    for c in ['video_id', 'channel_id', 'published_at']:
        if c in X.columns:
            X = X.drop(columns=c)

    # Treat genre_id as categorical
    if 'genre_id' in X.columns:
        X['genre_id'] = X['genre_id'].astype('category')
        # one-hot encode
        X = pd.get_dummies(X, columns=['genre_id'], prefix='genre')

    # Ensure numeric dtype for all remaining columns
    X = X.apply(pd.to_numeric, errors='coerce')

    # Simple imputation: fill numeric NaNs with median
    X = X.fillna(X.median())

    return X, y


def split_data(X: pd.DataFrame, y: pd.Series) -> dict:
    # We want train:75%, val:10%, test:15%
    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )
    val_frac_of_rest = 10 / 85  # 10% of total over the 85% remaining
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest, y_rest, test_size=val_frac_of_rest, random_state=42
    )
    return {
        'train': (X_train, y_train),
        'val': (X_val, y_val),
        'test': (X_test, y_test),
    }


def train_rf(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestRegressor:
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    return rf


def evaluate(model, X, y) -> dict:
    preds = model.predict(X)
    # Some sklearn versions don't accept the `squared` kwarg; compute RMSE directly for compatibility
    rmse = mean_squared_error(y, preds)
    rmse = float(np.sqrt(rmse))
    mae = mean_absolute_error(y, preds)
    r2 = r2_score(y, preds)
    return {'preds': preds, 'rmse': rmse, 'mae': mae, 'r2': r2}


def plot_and_save_hist(y_train, y_val):
    plt.figure(figsize=(8, 4))
    sns.histplot(y_train, color='C0', label='train', kde=True, stat='density')
    sns.histplot(y_val, color='C1', label='val', kde=True, stat='density')
    plt.legend()
    plt.title('Target (log_dislikes) distribution: train vs val')
    f = PLOTS_DIR / 'target_distribution_train_val.png'
    plt.tight_layout()
    plt.savefig(f)
    plt.close()
    return f


def plot_pred_vs_actual(y_true, y_pred, split_name: str) -> Path:
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.4, s=10)
    mn = min(y_true.min(), y_pred.min())
    mx = max(y_true.max(), y_pred.max())
    plt.plot([mn, mx], [mn, mx], 'r--')
    plt.xlabel('Actual log_dislikes')
    plt.ylabel('Predicted log_dislikes')
    plt.title(f'Predicted vs Actual ({split_name})')
    f = PLOTS_DIR / f'pred_vs_actual_{split_name}.png'
    plt.tight_layout()
    plt.savefig(f)
    plt.close()
    return f


def plot_residuals(y_true, y_pred, split_name: str) -> Path:
    res = y_true - y_pred
    plt.figure(figsize=(6, 4))
    sns.histplot(res, kde=True)
    plt.title(f'Residuals distribution ({split_name})')
    f = PLOTS_DIR / f'residuals_{split_name}.png'
    plt.tight_layout()
    plt.savefig(f)
    plt.close()
    return f


def plot_feature_importance(model: RandomForestRegressor, feature_names) -> Path:
    imp = model.feature_importances_
    order = np.argsort(imp)[::-1]
    topk = min(30, len(feature_names))
    plt.figure(figsize=(8, max(4, topk * 0.25)))
    sns.barplot(x=imp[order][:topk], y=np.array(feature_names)[order][:topk])
    plt.title('Random Forest feature importances (top {})'.format(topk))
    plt.xlabel('Importance')
    f = PLOTS_DIR / 'feature_importances.png'
    plt.tight_layout()
    plt.savefig(f)
    plt.close()
    return f


def write_report(metrics_train, metrics_val, paths: dict, model_path: Path):
    rpt_path = OUT_DIR / 'RANDOM_FOREST_REPORT.md'
    with open(rpt_path, 'w', encoding='utf8') as fh:
        fh.write('# Random Forest Regression — Report\n\n')
        fh.write('Dataset: `model_selection/yt_dataset_v5.csv`\n\n')
        fh.write('Target: `log_dislikes` (log of dislikes). `dislikes` column was excluded from features.\n\n')

        fh.write('## Splits\n')
        fh.write('- Train: 75%\n')
        fh.write('- Validation: 10% (used for model selection / early evaluation)\n')
        fh.write('- Test: 15% (held out, not used here)\n\n')

        fh.write('## Validation results\n')
        fh.write('### Train metrics\n')
        fh.write(f'- RMSE: {metrics_train["rmse"]:.4f}\n')
        fh.write(f'- MAE: {metrics_train["mae"]:.4f}\n')
        fh.write(f'- R2: {metrics_train["r2"]:.4f}\n\n')

        fh.write('### Validation metrics\n')
        fh.write(f'- RMSE: {metrics_val["rmse"]:.4f}\n')
        fh.write(f'- MAE: {metrics_val["mae"]:.4f}\n')
        fh.write(f'- R2: {metrics_val["r2"]:.4f}\n\n')

        fh.write('## Plots\n')
        for name, p in paths.items():
            fh.write(f'- {name}: `{p.name}`\n')

        fh.write('\n')
        fh.write(f'Model saved to: `{model_path.name}`\n')

    return rpt_path


def main():
    print('Loading data...')
    df = load_data(DATA_PATH)
    print('Preparing features...')
    X, y = prepare_features(df)
    print(f'Feature matrix shape: {X.shape}')

    splits = split_data(X, y)
    X_train, y_train = splits['train']
    X_val, y_val = splits['val']
    X_test, y_test = splits['test']

    print('Training Random Forest...')
    model = train_rf(X_train, y_train)

    print('Evaluating on train and validation sets...')
    res_train = evaluate(model, X_train, y_train)
    res_val = evaluate(model, X_val, y_val)

    print('Creating plots...')
    plots = {}
    plots['target_distribution'] = plot_and_save_hist(y_train, y_val)
    plots['pred_vs_actual_train'] = plot_pred_vs_actual(y_train, res_train['preds'], 'train')
    plots['pred_vs_actual_val'] = plot_pred_vs_actual(y_val, res_val['preds'], 'val')
    plots['residuals_train'] = plot_residuals(y_train, res_train['preds'], 'train')
    plots['residuals_val'] = plot_residuals(y_val, res_val['preds'], 'val')
    plots['feature_importances'] = plot_feature_importance(model, X.columns.tolist())

    model_path = OUT_DIR / 'rf_model.pkl'
    with open(model_path, 'wb') as fh:
        pickle.dump({'model': model, 'features': X.columns.tolist()}, fh)

    rpt = write_report(res_train, res_val, plots, model_path)
    print(f'Report written to: {rpt}')
    print('Plots saved to:', PLOTS_DIR)


if __name__ == '__main__':
    main()
