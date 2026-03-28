import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load dataset
DATA_PATH = os.path.join(os.path.dirname(__file__), 'yt_dataset_v5.csv')
df = pd.read_csv(DATA_PATH)

# Basic preprocessing (customize as needed)
df = df.dropna()
# Drop non-numeric columns
y_leakage = ['log_dislikes', 'dislikes'] if 'dislikes' in df.columns else ['log_dislikes']
non_numeric_cols = df.drop(y_leakage, axis=1).select_dtypes(include=['object']).columns.tolist()
X = df.drop(y_leakage + non_numeric_cols, axis=1)
y = df['log_dislikes']

# Split data: 75% train, 10% val, 15% test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.25, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.6, random_state=42)

# XGBoost regressor
xgb_reg = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

# Hyperparameter grid
param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
}

# Randomized search with cross-validation
random_search = RandomizedSearchCV(
    xgb_reg, param_distributions=param_dist, n_iter=20, scoring='neg_mean_squared_error',
    cv=5, verbose=1, n_jobs=-1, random_state=42
)
random_search.fit(X_train, y_train)

best_model = random_search.best_estimator_

# Cross-validation scores
cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')

# Validation and test evaluation
val_pred = best_model.predict(X_val)
test_pred = best_model.predict(X_test)
val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
val_r2 = r2_score(y_val, val_pred)
test_r2 = r2_score(y_test, test_pred)

# Visualizations
def plot_feature_importance(model, X):
    importance = model.feature_importances_
    features = X.columns
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importance, y=features)
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

def plot_predictions(y_true, y_pred, name):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'Actual vs Predicted ({name})')
    plt.tight_layout()
    plt.savefig(f'actual_vs_pred_{name}.png')
    plt.close()

def plot_error_distribution(y_true, y_pred, name):
    errors = y_true - y_pred
    plt.figure(figsize=(8, 6))
    sns.histplot(errors, bins=30, kde=True)
    plt.title(f'Error Distribution ({name})')
    plt.tight_layout()
    plt.savefig(f'error_dist_{name}.png')
    plt.close()

plot_feature_importance(best_model, X_train)
plot_predictions(y_val, val_pred, 'val')
plot_predictions(y_test, test_pred, 'test')
plot_error_distribution(y_val, val_pred, 'val')
plot_error_distribution(y_test, test_pred, 'test')

# Report
def write_report():
    with open('XGBoost_Model_Report.md', 'w') as f:
        f.write('# XGBoost Regression Model Report\n\n')
        f.write('## Data Split\n')
        f.write('Train: 75%\nValidation: 10%\nTest: 15%\n\n')
        f.write('## Hyperparameter Optimization\n')
        f.write(f'Best Parameters: {random_search.best_params_}\n\n')
        f.write('## Cross-Validation\n')
        f.write(f'CV RMSE (mean): {-np.mean(cv_scores):.4f}\nCV RMSE (std): {np.std(cv_scores):.4f}\n\n')
        f.write('## Validation Results\n')
        f.write(f'RMSE: {val_rmse:.4f}\nR2: {val_r2:.4f}\n\n')
        f.write('## Test Results\n')
        f.write(f'RMSE: {test_rmse:.4f}\nR2: {test_r2:.4f}\n\n')
        f.write('## Visualizations\n')
        f.write('- Feature Importance: feature_importance.png\n')
        f.write('- Actual vs Predicted (Validation): actual_vs_pred_val.png\n')
        f.write('- Actual vs Predicted (Test): actual_vs_pred_test.png\n')
        f.write('- Error Distribution (Validation): error_dist_val.png\n')
        f.write('- Error Distribution (Test): error_dist_test.png\n')

write_report()

print('Training, evaluation, and report complete.')
