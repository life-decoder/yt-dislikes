import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

DATESET_PATH = 'final_model/yt_dataset_v5.csv'
# Load and prepare data
df = pd.read_csv(DATESET_PATH, header=0)

# Handle missing values
df['log_comment_count'].fillna(df['log_comment_count'].median(), inplace=True)
df['log_likes'].fillna(df['log_likes'].median(), inplace=True)

# Feature engineering
df['published_at'] = pd.to_datetime(df['published_at'], format='%d/%m/%Y %H:%M')
df['publish_hour'] = df['published_at'].dt.hour
df['publish_day'] = df['published_at'].dt.dayofweek
df['publish_month'] = df['published_at'].dt.month

# Define features (excluding dislikes as requested)
exclude_cols = ['video_id', 'channel_id', 'published_at', 'dislikes', 'log_dislikes']
feature_cols = [col for col in df.columns if col not in exclude_cols]
X = df[feature_cols]
y = df['log_dislikes']

# 75% train, 10% validation, 15% test
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1176, random_state=42)

# Enhanced XGBoost with comprehensive hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [3, 4, 5, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1, 0.15],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.2],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [1, 1.5, 2]
}

# 5-fold cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    random_state=42,
    tree_method='hist'  # Faster training
)

# Grid search with cross-validation
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    cv=kfold,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# Predictions on all sets
y_train_pred = best_model.predict(X_train)
y_val_pred = best_model.predict(X_val)
y_test_pred = best_model.predict(X_test)

# Evaluation metrics
def evaluate_model(y_true, y_pred, dataset_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"{dataset_name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
    return {'RMSE': rmse, 'MAE': mae, 'R2': r2}

train_metrics = evaluate_model(y_train, y_train_pred, "Train")
val_metrics = evaluate_model(y_val, y_val_pred, "Validation")
test_metrics = evaluate_model(y_test, y_test_pred, "Test")

plt.figure(figsize=(12, 8))
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

sns.barplot(data=feature_importance.head(15), x='importance', y='feature')
plt.title('Top 15 Feature Importances')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

datasets = [(y_train, y_train_pred, 'Train'), 
           (y_val, y_val_pred, 'Validation'), 
           (y_test, y_test_pred, 'Test')]

for i, (y_true, y_pred, title) in enumerate(datasets):
    axes[i].scatter(y_true, y_pred, alpha=0.5)
    axes[i].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    axes[i].set_xlabel('Actual Log Dislikes')
    axes[i].set_ylabel('Predicted Log Dislikes')
    axes[i].set_title(f'{title} Set\nR² = {r2_score(y_true, y_pred):.4f}')
    
plt.tight_layout()
plt.show()

residuals = y_test - y_test_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_test_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Log Dislikes')
plt.ylabel('Residuals')
plt.title('Residual Plot (Test Set)')
plt.show()

# Residual distribution
plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=50, alpha=0.7)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Residual Distribution (Test Set)')
plt.show()