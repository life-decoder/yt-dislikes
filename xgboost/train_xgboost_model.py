"""
XGBoost Model Training for YouTube Dislikes Prediction
=======================================================
Train-Validation-Test Split: 75-10-15
Target: log_dislikes (as recommended by feature engineering analysis)
Features: Tier 2 Tree-Based features
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import xgboost as xgb
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("XGBoost Model Training for YouTube Dislikes Prediction")
print("="*80)
print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("1. Loading dataset...")
df = pd.read_csv('../yt_dataset_v4.csv')
print(f"   Dataset shape: {df.shape}")
print(f"   Total samples: {len(df):,}")

# ============================================================================
# 2. FEATURE SELECTION
# ============================================================================
print("\n2. Selecting features...")

# Tier 2 Tree-Based features (recommended for XGBoost)
FEATURES = [
    'view_count',
    'likes',
    'comment_count',
    'avg_compound',
    'avg_pos',
    'avg_neg',
    'comment_sample_size',
    'no_comments',
    'view_like_ratio',
    'age'
]

# Target variable (log-transformed as recommended)
TARGET = 'log_dislikes'

print(f"   Selected features ({len(FEATURES)}):")
for i, feat in enumerate(FEATURES, 1):
    print(f"      {i:2d}. {feat}")
print(f"   Target variable: {TARGET}")

# ============================================================================
# 3. HANDLE MISSING VALUES
# ============================================================================
print("\n3. Handling missing values...")
print(f"   Missing values before handling:")
missing_before = df[FEATURES + [TARGET]].isnull().sum()
for col in missing_before[missing_before > 0].index:
    print(f"      {col}: {missing_before[col]} ({missing_before[col]/len(df)*100:.2f}%)")

# Handle missing values - fill with 0 for log features (represents log(1))
df_clean = df.copy()
for col in FEATURES + [TARGET]:
    if df_clean[col].isnull().any():
        if 'log' in col:
            df_clean[col].fillna(0, inplace=True)  # log(1) = 0
        else:
            df_clean[col].fillna(df_clean[col].median(), inplace=True)

print(f"   Missing values after handling: {df_clean[FEATURES + [TARGET]].isnull().sum().sum()}")

# ============================================================================
# 4. DATA SPLITTING (75-10-15)
# ============================================================================
print("\n4. Splitting data (Train: 75%, Val: 10%, Test: 15%)...")

X = df_clean[FEATURES]
y = df_clean[TARGET]

# First split: 75% train, 25% temp (for val+test)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_SEED
)

# Second split: split temp into 10% val and 15% test (40% and 60% of temp)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.6, random_state=RANDOM_SEED
)

print(f"   Training set:   {len(X_train):,} samples ({len(X_train)/len(df)*100:.1f}%)")
print(f"   Validation set: {len(X_val):,} samples ({len(X_val)/len(df)*100:.1f}%)")
print(f"   Test set:       {len(X_test):,} samples ({len(X_test)/len(df)*100:.1f}%)")
print(f"   Total:          {len(df):,} samples (100.0%)")

# ============================================================================
# 5. TRAIN XGBOOST MODEL
# ============================================================================
print("\n5. Training XGBoost model...")
print("   Hyperparameters:")
print("      - n_estimators: 200")
print("      - max_depth: 6")
print("      - learning_rate: 0.1")
print("      - subsample: 0.8")
print("      - colsample_bytree: 0.8")

# Create XGBoost model
model = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_SEED,
    objective='reg:squarederror',
    n_jobs=-1
)

# Train with validation set
eval_set = [(X_train, y_train), (X_val, y_val)]
model.fit(
    X_train, y_train,
    eval_set=eval_set,
    verbose=False
)

print(f"   ✓ Training complete!")
print(f"   Total iterations: {model.n_estimators}")

# ============================================================================
# 6. PREDICTIONS (Train and Validation only - NO TEST SET YET)
# ============================================================================
print("\n6. Making predictions on Train and Validation sets...")

y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)

# Convert predictions back to raw scale for interpretability
y_train_raw = np.expm1(y_train)
y_train_pred_raw = np.expm1(y_train_pred)
y_val_raw = np.expm1(y_val)
y_val_pred_raw = np.expm1(y_val_pred)

print("   ✓ Predictions complete!")

# ============================================================================
# 7. EVALUATE PERFORMANCE
# ============================================================================
print("\n7. Model Performance Evaluation")
print("="*80)

def calculate_metrics(y_true, y_pred, y_true_raw, y_pred_raw, set_name):
    """Calculate and display metrics for both log and raw scales"""
    print(f"\n{set_name} Set Performance:")
    print("-" * 40)
    
    # Log scale metrics
    rmse_log = np.sqrt(mean_squared_error(y_true, y_pred))
    mae_log = mean_absolute_error(y_true, y_pred)
    r2_log = r2_score(y_true, y_pred)
    
    print("  Log Scale Metrics:")
    print(f"    RMSE:  {rmse_log:.4f}")
    print(f"    MAE:   {mae_log:.4f}")
    print(f"    R²:    {r2_log:.4f}")
    
    # Raw scale metrics
    rmse_raw = np.sqrt(mean_squared_error(y_true_raw, y_pred_raw))
    mae_raw = mean_absolute_error(y_true_raw, y_pred_raw)
    r2_raw = r2_score(y_true_raw, y_pred_raw)
    mape_raw = mean_absolute_percentage_error(y_true_raw, y_pred_raw) * 100
    
    print("  Raw Scale Metrics:")
    print(f"    RMSE:  {rmse_raw:,.2f} dislikes")
    print(f"    MAE:   {mae_raw:,.2f} dislikes")
    print(f"    R²:    {r2_raw:.4f}")
    print(f"    MAPE:  {mape_raw:.2f}%")
    
    return {
        'rmse_log': rmse_log, 'mae_log': mae_log, 'r2_log': r2_log,
        'rmse_raw': rmse_raw, 'mae_raw': mae_raw, 'r2_raw': r2_raw, 'mape_raw': mape_raw
    }

# Calculate metrics
train_metrics = calculate_metrics(y_train, y_train_pred, y_train_raw, y_train_pred_raw, "Training")
val_metrics = calculate_metrics(y_val, y_val_pred, y_val_raw, y_val_pred_raw, "Validation")

# Check for overfitting
print("\n" + "="*80)
print("Overfitting Analysis:")
print("-" * 40)
r2_diff = train_metrics['r2_log'] - val_metrics['r2_log']
print(f"  R² difference (Train - Val): {r2_diff:.4f}")
if r2_diff < 0.05:
    print("  ✓ Model generalizes well (minimal overfitting)")
elif r2_diff < 0.10:
    print("  ⚠ Slight overfitting detected")
else:
    print("  ⚠⚠ Significant overfitting detected")

print("\n" + "="*80)

# ============================================================================
# 8. FEATURE IMPORTANCE
# ============================================================================
print("\n8. Feature Importance Analysis...")

feature_importance = pd.DataFrame({
    'feature': FEATURES,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n  Feature Importance Rankings:")
for idx, row in feature_importance.iterrows():
    print(f"    {row['feature']:25s}: {row['importance']:.4f}")

# ============================================================================
# 9. VISUALIZATIONS
# ============================================================================
print("\n9. Creating visualizations...")

# Create figure with subplots
fig = plt.figure(figsize=(20, 12))

# Plot 1: Training History (Learning Curves)
ax1 = plt.subplot(2, 3, 1)
results = model.evals_result()
epochs = len(results['validation_0']['rmse'])
x_axis = range(0, epochs)
ax1.plot(x_axis, results['validation_0']['rmse'], label='Train', linewidth=2)
ax1.plot(x_axis, results['validation_1']['rmse'], label='Validation', linewidth=2)
ax1.set_xlabel('Boosting Round', fontsize=11, fontweight='bold')
ax1.set_ylabel('RMSE (Log Scale)', fontsize=11, fontweight='bold')
ax1.set_title('Training History - Learning Curves', fontsize=13, fontweight='bold', pad=15)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Plot 2: Feature Importance
ax2 = plt.subplot(2, 3, 2)
feature_importance_sorted = feature_importance.head(10)
colors_cmap = plt.get_cmap('viridis')
colors = colors_cmap(np.linspace(0.3, 0.9, len(feature_importance_sorted)))
bars = ax2.barh(range(len(feature_importance_sorted)), 
                feature_importance_sorted['importance'], 
                color=colors)
ax2.set_yticks(range(len(feature_importance_sorted)))
ax2.set_yticklabels(feature_importance_sorted['feature'])
ax2.set_xlabel('Importance Score', fontsize=11, fontweight='bold')
ax2.set_title('Top 10 Feature Importance', fontsize=13, fontweight='bold', pad=15)
ax2.grid(True, alpha=0.3, axis='x')
ax2.invert_yaxis()

# Add value labels on bars
for i, bar in enumerate(bars):
    width = bar.get_width()
    ax2.text(width, bar.get_y() + bar.get_height()/2, 
             f'{width:.4f}', 
             ha='left', va='center', fontsize=9, fontweight='bold')

# Plot 3: Actual vs Predicted (Training) - Log Scale
ax3 = plt.subplot(2, 3, 3)
scatter1 = ax3.scatter(y_train, y_train_pred, alpha=0.3, s=10, c='blue', label='Training')
ax3.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 
         'r--', lw=2, label='Perfect Prediction')
ax3.set_xlabel('Actual log(dislikes)', fontsize=11, fontweight='bold')
ax3.set_ylabel('Predicted log(dislikes)', fontsize=11, fontweight='bold')
ax3.set_title(f'Actual vs Predicted - Training Set\nR² = {train_metrics["r2_log"]:.4f}', 
              fontsize=13, fontweight='bold', pad=15)
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

# Plot 4: Actual vs Predicted (Validation) - Log Scale
ax4 = plt.subplot(2, 3, 4)
scatter2 = ax4.scatter(y_val, y_val_pred, alpha=0.3, s=10, c='green', label='Validation')
ax4.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 
         'r--', lw=2, label='Perfect Prediction')
ax4.set_xlabel('Actual log(dislikes)', fontsize=11, fontweight='bold')
ax4.set_ylabel('Predicted log(dislikes)', fontsize=11, fontweight='bold')
ax4.set_title(f'Actual vs Predicted - Validation Set\nR² = {val_metrics["r2_log"]:.4f}', 
              fontsize=13, fontweight='bold', pad=15)
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)

# Plot 5: Residuals Distribution (Validation)
ax5 = plt.subplot(2, 3, 5)
residuals = y_val - y_val_pred
ax5.hist(residuals, bins=50, color='purple', alpha=0.7, edgecolor='black')
ax5.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
ax5.set_xlabel('Residuals (log scale)', fontsize=11, fontweight='bold')
ax5.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax5.set_title(f'Residual Distribution - Validation Set\nMean = {residuals.mean():.4f}, Std = {residuals.std():.4f}', 
              fontsize=13, fontweight='bold', pad=15)
ax5.legend(fontsize=10)
ax5.grid(True, alpha=0.3, axis='y')

# Plot 6: Residuals vs Predicted (Validation)
ax6 = plt.subplot(2, 3, 6)
ax6.scatter(y_val_pred, residuals, alpha=0.3, s=10, c='orange')
ax6.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
ax6.set_xlabel('Predicted log(dislikes)', fontsize=11, fontweight='bold')
ax6.set_ylabel('Residuals', fontsize=11, fontweight='bold')
ax6.set_title('Residual Plot - Validation Set\n(Check for Homoscedasticity)', 
              fontsize=13, fontweight='bold', pad=15)
ax6.legend(fontsize=10)
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('xgboost_performance_analysis.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: xgboost_performance_analysis.png")

# ============================================================================
# Additional Plot: Raw Scale Predictions
# ============================================================================
fig2, axes = plt.subplots(1, 2, figsize=(16, 6))

# Raw Scale - Training
axes[0].scatter(y_train_raw, y_train_pred_raw, alpha=0.3, s=10, c='blue')
axes[0].plot([y_train_raw.min(), y_train_raw.max()], 
             [y_train_raw.min(), y_train_raw.max()], 
             'r--', lw=2, label='Perfect Prediction')
axes[0].set_xlabel('Actual Dislikes', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Predicted Dislikes', fontsize=12, fontweight='bold')
axes[0].set_title(f'Raw Scale Predictions - Training Set\nR² = {train_metrics["r2_raw"]:.4f}, MAPE = {train_metrics["mape_raw"]:.2f}%', 
                  fontsize=13, fontweight='bold', pad=15)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)
axes[0].set_xscale('log')
axes[0].set_yscale('log')

# Raw Scale - Validation
axes[1].scatter(y_val_raw, y_val_pred_raw, alpha=0.3, s=10, c='green')
axes[1].plot([y_val_raw.min(), y_val_raw.max()], 
             [y_val_raw.min(), y_val_raw.max()], 
             'r--', lw=2, label='Perfect Prediction')
axes[1].set_xlabel('Actual Dislikes', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Predicted Dislikes', fontsize=12, fontweight='bold')
axes[1].set_title(f'Raw Scale Predictions - Validation Set\nR² = {val_metrics["r2_raw"]:.4f}, MAPE = {val_metrics["mape_raw"]:.2f}%', 
                  fontsize=13, fontweight='bold', pad=15)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)
axes[1].set_xscale('log')
axes[1].set_yscale('log')

plt.tight_layout()
plt.savefig('xgboost_raw_scale_predictions.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: xgboost_raw_scale_predictions.png")

# ============================================================================
# Additional Plot: Error Analysis
# ============================================================================
fig3, axes = plt.subplots(2, 2, figsize=(16, 12))

# Error distribution by prediction magnitude (Training)
axes[0, 0].scatter(y_train_pred, np.abs(y_train - y_train_pred), alpha=0.3, s=10, c='blue')
axes[0, 0].set_xlabel('Predicted log(dislikes)', fontsize=11, fontweight='bold')
axes[0, 0].set_ylabel('Absolute Error', fontsize=11, fontweight='bold')
axes[0, 0].set_title('Absolute Error vs Prediction - Training', fontsize=13, fontweight='bold', pad=15)
axes[0, 0].grid(True, alpha=0.3)

# Error distribution by prediction magnitude (Validation)
axes[0, 1].scatter(y_val_pred, np.abs(y_val - y_val_pred), alpha=0.3, s=10, c='green')
axes[0, 1].set_xlabel('Predicted log(dislikes)', fontsize=11, fontweight='bold')
axes[0, 1].set_ylabel('Absolute Error', fontsize=11, fontweight='bold')
axes[0, 1].set_title('Absolute Error vs Prediction - Validation', fontsize=13, fontweight='bold', pad=15)
axes[0, 1].grid(True, alpha=0.3)

# Q-Q plot for residuals (Training)
from scipy import stats
stats.probplot(y_train - y_train_pred, dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('Q-Q Plot - Training Residuals', fontsize=13, fontweight='bold', pad=15)
axes[1, 0].grid(True, alpha=0.3)

# Q-Q plot for residuals (Validation)
stats.probplot(y_val - y_val_pred, dist="norm", plot=axes[1, 1])
axes[1, 1].set_title('Q-Q Plot - Validation Residuals', fontsize=13, fontweight='bold', pad=15)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('xgboost_error_analysis.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: xgboost_error_analysis.png")

# ============================================================================
# 10. SAVE MODEL AND RESULTS
# ============================================================================
print("\n10. Saving model and results...")

# Save model
model.save_model('xgboost_model.json')
print("   ✓ Saved: xgboost_model.json")

# Save metrics to CSV
metrics_df = pd.DataFrame({
    'Dataset': ['Training', 'Validation'],
    'RMSE_Log': [train_metrics['rmse_log'], val_metrics['rmse_log']],
    'MAE_Log': [train_metrics['mae_log'], val_metrics['mae_log']],
    'R2_Log': [train_metrics['r2_log'], val_metrics['r2_log']],
    'RMSE_Raw': [train_metrics['rmse_raw'], val_metrics['rmse_raw']],
    'MAE_Raw': [train_metrics['mae_raw'], val_metrics['mae_raw']],
    'R2_Raw': [train_metrics['r2_raw'], val_metrics['r2_raw']],
    'MAPE_Raw': [train_metrics['mape_raw'], val_metrics['mape_raw']]
})
metrics_df.to_csv('xgboost_metrics.csv', index=False)
print("   ✓ Saved: xgboost_metrics.csv")

# Save feature importance
feature_importance.to_csv('xgboost_feature_importance.csv', index=False)
print("   ✓ Saved: xgboost_feature_importance.csv")

# Save predictions for further analysis
predictions_df = pd.DataFrame({
    'actual_log': np.concatenate([y_train, y_val]),
    'predicted_log': np.concatenate([y_train_pred, y_val_pred]),
    'actual_raw': np.concatenate([y_train_raw, y_val_raw]),
    'predicted_raw': np.concatenate([y_train_pred_raw, y_val_pred_raw]),
    'dataset': ['train'] * len(y_train) + ['validation'] * len(y_val)
})
predictions_df.to_csv('xgboost_predictions.csv', index=False)
print("   ✓ Saved: xgboost_predictions.csv")

# ============================================================================
# 11. SUMMARY
# ============================================================================
print("\n" + "="*80)
print("TRAINING SUMMARY")
print("="*80)
print(f"\n✓ Model: XGBoost Regressor")
print(f"✓ Target: log_dislikes (log-transformed)")
print(f"✓ Features: {len(FEATURES)} (Tier 2 Tree-Based)")
print(f"✓ Training samples: {len(X_train):,}")
print(f"✓ Validation samples: {len(X_val):,}")
print(f"✓ Test samples: {len(X_test):,} (NOT USED YET - Reserved for final evaluation)")
print(f"\n✓ Best Performance:")
print(f"   - Validation R² (log): {val_metrics['r2_log']:.4f}")
print(f"   - Validation R² (raw): {val_metrics['r2_raw']:.4f}")
print(f"   - Validation MAPE: {val_metrics['mape_raw']:.2f}%")
print(f"   - Validation MAE: {val_metrics['mae_raw']:,.2f} dislikes")
print(f"\n✓ Outputs saved:")
print(f"   1. xgboost_model.json")
print(f"   2. xgboost_metrics.csv")
print(f"   3. xgboost_feature_importance.csv")
print(f"   4. xgboost_predictions.csv")
print(f"   5. xgboost_performance_analysis.png")
print(f"   6. xgboost_raw_scale_predictions.png")
print(f"   7. xgboost_error_analysis.png")
print(f"\n✓ Test set is preserved and untouched for final model evaluation")
print("="*80)
print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
