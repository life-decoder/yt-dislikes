"""
Linear Regression Model Training for YouTube Dislikes Prediction
================================================================
Train-Validation-Test Split: 75-10-15
Target: log_dislikes (as recommended by feature engineering analysis)
Features: Tier 2 Tree-Based features (baseline comparison with XGBoost)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
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
print("Linear Regression Model Training for YouTube Dislikes Prediction")
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

# Tier 2 Tree-Based features (same as XGBoost for fair comparison)
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
# 5. FEATURE SCALING (Important for Linear Regression)
# ============================================================================
print("\n5. Scaling features (StandardScaler)...")
print("   Note: Feature scaling is crucial for Linear Regression!")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("   ✓ Features scaled using StandardScaler (mean=0, std=1)")

# ============================================================================
# 6. TRAIN LINEAR REGRESSION MODEL
# ============================================================================
print("\n6. Training Linear Regression model...")
print("   Hyperparameters:")
print("      - fit_intercept: True")
print("      - n_jobs: -1 (use all CPUs)")

# Create Linear Regression model
model = LinearRegression(
    fit_intercept=True,
    n_jobs=-1
)

# Train the model
model.fit(X_train_scaled, y_train)

print(f"   ✓ Training complete!")
print(f"   Intercept: {model.intercept_:.4f}")
print(f"   Number of coefficients: {len(model.coef_)}")

# ============================================================================
# 7. PREDICTIONS (Train and Validation only - NO TEST SET YET)
# ============================================================================
print("\n7. Making predictions on Train and Validation sets...")

y_train_pred = model.predict(X_train_scaled)
y_val_pred = model.predict(X_val_scaled)

# Convert predictions back to raw scale for interpretability
y_train_raw = np.expm1(y_train)
y_train_pred_raw = np.expm1(y_train_pred)
y_val_raw = np.expm1(y_val)
y_val_pred_raw = np.expm1(y_val_pred)

print("   ✓ Predictions complete!")

# ============================================================================
# 8. EVALUATE PERFORMANCE
# ============================================================================
print("\n8. Model Performance Evaluation")
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
# 9. COEFFICIENT ANALYSIS (Feature Importance for Linear Regression)
# ============================================================================
print("\n9. Coefficient Analysis (Feature Importance)...")

# Get coefficients
coefficients = pd.DataFrame({
    'feature': FEATURES,
    'coefficient': model.coef_,
    'abs_coefficient': np.abs(model.coef_)
}).sort_values('abs_coefficient', ascending=False)

print("\n  Coefficient Rankings (by absolute value):")
print("  " + "-" * 60)
for idx, row in coefficients.iterrows():
    sign = '+' if row['coefficient'] >= 0 else '-'
    print(f"    {row['feature']:25s}: {sign}{abs(row['coefficient']):8.4f}")

print("\n  Note: Coefficients represent the change in log_dislikes for a")
print("        1-standard-deviation change in the feature (after scaling).")

# ============================================================================
# 10. VISUALIZATIONS
# ============================================================================
print("\n10. Creating visualizations...")

# Create figure with subplots
fig = plt.figure(figsize=(20, 12))

# Plot 1: Coefficient Plot (Feature Importance)
ax1 = plt.subplot(2, 3, 1)
coef_sorted = coefficients.sort_values('coefficient')
colors = ['red' if x < 0 else 'green' for x in coef_sorted['coefficient']]
ax1.barh(range(len(coef_sorted)), coef_sorted['coefficient'], color=colors, alpha=0.7)
ax1.set_yticks(range(len(coef_sorted)))
ax1.set_yticklabels(coef_sorted['feature'], fontsize=10)
ax1.set_xlabel('Coefficient Value (Standardized)', fontsize=11, fontweight='bold')
ax1.set_title('Feature Coefficients (Linear Regression)', fontsize=13, fontweight='bold', pad=15)
ax1.axvline(x=0, color='black', linestyle='--', linewidth=1)
ax1.grid(True, alpha=0.3)

# Plot 2: Predictions vs Actual (Validation Set - Log Scale)
ax2 = plt.subplot(2, 3, 2)
ax2.scatter(y_val, y_val_pred, alpha=0.5, s=20, edgecolors='k', linewidths=0.5)
min_val = min(y_val.min(), y_val_pred.min())
max_val = max(y_val.max(), y_val_pred.max())
ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
ax2.set_xlabel('Actual log_dislikes', fontsize=11, fontweight='bold')
ax2.set_ylabel('Predicted log_dislikes', fontsize=11, fontweight='bold')
ax2.set_title(f'Validation Set: Predictions vs Actual (Log)\nR² = {val_metrics["r2_log"]:.4f}', 
              fontsize=13, fontweight='bold', pad=15)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# Plot 3: Residuals Distribution (Validation Set)
ax3 = plt.subplot(2, 3, 3)
residuals_val = y_val - y_val_pred
ax3.hist(residuals_val, bins=50, edgecolor='black', alpha=0.7)
ax3.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
ax3.set_xlabel('Residual (Actual - Predicted)', fontsize=11, fontweight='bold')
ax3.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax3.set_title(f'Residuals Distribution (Validation)\nMean: {residuals_val.mean():.4f}, Std: {residuals_val.std():.4f}', 
              fontsize=13, fontweight='bold', pad=15)
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

# Plot 4: Raw Scale Predictions vs Actual (Validation Set)
ax4 = plt.subplot(2, 3, 4)
ax4.scatter(y_val_raw, y_val_pred_raw, alpha=0.5, s=20, edgecolors='k', linewidths=0.5)
min_val_raw = min(y_val_raw.min(), y_val_pred_raw.min())
max_val_raw = max(y_val_raw.max(), y_val_pred_raw.max())
ax4.plot([min_val_raw, max_val_raw], [min_val_raw, max_val_raw], 'r--', linewidth=2, label='Perfect Prediction')
ax4.set_xlabel('Actual Dislikes', fontsize=11, fontweight='bold')
ax4.set_ylabel('Predicted Dislikes', fontsize=11, fontweight='bold')
ax4.set_title(f'Validation Set: Predictions vs Actual (Raw Scale)\nR² = {val_metrics["r2_raw"]:.4f}', 
              fontsize=13, fontweight='bold', pad=15)
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)

# Plot 5: Error Distribution (Absolute Error in Raw Scale)
ax5 = plt.subplot(2, 3, 5)
abs_errors = np.abs(y_val_raw - y_val_pred_raw)
ax5.hist(abs_errors, bins=50, edgecolor='black', alpha=0.7)
median_error = np.median(abs_errors)
ax5.axvline(x=median_error, color='red', linestyle='--', linewidth=2, 
            label=f'Median: {median_error:,.0f}')
ax5.set_xlabel('Absolute Error (Dislikes)', fontsize=11, fontweight='bold')
ax5.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax5.set_title('Absolute Error Distribution (Validation)', fontsize=13, fontweight='bold', pad=15)
ax5.legend(fontsize=10)
ax5.grid(True, alpha=0.3)

# Plot 6: Performance Comparison (Train vs Validation)
ax6 = plt.subplot(2, 3, 6)
metrics_comparison = {
    'R² (Log)': [train_metrics['r2_log'], val_metrics['r2_log']],
    'MAE (Raw)': [train_metrics['mae_raw'], val_metrics['mae_raw']],
    'RMSE (Raw)': [train_metrics['rmse_raw'], val_metrics['rmse_raw']]
}

x_pos = np.arange(len(metrics_comparison))
width = 0.35

# Normalize metrics for visualization
r2_train, r2_val = metrics_comparison['R² (Log)']
mae_train, mae_val = metrics_comparison['MAE (Raw)']
rmse_train, rmse_val = metrics_comparison['RMSE (Raw)']

# Normalize MAE and RMSE (divide by 1000 for better scale)
mae_train_norm, mae_val_norm = mae_train/1000, mae_val/1000
rmse_train_norm, rmse_val_norm = rmse_train/1000, rmse_val/1000

train_values = [r2_train, mae_train_norm, rmse_train_norm]
val_values = [r2_val, mae_val_norm, rmse_val_norm]

bars1 = ax6.bar(x_pos - width/2, train_values, width, label='Train', alpha=0.8)
bars2 = ax6.bar(x_pos + width/2, val_values, width, label='Validation', alpha=0.8)

ax6.set_ylabel('Metric Value', fontsize=11, fontweight='bold')
ax6.set_title('Performance Comparison: Train vs Validation', fontsize=13, fontweight='bold', pad=15)
ax6.set_xticks(x_pos)
ax6.set_xticklabels(['R² (Log)', 'MAE/1000', 'RMSE/1000'], fontsize=10)
ax6.legend(fontsize=10)
ax6.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('linear_regression_performance_analysis.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: linear_regression_performance_analysis.png")

# ============================================================================
# Additional Visualization: Raw Scale Predictions (Separate Figure)
# ============================================================================
fig2, axes = plt.subplots(1, 2, figsize=(16, 6))

# Left plot: Training set
axes[0].scatter(y_train_raw, y_train_pred_raw, alpha=0.3, s=10, edgecolors='k', linewidths=0.3)
min_val = min(y_train_raw.min(), y_train_pred_raw.min())
max_val = max(y_train_raw.max(), y_train_pred_raw.max())
axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
axes[0].set_xlabel('Actual Dislikes', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Predicted Dislikes', fontsize=12, fontweight='bold')
axes[0].set_title(f'Training Set (Raw Scale)\nR² = {train_metrics["r2_raw"]:.4f}', 
                 fontsize=14, fontweight='bold', pad=15)
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# Right plot: Validation set
axes[1].scatter(y_val_raw, y_val_pred_raw, alpha=0.5, s=20, edgecolors='k', linewidths=0.5)
min_val = min(y_val_raw.min(), y_val_pred_raw.min())
max_val = max(y_val_raw.max(), y_val_pred_raw.max())
axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
axes[1].set_xlabel('Actual Dislikes', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Predicted Dislikes', fontsize=12, fontweight='bold')
axes[1].set_title(f'Validation Set (Raw Scale)\nR² = {val_metrics["r2_raw"]:.4f}', 
                 fontsize=14, fontweight='bold', pad=15)
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('linear_regression_raw_scale_predictions.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: linear_regression_raw_scale_predictions.png")

# ============================================================================
# 11. SAVE RESULTS
# ============================================================================
print("\n11. Saving results...")

# Save model coefficients
coefficients.to_csv('linear_regression_coefficients.csv', index=False)
print("   ✓ Saved: linear_regression_coefficients.csv")

# Save scaler parameters
scaler_params = pd.DataFrame({
    'feature': FEATURES,
    'mean': scaler.mean_,
    'scale': scaler.scale_
})
scaler_params.to_csv('linear_regression_scaler_params.csv', index=False)
print("   ✓ Saved: linear_regression_scaler_params.csv")

# Save predictions
predictions_df = pd.DataFrame({
    'set': ['train'] * len(y_train) + ['validation'] * len(y_val),
    'actual_log': pd.concat([y_train.reset_index(drop=True), y_val.reset_index(drop=True)]),
    'predicted_log': np.concatenate([y_train_pred, y_val_pred]),
    'actual_raw': pd.concat([y_train_raw.reset_index(drop=True), y_val_raw.reset_index(drop=True)]),
    'predicted_raw': np.concatenate([y_train_pred_raw, y_val_pred_raw])
})
predictions_df.to_csv('linear_regression_predictions.csv', index=False)
print("   ✓ Saved: linear_regression_predictions.csv")

# Save metrics summary
metrics_df = pd.DataFrame({
    'set': ['training', 'validation'],
    'rmse_log': [train_metrics['rmse_log'], val_metrics['rmse_log']],
    'mae_log': [train_metrics['mae_log'], val_metrics['mae_log']],
    'r2_log': [train_metrics['r2_log'], val_metrics['r2_log']],
    'rmse_raw': [train_metrics['rmse_raw'], val_metrics['rmse_raw']],
    'mae_raw': [train_metrics['mae_raw'], val_metrics['mae_raw']],
    'r2_raw': [train_metrics['r2_raw'], val_metrics['r2_raw']],
    'mape_raw': [train_metrics['mape_raw'], val_metrics['mape_raw']]
})
metrics_df.to_csv('linear_regression_metrics.csv', index=False)
print("   ✓ Saved: linear_regression_metrics.csv")

# ============================================================================
# 12. FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("TRAINING COMPLETE - FINAL SUMMARY")
print("="*80)

print(f"\n📊 Dataset Split:")
print(f"   Training:   {len(X_train):,} samples (75%)")
print(f"   Validation: {len(X_val):,} samples (10%)")
print(f"   Test:       {len(X_test):,} samples (15%) - RESERVED")

print(f"\n🎯 Model Performance:")
print(f"   Validation R² (log):  {val_metrics['r2_log']:.4f}")
print(f"   Validation R² (raw):  {val_metrics['r2_raw']:.4f}")
print(f"   Validation MAE:       {val_metrics['mae_raw']:,.2f} dislikes")
print(f"   Validation RMSE:      {val_metrics['rmse_raw']:,.2f} dislikes")

print(f"\n📈 Overfitting Check:")
print(f"   R² difference: {r2_diff:.4f}")

print(f"\n🏆 Top 3 Most Important Features (by absolute coefficient):")
top_features = coefficients.head(3)
for idx, row in top_features.iterrows():
    print(f"   {row['feature']:25s}: {row['abs_coefficient']:.4f}")

print(f"\n📁 Output Files Created:")
print(f"   ✓ linear_regression_performance_analysis.png")
print(f"   ✓ linear_regression_raw_scale_predictions.png")
print(f"   ✓ linear_regression_coefficients.csv")
print(f"   ✓ linear_regression_scaler_params.csv")
print(f"   ✓ linear_regression_predictions.csv")
print(f"   ✓ linear_regression_metrics.csv")

print(f"\n⚠️  TEST SET STATUS: RESERVED (not used in training or evaluation)")
print(f"   The test set will only be used for final model evaluation after")
print(f"   model selection is complete.\n")

print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
