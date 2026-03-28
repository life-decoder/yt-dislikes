"""
Detailed Analysis for Linear Regression Model
=============================================
Perform in-depth error analysis, prediction quality assessment,
and residual diagnostics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

print("="*80)
print("LINEAR REGRESSION MODEL - DETAILED ANALYSIS")
print("="*80)

# Load data
predictions_df = pd.read_csv('linear_regression_predictions.csv')
metrics_df = pd.read_csv('linear_regression_metrics.csv')
coefficients_df = pd.read_csv('linear_regression_coefficients.csv')

# Separate train and validation
train_pred = predictions_df[predictions_df['set'] == 'train'].copy()
val_pred = predictions_df[predictions_df['set'] == 'validation'].copy()

# Calculate errors
train_pred['error_raw'] = train_pred['actual_raw'] - train_pred['predicted_raw']
val_pred['error_raw'] = val_pred['actual_raw'] - val_pred['predicted_raw']
train_pred['abs_error_raw'] = np.abs(train_pred['error_raw'])
val_pred['abs_error_raw'] = np.abs(val_pred['error_raw'])
train_pred['pct_error'] = (train_pred['abs_error_raw'] / train_pred['actual_raw']) * 100
val_pred['pct_error'] = (val_pred['abs_error_raw'] / val_pred['actual_raw']) * 100

print("\n1. ERROR STATISTICS")
print("="*80)

for name, df in [("Training", train_pred), ("Validation", val_pred)]:
    print(f"\n{name} Set:")
    print(f"  Mean Absolute Error:     {df['abs_error_raw'].mean():,.2f} dislikes")
    print(f"  Median Absolute Error:   {df['abs_error_raw'].median():,.2f} dislikes")
    print(f"  Std Dev of Errors:       {df['error_raw'].std():,.2f} dislikes")
    print(f"  Max Error:               {df['abs_error_raw'].max():,.2f} dislikes")
    print(f"  Mean Percentage Error:   {df['pct_error'].mean():.2f}%")
    print(f"  Median Percentage Error: {df['pct_error'].median():.2f}%")

print("\n\n2. PREDICTION ACCURACY BY THRESHOLD (Validation Set)")
print("="*80)

thresholds = [100, 250, 500, 1000, 2500, 5000, 10000]
print(f"\n{'Threshold':<15} {'Count':<10} {'Percentage':<12}")
print("-" * 40)
for thresh in thresholds:
    count = (val_pred['abs_error_raw'] <= thresh).sum()
    pct = count / len(val_pred) * 100
    print(f"±{thresh:,} dislikes{'':<5} {count:<10} {pct:>6.2f}%")

print("\n\n3. PERFORMANCE BY VIDEO SIZE (Validation Set)")
print("="*80)

# Define video size categories
def categorize_video_size(dislikes):
    if dislikes < 100:
        return 'Tiny (<100)'
    elif dislikes < 500:
        return 'Small (100-500)'
    elif dislikes < 2500:
        return 'Medium (500-2.5K)'
    elif dislikes < 10000:
        return 'Large (2.5K-10K)'
    else:
        return 'Huge (>10K)'

val_pred['size_category'] = val_pred['actual_raw'].apply(categorize_video_size)

print(f"\n{'Category':<20} {'Count':<10} {'Median Error':<15} {'Mean Error':<15} {'R²':<10}")
print("-" * 75)

for category in ['Tiny (<100)', 'Small (100-500)', 'Medium (500-2.5K)', 
                 'Large (2.5K-10K)', 'Huge (>10K)']:
    cat_data = val_pred[val_pred['size_category'] == category]
    if len(cat_data) > 0:
        median_err = cat_data['abs_error_raw'].median()
        mean_err = cat_data['abs_error_raw'].mean()
        
        # Calculate R² for this category
        ss_res = np.sum((cat_data['actual_raw'] - cat_data['predicted_raw'])**2)
        ss_tot = np.sum((cat_data['actual_raw'] - cat_data['actual_raw'].mean())**2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        print(f"{category:<20} {len(cat_data):<10} {median_err:>10,.0f}     {mean_err:>10,.0f}     {r2:>6.4f}")

print("\n\n4. RESIDUAL ANALYSIS (Validation Set)")
print("="*80)

residuals = val_pred['error_raw']

# Normality test
_, p_value = stats.normaltest(residuals)
print(f"\nNormality Test (D'Agostino-Pearson):")
print(f"  p-value: {p_value:.6f}")
if p_value > 0.05:
    print(f"  ✓ Residuals appear normally distributed (p > 0.05)")
else:
    print(f"  ⚠ Residuals may not be normally distributed (p < 0.05)")

print(f"\nResidual Statistics:")
print(f"  Mean:     {residuals.mean():,.2f} (should be ~0)")
print(f"  Median:   {residuals.median():,.2f}")
print(f"  Std Dev:  {residuals.std():,.2f}")
print(f"  Skewness: {residuals.skew():.4f}")
print(f"  Kurtosis: {residuals.kurtosis():.4f}")

print("\n\n5. COEFFICIENT ANALYSIS")
print("="*80)

print(f"\nAll Feature Coefficients:")
print(f"{'Rank':<6} {'Feature':<25} {'Coefficient':<12} {'Abs Value':<12}")
print("-" * 60)
for idx, row in coefficients_df.iterrows():
    sign = '+' if row['coefficient'] >= 0 else ''
    print(f"{idx+1:<6} {row['feature']:<25} {sign}{row['coefficient']:<11.4f} {row['abs_coefficient']:<11.4f}")

# Calculate contribution percentages
total_abs_coef = coefficients_df['abs_coefficient'].sum()
coefficients_df['contribution_pct'] = (coefficients_df['abs_coefficient'] / total_abs_coef) * 100

print(f"\nContribution to Total Effect (by absolute coefficient):")
for idx, row in coefficients_df.head(5).iterrows():
    print(f"  {row['feature']:25s}: {row['contribution_pct']:>6.2f}%")

print("\n\n6. MODEL COMPARISON METRICS")
print("="*80)

train_metrics = metrics_df[metrics_df['set'] == 'training'].iloc[0]
val_metrics = metrics_df[metrics_df['set'] == 'validation'].iloc[0]

comparison = pd.DataFrame({
    'Metric': ['R² (log)', 'R² (raw)', 'MAE (raw)', 'RMSE (raw)', 'MAPE (%)'],
    'Training': [
        train_metrics['r2_log'],
        train_metrics['r2_raw'],
        train_metrics['mae_raw'],
        train_metrics['rmse_raw'],
        train_metrics['mape_raw']
    ],
    'Validation': [
        val_metrics['r2_log'],
        val_metrics['r2_raw'],
        val_metrics['mae_raw'],
        val_metrics['rmse_raw'],
        val_metrics['mape_raw']
    ]
})

comparison['Difference'] = comparison['Training'] - comparison['Validation']
comparison['Diff %'] = (comparison['Difference'] / comparison['Training']) * 100

print(f"\n{'Metric':<15} {'Training':<15} {'Validation':<15} {'Difference':<15} {'Diff %':<10}")
print("-" * 75)
for _, row in comparison.iterrows():
    print(f"{row['Metric']:<15} {row['Training']:>12.4f}   {row['Validation']:>12.4f}   "
          f"{row['Difference']:>12.4f}   {row['Diff %']:>8.2f}%")

print("\n" + "="*80)
print("7. CREATING DETAILED VISUALIZATIONS")
print("="*80)

# Create detailed analysis figure
fig = plt.figure(figsize=(20, 12))

# Plot 1: Residuals vs Predicted (Validation)
ax1 = plt.subplot(2, 3, 1)
ax1.scatter(val_pred['predicted_raw'], val_pred['error_raw'], alpha=0.5, s=20)
ax1.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax1.set_xlabel('Predicted Dislikes', fontsize=11, fontweight='bold')
ax1.set_ylabel('Residual (Actual - Predicted)', fontsize=11, fontweight='bold')
ax1.set_title('Residual Plot (Validation)\nCheck for Homoscedasticity', 
              fontsize=13, fontweight='bold', pad=15)
ax1.grid(True, alpha=0.3)

# Plot 2: Q-Q Plot (Validation)
ax2 = plt.subplot(2, 3, 2)
stats.probplot(residuals, dist="norm", plot=ax2)
ax2.set_title('Q-Q Plot (Validation)\nCheck for Normality', 
              fontsize=13, fontweight='bold', pad=15)
ax2.grid(True, alpha=0.3)

# Plot 3: Error by Video Size
ax3 = plt.subplot(2, 3, 3)
categories = ['Tiny (<100)', 'Small (100-500)', 'Medium (500-2.5K)', 
              'Large (2.5K-10K)', 'Huge (>10K)']
median_errors = [val_pred[val_pred['size_category'] == cat]['abs_error_raw'].median() 
                 for cat in categories]
colors = sns.color_palette("husl", len(categories))
ax3.bar(range(len(categories)), median_errors, color=colors, alpha=0.7, edgecolor='black')
ax3.set_xticks(range(len(categories)))
ax3.set_xticklabels(categories, rotation=45, ha='right', fontsize=9)
ax3.set_ylabel('Median Absolute Error', fontsize=11, fontweight='bold')
ax3.set_title('Prediction Error by Video Size\n(Validation Set)', 
              fontsize=13, fontweight='bold', pad=15)
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: Cumulative Error Distribution
ax4 = plt.subplot(2, 3, 4)
sorted_errors = np.sort(val_pred['abs_error_raw'])
cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100
ax4.plot(sorted_errors, cumulative, linewidth=2)
ax4.set_xlabel('Absolute Error (Dislikes)', fontsize=11, fontweight='bold')
ax4.set_ylabel('Cumulative Percentage', fontsize=11, fontweight='bold')
ax4.set_title('Cumulative Error Distribution\n(Validation Set)', 
              fontsize=13, fontweight='bold', pad=15)
ax4.grid(True, alpha=0.3)
ax4.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50th percentile')
ax4.axhline(y=75, color='orange', linestyle='--', alpha=0.5, label='75th percentile')
ax4.legend(fontsize=9)

# Plot 5: Coefficient Magnitude
ax5 = plt.subplot(2, 3, 5)
top_features = coefficients_df.head(10)
colors_coef = ['green' if x >= 0 else 'red' for x in top_features['coefficient']]
ax5.barh(range(len(top_features)), top_features['abs_coefficient'], 
         color=colors_coef, alpha=0.7, edgecolor='black')
ax5.set_yticks(range(len(top_features)))
ax5.set_yticklabels(top_features['feature'], fontsize=10)
ax5.set_xlabel('Absolute Coefficient Value', fontsize=11, fontweight='bold')
ax5.set_title('Top 10 Features by Coefficient Magnitude\n(Green=Positive, Red=Negative)', 
              fontsize=13, fontweight='bold', pad=15)
ax5.grid(True, alpha=0.3, axis='x')

# Plot 6: Prediction Scatter with Density
ax6 = plt.subplot(2, 3, 6)
scatter = ax6.scatter(val_pred['actual_raw'], val_pred['predicted_raw'], 
                     c=val_pred['abs_error_raw'], cmap='viridis', 
                     alpha=0.6, s=30, edgecolors='k', linewidths=0.5)
min_val = min(val_pred['actual_raw'].min(), val_pred['predicted_raw'].min())
max_val = max(val_pred['actual_raw'].max(), val_pred['predicted_raw'].max())
ax6.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
ax6.set_xlabel('Actual Dislikes', fontsize=11, fontweight='bold')
ax6.set_ylabel('Predicted Dislikes', fontsize=11, fontweight='bold')
ax6.set_title('Predictions Colored by Error\n(Validation Set)', 
              fontsize=13, fontweight='bold', pad=15)
cbar = plt.colorbar(scatter, ax=ax6)
cbar.set_label('Absolute Error', fontsize=10, fontweight='bold')
ax6.legend(fontsize=9)
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('linear_regression_detailed_analysis.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: linear_regression_detailed_analysis.png")

print("\n" + "="*80)
print("DETAILED ANALYSIS COMPLETE")
print("="*80 + "\n")
