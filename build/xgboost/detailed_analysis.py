"""
Additional Analysis: Prediction Confidence and Error Analysis by Video Size
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("Creating additional analysis visualizations...")

# Load predictions
predictions = pd.read_csv('xgboost_predictions.csv')
val_preds = predictions[predictions['dataset'] == 'validation'].copy()

# Calculate errors
val_preds['error'] = val_preds['actual_raw'] - val_preds['predicted_raw']
val_preds['abs_error'] = np.abs(val_preds['error'])
val_preds['pct_error'] = np.where(
    val_preds['actual_raw'] > 0,
    (val_preds['abs_error'] / val_preds['actual_raw']) * 100,
    0
)

# Create bins for video size
val_preds['size_category'] = pd.cut(
    val_preds['actual_raw'],
    bins=[0, 100, 500, 1000, 5000, 10000, np.inf],
    labels=['Tiny\n(0-100)', 'Small\n(100-500)', 'Medium\n(500-1K)', 
            'Large\n(1K-5K)', 'Very Large\n(5K-10K)', 'Huge\n(10K+)']
)

# Create comprehensive analysis figure
fig = plt.figure(figsize=(20, 12))

# Plot 1: Error by video size (Box plot)
ax1 = plt.subplot(2, 3, 1)
box_data = [val_preds[val_preds['size_category'] == cat]['abs_error'].values 
            for cat in val_preds['size_category'].cat.categories]
bp = ax1.boxplot(box_data, labels=val_preds['size_category'].cat.categories,
                 patch_artist=True, showfliers=True)
for patch, color in zip(bp['boxes'], sns.color_palette("husl", len(bp['boxes']))):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax1.set_ylabel('Absolute Error (dislikes)', fontsize=11, fontweight='bold')
ax1.set_xlabel('Video Size Category', fontsize=11, fontweight='bold')
ax1.set_title('Prediction Error Distribution by Video Size', fontsize=13, fontweight='bold', pad=15)
ax1.grid(True, alpha=0.3, axis='y')
ax1.set_yscale('log')

# Plot 2: Percentage error by video size
ax2 = plt.subplot(2, 3, 2)
# Cap percentage errors for better visualization
val_preds_capped = val_preds.copy()
val_preds_capped['pct_error_capped'] = val_preds_capped['pct_error'].clip(0, 200)
box_data2 = [val_preds_capped[val_preds_capped['size_category'] == cat]['pct_error_capped'].values 
             for cat in val_preds_capped['size_category'].cat.categories]
bp2 = ax2.boxplot(box_data2, labels=val_preds_capped['size_category'].cat.categories,
                  patch_artist=True, showfliers=False)
for patch, color in zip(bp2['boxes'], sns.color_palette("husl", len(bp2['boxes']))):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax2.set_ylabel('Percentage Error (%) [Capped at 200%]', fontsize=11, fontweight='bold')
ax2.set_xlabel('Video Size Category', fontsize=11, fontweight='bold')
ax2.set_title('Percentage Error by Video Size', fontsize=13, fontweight='bold', pad=15)
ax2.grid(True, alpha=0.3, axis='y')
ax2.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% error line')
ax2.legend()

# Plot 3: Sample count by category
ax3 = plt.subplot(2, 3, 3)
category_counts = val_preds['size_category'].value_counts().sort_index()
bars = ax3.bar(range(len(category_counts)), category_counts.values, 
               color=sns.color_palette("husl", len(category_counts)), alpha=0.8)
ax3.set_xticks(range(len(category_counts)))
ax3.set_xticklabels(category_counts.index, rotation=0)
ax3.set_ylabel('Number of Samples', fontsize=11, fontweight='bold')
ax3.set_xlabel('Video Size Category', fontsize=11, fontweight='bold')
ax3.set_title('Validation Set Distribution by Size', fontsize=13, fontweight='bold', pad=15)
ax3.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}',
             ha='center', va='bottom', fontweight='bold')

# Plot 4: Prediction scatter with confidence bands
ax4 = plt.subplot(2, 3, 4)
# Calculate prediction confidence (based on error std at different ranges)
sorted_vals = val_preds.sort_values('predicted_raw')
window_size = 300
rolling_std = sorted_vals['error'].rolling(window=window_size, center=True).std()
rolling_std = rolling_std.fillna(rolling_std.mean())

ax4.scatter(val_preds['predicted_raw'], val_preds['actual_raw'], 
           alpha=0.3, s=10, c='blue', label='Predictions')
ax4.plot([val_preds['predicted_raw'].min(), val_preds['predicted_raw'].max()],
         [val_preds['predicted_raw'].min(), val_preds['predicted_raw'].max()],
         'r--', lw=2, label='Perfect Prediction')
ax4.set_xlabel('Predicted Dislikes', fontsize=11, fontweight='bold')
ax4.set_ylabel('Actual Dislikes', fontsize=11, fontweight='bold')
ax4.set_title('Predictions with Log Scale', fontsize=13, fontweight='bold', pad=15)
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.set_xscale('log')
ax4.set_yscale('log')

# Plot 5: Error vs predicted (linear scale, zoomed)
ax5 = plt.subplot(2, 3, 5)
# Filter to reasonable range for visualization
plot_data = val_preds[val_preds['predicted_raw'] < 20000]
ax5.scatter(plot_data['predicted_raw'], plot_data['error'], 
           alpha=0.3, s=10, c='orange')
ax5.axhline(y=0, color='red', linestyle='--', lw=2, label='Zero Error')
ax5.set_xlabel('Predicted Dislikes', fontsize=11, fontweight='bold')
ax5.set_ylabel('Prediction Error (Actual - Predicted)', fontsize=11, fontweight='bold')
ax5.set_title('Error Distribution (Videos < 20K dislikes)', fontsize=13, fontweight='bold', pad=15)
ax5.legend()
ax5.grid(True, alpha=0.3)

# Plot 6: Cumulative accuracy
ax6 = plt.subplot(2, 3, 6)
error_thresholds = np.linspace(0, 10000, 100)
cumulative_pct = [100 * (val_preds['abs_error'] <= thresh).sum() / len(val_preds) 
                  for thresh in error_thresholds]
ax6.plot(error_thresholds, cumulative_pct, linewidth=3, color='green')
ax6.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% of predictions')
ax6.axhline(y=75, color='orange', linestyle='--', alpha=0.5, label='75% of predictions')
ax6.axhline(y=90, color='blue', linestyle='--', alpha=0.5, label='90% of predictions')
ax6.set_xlabel('Error Threshold (dislikes)', fontsize=11, fontweight='bold')
ax6.set_ylabel('Cumulative % of Predictions', fontsize=11, fontweight='bold')
ax6.set_title('Cumulative Prediction Accuracy', fontsize=13, fontweight='bold', pad=15)
ax6.legend()
ax6.grid(True, alpha=0.3)
ax6.set_xlim(0, 10000)

plt.tight_layout()
plt.savefig('xgboost_detailed_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: xgboost_detailed_analysis.png")

# Print summary statistics by category
print("\n" + "="*80)
print("PERFORMANCE BY VIDEO SIZE CATEGORY")
print("="*80)

for cat in val_preds['size_category'].cat.categories:
    cat_data = val_preds[val_preds['size_category'] == cat]
    print(f"\n{cat}:")
    print(f"  Sample size:          {len(cat_data):>6,}")
    print(f"  Mean Abs Error:       {cat_data['abs_error'].mean():>6,.0f} dislikes")
    print(f"  Median Abs Error:     {cat_data['abs_error'].median():>6,.0f} dislikes")
    print(f"  Mean % Error:         {cat_data['pct_error'].mean():>6,.1f}%")
    print(f"  Median % Error:       {cat_data['pct_error'].median():>6,.1f}%")
    print(f"  R² score:             {np.corrcoef(cat_data['actual_raw'], cat_data['predicted_raw'])[0,1]**2:>6.4f}")

print("\n" + "="*80)
print("Analysis complete!")
print("="*80)
