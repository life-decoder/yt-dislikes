"""
Create a comprehensive summary visualization for the XGBoost model
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-darkgrid')

# Create figure
fig = plt.figure(figsize=(16, 10))
fig.suptitle('XGBoost Model - Complete Performance Summary', 
             fontsize=18, fontweight='bold', y=0.98)

# Load data
metrics = pd.read_csv('xgboost_metrics.csv')
feature_imp = pd.read_csv('xgboost_feature_importance.csv')
predictions = pd.read_csv('xgboost_predictions.csv')
val_preds = predictions[predictions['dataset'] == 'validation']

# Calculate additional stats
val_preds['abs_error'] = np.abs(val_preds['actual_raw'] - val_preds['predicted_raw'])

# Subplot 1: Model Performance Comparison
ax1 = plt.subplot(2, 3, 1)
metrics_to_plot = metrics[['Dataset', 'R2_Log', 'R2_Raw']].set_index('Dataset')
x = np.arange(len(metrics_to_plot))
width = 0.35
bars1 = ax1.bar(x - width/2, metrics_to_plot['R2_Log'], width, 
                label='R² (Log)', color='#3498db', alpha=0.8)
bars2 = ax1.bar(x + width/2, metrics_to_plot['R2_Raw'], width, 
                label='R² (Raw)', color='#e74c3c', alpha=0.8)
ax1.set_ylabel('R² Score', fontsize=11, fontweight='bold')
ax1.set_title('Model Performance\n(Train vs Validation)', fontsize=12, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(metrics_to_plot.index)
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')
ax1.set_ylim(0, 1)

# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
for bar in bars2:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Subplot 2: Top 5 Feature Importance
ax2 = plt.subplot(2, 3, 2)
top5 = feature_imp.head(5)
colors_list = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
bars = ax2.barh(range(len(top5)), top5['importance'], color=colors_list, alpha=0.8)
ax2.set_yticks(range(len(top5)))
ax2.set_yticklabels(top5['feature'])
ax2.set_xlabel('Importance Score', fontsize=11, fontweight='bold')
ax2.set_title('Top 5 Feature Importance\n(81% of total)', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='x')
ax2.invert_yaxis()

# Add percentages
for i, bar in enumerate(bars):
    width_val = bar.get_width()
    ax2.text(width_val, bar.get_y() + bar.get_height()/2,
             f' {width_val:.3f} ({width_val*100:.1f}%)',
             ha='left', va='center', fontsize=9, fontweight='bold')

# Subplot 3: Prediction Accuracy Distribution
ax3 = plt.subplot(2, 3, 3)
thresholds = [500, 1000, 2500, 5000, 10000]
accuracy_pcts = []
for thresh in thresholds:
    pct = (val_preds['abs_error'] <= thresh).sum() / len(val_preds) * 100
    accuracy_pcts.append(pct)

bars = ax3.bar(range(len(thresholds)), accuracy_pcts, 
               color=['#2ecc71', '#27ae60', '#3498db', '#2980b9', '#9b59b6'],
               alpha=0.8)
ax3.set_xticks(range(len(thresholds)))
ax3.set_xticklabels([f'±{t:,}' for t in thresholds], rotation=45)
ax3.set_xlabel('Error Threshold (dislikes)', fontsize=11, fontweight='bold')
ax3.set_ylabel('% of Predictions', fontsize=11, fontweight='bold')
ax3.set_title('Prediction Accuracy\n(Validation Set)', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')
ax3.set_ylim(0, 100)

# Add percentage labels
for bar in bars:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Subplot 4: Error Statistics
ax4 = plt.subplot(2, 3, 4)
error_stats = {
    'Mean Error': val_preds['abs_error'].mean(),
    'Median Error': val_preds['abs_error'].median(),
    '75th Percentile': val_preds['abs_error'].quantile(0.75),
    '90th Percentile': val_preds['abs_error'].quantile(0.90),
    '95th Percentile': val_preds['abs_error'].quantile(0.95)
}
bars = ax4.barh(range(len(error_stats)), list(error_stats.values()),
                color=['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6'],
                alpha=0.8)
ax4.set_yticks(range(len(error_stats)))
ax4.set_yticklabels(list(error_stats.keys()))
ax4.set_xlabel('Absolute Error (dislikes)', fontsize=11, fontweight='bold')
ax4.set_title('Error Statistics\n(Validation Set)', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='x')
ax4.set_xscale('log')
ax4.invert_yaxis()

# Add value labels
for bar in bars:
    width_val = bar.get_width()
    ax4.text(width_val, bar.get_y() + bar.get_height()/2,
             f' {width_val:,.0f}',
             ha='left', va='center', fontsize=9, fontweight='bold')

# Subplot 5: Data Split Overview
ax5 = plt.subplot(2, 3, 5)
split_data = {
    'Training\n(75%)': 23150,
    'Validation\n(10%)': 3086,
    'Test\n(15%)\n[RESERVED]': 4631
}
colors_split = ['#3498db', '#2ecc71', '#e74c3c']
explode = (0, 0.05, 0.1)
wedges, texts, autotexts = ax5.pie(list(split_data.values()), 
                                     labels=list(split_data.keys()),
                                     autopct='%1.1f%%',
                                     startangle=90,
                                     colors=colors_split,
                                     explode=explode,
                                     textprops={'fontsize': 10, 'fontweight': 'bold'})
ax5.set_title('Dataset Split\n(30,867 total samples)', fontsize=12, fontweight='bold')

# Subplot 6: Key Metrics Summary (Text Box)
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')

summary_text = f"""
╔══════════════════════════════════════╗
║       KEY PERFORMANCE METRICS        ║
╚══════════════════════════════════════╝

📊 VALIDATION PERFORMANCE
   • R² (log scale):      0.8369 (83.7%)
   • R² (raw scale):      0.8206 (82.1%)
   • MAE:                 1,793 dislikes
   • Median Error:        290 dislikes

🎯 PREDICTION ACCURACY
   • Within ±500:         62.3%
   • Within ±1,000:       75.7%
   • Within ±2,500:       88.2%
   • Within ±5,000:       93.4%

⚠️  OVERFITTING ANALYSIS
   • Train R² (log):      0.8944
   • Val R² (log):        0.8369
   • Difference:          5.76%
   • Status:              Slight overfitting

⏱️  EFFICIENCY
   • Training Time:       ~8 seconds
   • Boosting Rounds:     200
   • Features Used:       10

🏆 CONCLUSION
   ✓ Strong baseline performance
   ✓ Good generalization
   ✓ Fast training & inference
   ✓ Interpretable results
"""

ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
         fontsize=9, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('xgboost_summary_dashboard.png', dpi=300, bbox_inches='tight')
print("✓ Created: xgboost_summary_dashboard.png")
print("\n✅ Summary visualization complete!")
