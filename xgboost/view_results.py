"""
Quick Summary Script - XGBoost Model Results
Displays key metrics and insights from training
"""

import pandas as pd
import numpy as np

print("="*80)
print("XGBoost Model Training Summary")
print("="*80)
print()

# Load metrics
metrics = pd.read_csv('xgboost_metrics.csv')
print("📊 PERFORMANCE METRICS")
print("-" * 80)
print(metrics.to_string(index=False))
print()

# Load feature importance
feature_imp = pd.read_csv('xgboost_feature_importance.csv')
print("🎯 TOP 5 FEATURE IMPORTANCE")
print("-" * 80)
for idx, row in feature_imp.head(5).iterrows():
    print(f"  {idx+1}. {row['feature']:25s} : {row['importance']:.4f} ({row['importance']*100:.1f}%)")
print()

# Load predictions sample
predictions = pd.read_csv('xgboost_predictions.csv')
val_preds = predictions[predictions['dataset'] == 'validation']

print("📈 VALIDATION SET PREDICTIONS (Sample)")
print("-" * 80)
print(f"  Total validation samples: {len(val_preds):,}")
print()
print("  Sample predictions (first 10):")
print(f"  {'Actual (raw)':>15s} {'Predicted (raw)':>18s} {'Error':>12s} {'% Error':>12s}")
print("  " + "-" * 60)

for i in range(min(10, len(val_preds))):
    actual = val_preds.iloc[i]['actual_raw']
    pred = val_preds.iloc[i]['predicted_raw']
    error = actual - pred
    pct_error = (error / actual * 100) if actual > 0 else 0
    print(f"  {actual:>15,.0f} {pred:>18,.0f} {error:>12,.0f} {pct_error:>11,.1f}%")

print()

# Error statistics
errors = val_preds['actual_raw'] - val_preds['predicted_raw']
abs_errors = np.abs(errors)

print("📊 VALIDATION ERROR STATISTICS")
print("-" * 80)
print(f"  Mean Absolute Error:    {abs_errors.mean():>10,.2f} dislikes")
print(f"  Median Absolute Error:  {abs_errors.median():>10,.2f} dislikes")
print(f"  Std of Errors:          {errors.std():>10,.2f} dislikes")
print(f"  Max Absolute Error:     {abs_errors.max():>10,.0f} dislikes")
print(f"  Min Absolute Error:     {abs_errors.min():>10,.2f} dislikes")
print()

# Accuracy buckets
within_500 = (abs_errors <= 500).sum()
within_1000 = (abs_errors <= 1000).sum()
within_2500 = (abs_errors <= 2500).sum()
within_5000 = (abs_errors <= 5000).sum()

print("🎯 PREDICTION ACCURACY DISTRIBUTION")
print("-" * 80)
print(f"  Within ±500 dislikes:   {within_500:>5,} samples ({within_500/len(val_preds)*100:>5.1f}%)")
print(f"  Within ±1,000 dislikes: {within_1000:>5,} samples ({within_1000/len(val_preds)*100:>5.1f}%)")
print(f"  Within ±2,500 dislikes: {within_2500:>5,} samples ({within_2500/len(val_preds)*100:>5.1f}%)")
print(f"  Within ±5,000 dislikes: {within_5000:>5,} samples ({within_5000/len(val_preds)*100:>5.1f}%)")
print()

# Overfitting check
train_r2 = metrics[metrics['Dataset'] == 'Training']['R2_Log'].values[0]
val_r2 = metrics[metrics['Dataset'] == 'Validation']['R2_Log'].values[0]
r2_diff = train_r2 - val_r2

print("🔍 OVERFITTING ANALYSIS")
print("-" * 80)
print(f"  Training R² (log):      {train_r2:.4f}")
print(f"  Validation R² (log):    {val_r2:.4f}")
print(f"  Difference:             {r2_diff:.4f} ({r2_diff*100:.2f}%)")
if r2_diff < 0.05:
    status = "✅ Excellent - Minimal overfitting"
elif r2_diff < 0.10:
    status = "⚠️  Good - Slight overfitting"
else:
    status = "❌ Poor - Significant overfitting"
print(f"  Status:                 {status}")
print()

print("="*80)
print("✅ Model Training Complete!")
print("📁 View visualizations: xgboost_performance_analysis.png")
print("📁 Full report: TRAINING_RESULTS.md")
print("="*80)
