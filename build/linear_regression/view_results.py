"""
Quick Results Viewer for Linear Regression Model
================================================
View key metrics and performance statistics without retraining
"""

import pandas as pd
import numpy as np

print("="*80)
print("LINEAR REGRESSION MODEL - QUICK RESULTS")
print("="*80)

# Load metrics
try:
    metrics_df = pd.read_csv('linear_regression_metrics.csv')
    
    print("\n📊 MODEL PERFORMANCE METRICS")
    print("-" * 80)
    
    for _, row in metrics_df.iterrows():
        set_name = row['set'].upper()
        print(f"\n{set_name} SET:")
        print(f"  Log Scale:")
        print(f"    R²:   {row['r2_log']:.4f}")
        print(f"    MAE:  {row['mae_log']:.4f}")
        print(f"    RMSE: {row['rmse_log']:.4f}")
        print(f"  Raw Scale:")
        print(f"    R²:   {row['r2_raw']:.4f}")
        print(f"    MAE:  {row['mae_raw']:,.2f} dislikes")
        print(f"    RMSE: {row['rmse_raw']:,.2f} dislikes")
        print(f"    MAPE: {row['mape_raw']:.2f}%")
    
    # Overfitting analysis
    train_r2 = metrics_df[metrics_df['set'] == 'training']['r2_log'].values[0]
    val_r2 = metrics_df[metrics_df['set'] == 'validation']['r2_log'].values[0]
    r2_diff = train_r2 - val_r2
    
    print(f"\n🔍 OVERFITTING ANALYSIS:")
    print(f"  R² Difference (Train - Val): {r2_diff:.4f}")
    if r2_diff < 0.05:
        print("  ✓ Model generalizes well (minimal overfitting)")
    elif r2_diff < 0.10:
        print("  ⚠ Slight overfitting detected")
    else:
        print("  ⚠⚠ Significant overfitting detected")
    
except FileNotFoundError:
    print("\n❌ Metrics file not found. Please run train_linear_regression_model.py first.")

# Load coefficients
try:
    coefficients_df = pd.read_csv('linear_regression_coefficients.csv')
    
    print("\n\n📈 FEATURE COEFFICIENTS (Top 5)")
    print("-" * 80)
    
    top_5 = coefficients_df.head(5)
    for i, (idx, row) in enumerate(top_5.iterrows(), 1):
        sign = '+' if row['coefficient'] >= 0 else '-'
        print(f"  {i}. {row['feature']:25s}: {sign}{abs(row['coefficient']):8.4f}")
    
    print("\n  Note: Coefficients show impact per 1-std change (after scaling)")
    
except FileNotFoundError:
    print("\n❌ Coefficients file not found.")

# Load predictions for additional stats
try:
    predictions_df = pd.read_csv('linear_regression_predictions.csv')
    val_pred = predictions_df[predictions_df['set'] == 'validation']
    
    print("\n\n📊 PREDICTION ACCURACY (Validation Set)")
    print("-" * 80)
    
    errors = np.abs(val_pred['actual_raw'] - val_pred['predicted_raw'])
    median_error = np.median(errors)
    
    thresholds = [500, 1000, 2500, 5000]
    print(f"  Median Absolute Error: {median_error:,.0f} dislikes")
    print(f"\n  Accuracy within threshold:")
    for thresh in thresholds:
        pct = (errors <= thresh).sum() / len(errors) * 100
        print(f"    ±{thresh:,} dislikes: {pct:.1f}%")
    
except FileNotFoundError:
    print("\n❌ Predictions file not found.")

print("\n" + "="*80)
print("For detailed analysis, run: python detailed_analysis.py")
print("="*80 + "\n")
