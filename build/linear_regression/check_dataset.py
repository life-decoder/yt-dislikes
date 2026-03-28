"""
Dataset Inspector for Linear Regression Training
================================================
Quick check of dataset before training
"""

import pandas as pd
import numpy as np

print("="*80)
print("DATASET INSPECTION FOR LINEAR REGRESSION")
print("="*80)

# Load dataset
try:
    df = pd.read_csv('../yt_dataset_v4.csv')
    print(f"\n✓ Dataset loaded successfully")
    print(f"  Shape: {df.shape}")
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")
except FileNotFoundError:
    print("\n❌ Dataset not found at: ../yt_dataset_v4.csv")
    exit(1)

# Check for required features
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
TARGET = 'log_dislikes'

print(f"\n📊 FEATURE AVAILABILITY CHECK")
print("-" * 80)

all_features = FEATURES + [TARGET]
missing_features = []

for feat in all_features:
    if feat in df.columns:
        null_count = df[feat].isnull().sum()
        null_pct = (null_count / len(df)) * 100
        status = "✓"
        print(f"  {status} {feat:30s} - {null_count:6,} nulls ({null_pct:5.2f}%)")
    else:
        status = "✗"
        print(f"  {status} {feat:30s} - MISSING!")
        missing_features.append(feat)

if missing_features:
    print(f"\n❌ Missing features: {', '.join(missing_features)}")
    print("   Training cannot proceed!")
    exit(1)
else:
    print(f"\n✓ All required features present!")

# Basic statistics
print(f"\n📈 TARGET VARIABLE STATISTICS (log_dislikes)")
print("-" * 80)
target_data = df[TARGET].dropna()
print(f"  Count:  {len(target_data):,}")
print(f"  Mean:   {target_data.mean():.4f}")
print(f"  Median: {target_data.median():.4f}")
print(f"  Std:    {target_data.std():.4f}")
print(f"  Min:    {target_data.min():.4f}")
print(f"  Max:    {target_data.max():.4f}")

# Raw dislikes distribution
if 'dislikes' in df.columns:
    print(f"\n📊 RAW DISLIKES DISTRIBUTION")
    print("-" * 80)
    dislikes = df['dislikes'].dropna()
    print(f"  Count:  {len(dislikes):,}")
    print(f"  Mean:   {dislikes.mean():,.2f}")
    print(f"  Median: {dislikes.median():,.2f}")
    print(f"  Min:    {dislikes.min():,.0f}")
    print(f"  Max:    {dislikes.max():,.0f}")
    
    # Size distribution
    print(f"\n  Distribution by size:")
    print(f"    < 100:           {(dislikes < 100).sum():,} ({(dislikes < 100).sum()/len(dislikes)*100:.1f}%)")
    print(f"    100 - 500:       {((dislikes >= 100) & (dislikes < 500)).sum():,} ({((dislikes >= 100) & (dislikes < 500)).sum()/len(dislikes)*100:.1f}%)")
    print(f"    500 - 2,500:     {((dislikes >= 500) & (dislikes < 2500)).sum():,} ({((dislikes >= 500) & (dislikes < 2500)).sum()/len(dislikes)*100:.1f}%)")
    print(f"    2,500 - 10,000:  {((dislikes >= 2500) & (dislikes < 10000)).sum():,} ({((dislikes >= 2500) & (dislikes < 10000)).sum()/len(dislikes)*100:.1f}%)")
    print(f"    > 10,000:        {(dislikes >= 10000).sum():,} ({(dislikes >= 10000).sum()/len(dislikes)*100:.1f}%)")

# Feature correlations with target
print(f"\n🔗 FEATURE CORRELATIONS WITH TARGET")
print("-" * 80)
correlations = []
for feat in FEATURES:
    if feat in df.columns:
        corr = df[[feat, TARGET]].corr().iloc[0, 1]
        correlations.append((feat, corr))

correlations.sort(key=lambda x: abs(x[1]), reverse=True)

print(f"  {'Feature':<30} {'Correlation':<15}")
print("  " + "-" * 50)
for feat, corr in correlations:
    sign = '+' if corr >= 0 else ''
    print(f"  {feat:<30} {sign}{corr:>8.4f}")

print("\n" + "="*80)
print("DATASET INSPECTION COMPLETE")
print("="*80)
print("\n✓ Dataset is ready for training!")
print("  Run: python train_linear_regression_model.py\n")
