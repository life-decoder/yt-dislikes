"""
Create Filtered Dataset with Selected Features
================================================

This script creates a new dataset containing only the features selected
for model training, excluding all features with data leakage.

Features included:
- Core engagement metrics (view_count, likes, comment_count)
- Log-transformed versions (for linear models)
- Sentiment features (avg_pos, avg_neu, avg_neg, avg_compound)
- Metadata (age, no_comments, comment_sample_size, view_like_ratio)
- Target variable (dislikes)

Features excluded:
- All features derived from dislikes (10 features with data leakage)
- Identifier fields (video_id, channel_id, published_at retained for reference)

Author: Feature Engineering System
Date: October 10, 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 80)
print("CREATING FILTERED DATASET FOR MODEL TRAINING")
print("=" * 80)
print()

# Load original dataset
print("Loading original dataset...")
df = pd.read_csv('yt_dataset_en_v3.csv')
print(f"Original dataset: {df.shape[0]} rows, {df.shape[1]} columns")
print()

# Convert object columns to numeric where appropriate
numeric_cols = ['log_likes', 'log_dislikes', 'log_comment_count', 
                'log_dislike_like_ratio', 'log_like_dislike_score', 'log_engagement_rate']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Define feature sets
identifier_features = [
    'video_id',
    'channel_id',
    'published_at'
]

# Target variable
target_feature = ['dislikes']

# Features for modeling (no data leakage)
modeling_features = [
    # Raw engagement metrics
    'view_count',
    'likes',
    'comment_count',
    
    # Log-transformed engagement metrics
    'log_view_count',
    'log_likes',
    'log_comment_count',
    
    # Sentiment features
    'avg_pos',
    'avg_neu',
    'avg_neg',
    'avg_compound',
    
    # Metadata
    'age',
    'no_comments',
    'comment_sample_size',
    'view_like_ratio',
    'desc_lang'
]

# Combine all features to keep
features_to_keep = identifier_features + target_feature + modeling_features

print("Feature Selection Summary:")
print("-" * 80)
print(f"Identifier features: {len(identifier_features)}")
print(f"Target variable: {len(target_feature)}")
print(f"Modeling features: {len(modeling_features)}")
print(f"Total features to keep: {len(features_to_keep)}")
print()

# Create filtered dataset
print("Creating filtered dataset...")
df_filtered = df[features_to_keep].copy()
print(f"Filtered dataset: {df_filtered.shape[0]} rows, {df_filtered.shape[1]} columns")
print()

# Check for missing values
print("Missing Values Summary:")
print("-" * 80)
missing_counts = df_filtered.isnull().sum()
missing_features = missing_counts[missing_counts > 0]
if len(missing_features) > 0:
    print(missing_features)
else:
    print("No missing values found!")
print()

# Basic statistics
print("Dataset Statistics:")
print("-" * 80)
print(f"Target variable (dislikes):")
print(f"  Mean: {df_filtered['dislikes'].mean():,.2f}")
print(f"  Median: {df_filtered['dislikes'].median():,.2f}")
print(f"  Std Dev: {df_filtered['dislikes'].std():,.2f}")
print(f"  Min: {df_filtered['dislikes'].min():,.0f}")
print(f"  Max: {df_filtered['dislikes'].max():,.0f}")
print()

# Save filtered dataset
output_path = 'yt_dataset_filtered.csv'
print(f"Saving filtered dataset to: {output_path}")
df_filtered.to_csv(output_path, index=False)
print("Dataset saved successfully!")
print()

# Create feature sets for different model types
print("=" * 80)
print("CREATING FEATURE SET FILES")
print("=" * 80)
print()

# Tier 1: Essential features (minimal model)
tier1_features = [
    'view_count',
    'likes',
    'comment_count'
]

# Tier 2: Core features (recommended for tree-based models)
tier2_tree_features = [
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

# Tier 2: Core features (recommended for linear models)
tier2_linear_features = [
    'log_view_count',
    'log_likes',
    'log_comment_count',
    'avg_compound',
    'avg_pos',
    'avg_neg',
    'comment_sample_size',
    'no_comments',
    'view_like_ratio',
    'age'
]

# Tier 3: Extended features (all available, tree-based)
tier3_tree_features = [
    'view_count',
    'likes',
    'comment_count',
    'avg_pos',
    'avg_neu',
    'avg_neg',
    'avg_compound',
    'comment_sample_size',
    'no_comments',
    'view_like_ratio',
    'age',
    'desc_lang'
]

# Tier 3: Extended features (all available, linear models)
tier3_linear_features = [
    'log_view_count',
    'log_likes',
    'log_comment_count',
    'avg_pos',
    'avg_neu',
    'avg_neg',
    'avg_compound',
    'comment_sample_size',
    'no_comments',
    'view_like_ratio',
    'age',
    'desc_lang'
]

# Save feature sets
feature_sets = {
    'tier1_essential': tier1_features,
    'tier2_tree_based': tier2_tree_features,
    'tier2_linear': tier2_linear_features,
    'tier3_tree_based': tier3_tree_features,
    'tier3_linear': tier3_linear_features
}

# Create feature sets directory
feature_sets_dir = Path('feature_engineering/feature_sets')
feature_sets_dir.mkdir(parents=True, exist_ok=True)

for set_name, features in feature_sets.items():
    filepath = feature_sets_dir / f'{set_name}.txt'
    with open(filepath, 'w') as f:
        f.write(f"# {set_name.upper().replace('_', ' ')}\n")
        f.write(f"# Total features: {len(features)}\n")
        f.write(f"# Generated: October 10, 2025\n\n")
        for feature in features:
            f.write(f"{feature}\n")
    print(f"Saved feature set: {set_name} ({len(features)} features)")

print()

# Create Python file with feature lists
print("Creating Python feature configuration file...")
config_content = '''"""
Feature Configuration for Model Training
=========================================

This file contains pre-defined feature sets for different model types.
Import this file to use consistent feature selections across experiments.

Usage:
    from feature_sets_config import TIER1_FEATURES, TIER2_TREE, TARGET
    
    X_train = df_train[TIER2_TREE]
    y_train = df_train[TARGET]

Generated: October 10, 2025
"""

# Target variable
TARGET = 'dislikes'

# Identifier fields (not for modeling)
IDENTIFIERS = [
    'video_id',
    'channel_id',
    'published_at'
]

# TIER 1: Essential Features (Minimal Model)
# Use case: Baseline model, quick experiments
# Expected R²: 0.70-0.75
TIER1_FEATURES = [
    'view_count',
    'likes',
    'comment_count'
]

# TIER 2: Core Features - Tree-Based Models (RECOMMENDED)
# Use case: XGBoost, Random Forest, LightGBM
# Expected R²: 0.75-0.82
TIER2_TREE = [
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

# TIER 2: Core Features - Linear Models
# Use case: Ridge, Lasso, Elastic Net
# Expected R²: 0.75-0.82
TIER2_LINEAR = [
    'log_view_count',
    'log_likes',
    'log_comment_count',
    'avg_compound',
    'avg_pos',
    'avg_neg',
    'comment_sample_size',
    'no_comments',
    'view_like_ratio',
    'age'
]

# TIER 3: Extended Features - Tree-Based Models
# Use case: Comprehensive models, ensemble methods
# Expected R²: 0.80-0.85
TIER3_TREE = [
    'view_count',
    'likes',
    'comment_count',
    'avg_pos',
    'avg_neu',
    'avg_neg',
    'avg_compound',
    'comment_sample_size',
    'no_comments',
    'view_like_ratio',
    'age',
    'desc_lang'
]

# TIER 3: Extended Features - Linear Models
# Use case: Comprehensive linear models
# Expected R²: 0.80-0.85
TIER3_LINEAR = [
    'log_view_count',
    'log_likes',
    'log_comment_count',
    'avg_pos',
    'avg_neu',
    'avg_neg',
    'avg_compound',
    'comment_sample_size',
    'no_comments',
    'view_like_ratio',
    'age',
    'desc_lang'
]

# All available modeling features (no leakage)
ALL_MODELING_FEATURES = [
    'view_count',
    'likes',
    'comment_count',
    'log_view_count',
    'log_likes',
    'log_comment_count',
    'avg_pos',
    'avg_neu',
    'avg_neg',
    'avg_compound',
    'age',
    'no_comments',
    'comment_sample_size',
    'view_like_ratio',
    'desc_lang'
]

# Features with DATA LEAKAGE (DO NOT USE for training)
LEAKAGE_FEATURES = [
    'log_dislikes',
    'like_dislike_score',
    'view_dislike_ratio',
    'dislike_like_ratio',
    'engagement_rate',
    'log_view_dislike_ratio',
    'log_dislike_like_ratio',
    'log_like_dislike_score',
    'log_engagement_rate'
]

# Feature type mapping
NUMERIC_FEATURES = [
    'view_count', 'likes', 'comment_count',
    'log_view_count', 'log_likes', 'log_comment_count',
    'avg_pos', 'avg_neu', 'avg_neg', 'avg_compound',
    'age', 'comment_sample_size', 'view_like_ratio'
]

BINARY_FEATURES = [
    'no_comments'
]

CATEGORICAL_FEATURES = [
    'desc_lang'
]

# Quick reference dictionary
FEATURE_SETS = {
    'tier1': TIER1_FEATURES,
    'tier2_tree': TIER2_TREE,
    'tier2_linear': TIER2_LINEAR,
    'tier3_tree': TIER3_TREE,
    'tier3_linear': TIER3_LINEAR,
    'all': ALL_MODELING_FEATURES
}

def get_features(tier='tier2_tree'):
    """
    Get feature list by tier name.
    
    Args:
        tier (str): One of 'tier1', 'tier2_tree', 'tier2_linear', 
                    'tier3_tree', 'tier3_linear', 'all'
    
    Returns:
        list: Feature names
    
    Example:
        >>> features = get_features('tier2_tree')
        >>> X = df[features]
    """
    if tier not in FEATURE_SETS:
        raise ValueError(f"Unknown tier: {tier}. Choose from {list(FEATURE_SETS.keys())}")
    return FEATURE_SETS[tier]

def validate_features(features):
    """
    Check if any features have data leakage issues.
    
    Args:
        features (list): List of feature names to validate
    
    Returns:
        tuple: (is_valid, leakage_features_found)
    
    Example:
        >>> is_valid, leakage = validate_features(['view_count', 'likes'])
        >>> print(is_valid)  # True
    """
    leakage_found = [f for f in features if f in LEAKAGE_FEATURES]
    is_valid = len(leakage_found) == 0
    return is_valid, leakage_found
'''

config_path = feature_sets_dir / 'feature_sets_config.py'
with open(config_path, 'w') as f:
    f.write(config_content)

print(f"Saved Python config: {config_path}")
print()

# Create usage example
print("=" * 80)
print("USAGE EXAMPLE")
print("=" * 80)
print()

example_code = """
# Example: Load filtered dataset and train a model

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Import feature configuration
import sys
sys.path.append('feature_engineering/feature_sets')
from feature_sets_config import TIER2_TREE, TARGET

# Load filtered dataset
df = pd.read_csv('yt_dataset_filtered.csv')

# Select features and target
X = df[TIER2_TREE]
y = df[TARGET]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Handle categorical features if needed
if 'desc_lang' in X_train.columns:
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    X_train['desc_lang'] = le.fit_transform(X_train['desc_lang'])
    X_test['desc_lang'] = le.transform(X_test['desc_lang'])

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:,.2f}")
print(f"R² Score: {r2:.4f}")
"""

example_path = feature_sets_dir / 'usage_example.py'
with open(example_path, 'w') as f:
    f.write(example_code)

print(f"Saved usage example: {example_path}")
print()

# Summary
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print()
print("Files created:")
print(f"  1. yt_dataset_filtered.csv - Main filtered dataset ({df_filtered.shape[0]} rows, {df_filtered.shape[1]} cols)")
print(f"  2. feature_engineering/feature_sets/tier1_essential.txt")
print(f"  3. feature_engineering/feature_sets/tier2_tree_based.txt")
print(f"  4. feature_engineering/feature_sets/tier2_linear.txt")
print(f"  5. feature_engineering/feature_sets/tier3_tree_based.txt")
print(f"  6. feature_engineering/feature_sets/tier3_linear.txt")
print(f"  7. feature_engineering/feature_sets/feature_sets_config.py")
print(f"  8. feature_engineering/feature_sets/usage_example.py")
print()
print("Next steps:")
print("  1. Use yt_dataset_filtered.csv for all model training")
print("  2. Import feature_sets_config.py for consistent feature selection")
print("  3. Start with TIER2_TREE for tree-based models (recommended)")
print("  4. See usage_example.py for complete training example")
print()
print("DONE!")
