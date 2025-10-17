"""
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
