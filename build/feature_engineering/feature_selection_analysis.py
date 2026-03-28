"""
Feature Selection Analysis for YouTube Dislikes Prediction
============================================================

This script performs comprehensive feature selection analysis to identify:
1. The optimal target variable (dislikes vs log_dislikes)
2. Features without data leakage (not derived from dislikes)
3. Most relevant predictive features using multiple methods:
   - Pearson correlation analysis
   - Variance Inflation Factor (VIF) for multicollinearity
   - Principal Component Analysis (PCA)
   - Feature importance from Random Forest

Author: Feature Engineering System
Date: October 10, 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import pearsonr, skew, kurtosis
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Configure plotting
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Configure pandas display
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 100)
pd.set_option('display.max_colwidth', 30)

print("=" * 80)
print("FEATURE SELECTION ANALYSIS FOR YOUTUBE DISLIKES PREDICTION")
print("=" * 80)
print()

# Load dataset
print("Loading dataset...")
df = pd.read_csv('yt_dataset_en_v3.csv')

# Convert object columns to numeric where appropriate
numeric_cols = ['log_likes', 'log_dislikes', 'log_comment_count', 'log_dislike_like_ratio', 
                'log_like_dislike_score', 'log_engagement_rate']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print()

# ============================================================================
# STEP 1: IDENTIFY FEATURES WITH DATA LEAKAGE
# ============================================================================
print("=" * 80)
print("STEP 1: IDENTIFYING FEATURES WITH DATA LEAKAGE")
print("=" * 80)
print()

# Features that are derived from or include dislikes in their calculation
leakage_features = [
    'dislikes',  # Target variable (raw)
    'log_dislikes',  # Target variable (log-transformed)
    'like_dislike_score',  # Uses dislikes: likes / (likes + dislikes)
    'view_dislike_ratio',  # Uses dislikes: view_count / dislikes
    'dislike_like_ratio',  # Uses dislikes: dislikes / likes
    'engagement_rate',  # Uses dislikes: (likes + dislikes) / view_count
    'log_view_dislike_ratio',  # Log transform of view_dislike_ratio
    'log_dislike_like_ratio',  # Log transform of dislike_like_ratio
    'log_like_dislike_score',  # Log transform of like_dislike_score
    'log_engagement_rate',  # Log transform of engagement_rate
]

print("Features with DATA LEAKAGE (derived from dislikes):")
print("-" * 80)
for i, feature in enumerate(leakage_features, 1):
    print(f"{i:2d}. {feature}")
print()
print("WARNING: These features will be EXCLUDED from the predictive model training")
print("         to prevent data leakage and ensure valid predictions.")
print()

# Identify non-identifier columns
identifier_features = ['video_id', 'channel_id', 'published_at']

# Available features for prediction (excluding leakage and identifiers)
all_features = df.columns.tolist()
available_features = [f for f in all_features 
                     if f not in leakage_features 
                     and f not in identifier_features]

print("Available features for prediction (no leakage):")
print("-" * 80)
for i, feature in enumerate(available_features, 1):
    print(f"{i:2d}. {feature}")
print()
print(f"Total available features: {len(available_features)}")
print()

# ============================================================================
# STEP 2: TARGET VARIABLE SELECTION
# ============================================================================
print("=" * 80)
print("STEP 2: TARGET VARIABLE SELECTION")
print("=" * 80)
print()

# Compare dislikes vs log_dislikes distributions
target_comparison = pd.DataFrame({
    'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Skewness', 'Kurtosis'],
    'dislikes': [
        df['dislikes'].mean(),
        df['dislikes'].median(),
        df['dislikes'].std(),
        df['dislikes'].min(),
        df['dislikes'].max(),
        skew(df['dislikes']),
        kurtosis(df['dislikes'])
    ],
    'log_dislikes': [
        df['log_dislikes'].mean(),
        df['log_dislikes'].median(),
        df['log_dislikes'].std(),
        df['log_dislikes'].min(),
        df['log_dislikes'].max(),
        skew(df['log_dislikes']),
        kurtosis(df['log_dislikes'])
    ]
})

print("Target Variable Distribution Comparison:")
print("-" * 80)
print(target_comparison.to_string(index=False))
print()

# Interpretation
print(" Distribution Analysis:")
print("   - Skewness close to 0 indicates normal distribution (ideal for regression)")
print("   - Kurtosis close to 0 indicates normal tail behavior")
print()

if abs(skew(df['log_dislikes'])) < abs(skew(df['dislikes'])):
    print(" RECOMMENDATION: Use 'log_dislikes' as target variable")
    print("   Reasons:")
    print(f"   - Lower skewness: {skew(df['log_dislikes']):.4f} vs {skew(df['dislikes']):.4f}")
    print(f"   - More normal distribution (better for regression models)")
    print(f"   - Compresses extreme outliers effectively")
    target_variable = 'log_dislikes'
else:
    print(" RECOMMENDATION: Use 'dislikes' as target variable")
    target_variable = 'dislikes'

print()

# ============================================================================
# STEP 3: CORRELATION ANALYSIS
# ============================================================================
print("=" * 80)
print("STEP 3: CORRELATION ANALYSIS WITH TARGET VARIABLE")
print("=" * 80)
print()

# Calculate correlations with target variable
correlations = []
for feature in available_features:
    if df[feature].dtype in ['int64', 'float64']:
        corr, p_value = pearsonr(df[feature], df[target_variable])
        correlations.append({
            'Feature': feature,
            'Correlation': corr,
            'Abs_Correlation': abs(corr),
            'P_Value': p_value,
            'Significant': 'Yes' if p_value < 0.05 else 'No'
        })

corr_df = pd.DataFrame(correlations).sort_values('Abs_Correlation', ascending=False)

print("Top 15 Features by Absolute Correlation with log_dislikes:")
print("-" * 80)
print(corr_df.head(15).to_string(index=False))
print()

# Filter highly correlated features (|r| > 0.3)
high_corr_features = corr_df[corr_df['Abs_Correlation'] > 0.3]['Feature'].tolist()
print(f"Features with |correlation| > 0.3: {len(high_corr_features)}")
print()

# ============================================================================
# STEP 4: MULTICOLLINEARITY ANALYSIS (VIF)
# ============================================================================
print("=" * 80)
print("STEP 4: MULTICOLLINEARITY ANALYSIS (VIF)")
print("=" * 80)
print()

# Select numeric features for VIF calculation
numeric_features = [f for f in available_features if df[f].dtype in ['int64', 'float64']]
X_vif = df[numeric_features].fillna(0)

# Calculate VIF for each feature
vif_data = []
for i, feature in enumerate(numeric_features):
    try:
        vif = variance_inflation_factor(X_vif.values, i)
        vif_data.append({'Feature': feature, 'VIF': vif})
    except:
        vif_data.append({'Feature': feature, 'VIF': np.nan})

vif_df = pd.DataFrame(vif_data).sort_values('VIF', ascending=False)

print("Variance Inflation Factor (VIF) Analysis:")
print("-" * 80)
print("VIF Interpretation:")
print("  - VIF < 5: Low multicollinearity (good)")
print("  - VIF 5-10: Moderate multicollinearity (acceptable)")
print("  - VIF > 10: High multicollinearity (problematic)")
print()
print(vif_df.head(20).to_string(index=False))
print()

# Identify features with acceptable VIF
acceptable_vif_features = vif_df[vif_df['VIF'] < 10]['Feature'].tolist()
print(f"Features with VIF < 10 (low to moderate multicollinearity): {len(acceptable_vif_features)}")
print()

# ============================================================================
# STEP 5: PRINCIPAL COMPONENT ANALYSIS (PCA)
# ============================================================================
print("=" * 80)
print("STEP 5: PRINCIPAL COMPONENT ANALYSIS (PCA)")
print("=" * 80)
print()

# Prepare data for PCA
X_pca = df[numeric_features].fillna(0)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_pca)

# Perform PCA
pca = PCA()
pca.fit(X_scaled)

# Calculate cumulative explained variance
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

# Find number of components for 95% variance
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
n_components_90 = np.argmax(cumulative_variance >= 0.90) + 1

print("PCA Results:")
print("-" * 80)
print(f"Total features: {len(numeric_features)}")
print(f"Components for 90% variance: {n_components_90}")
print(f"Components for 95% variance: {n_components_95}")
print()

# Show explained variance by first 10 components
variance_df = pd.DataFrame({
    'Component': [f'PC{i+1}' for i in range(min(10, len(numeric_features)))],
    'Explained_Variance': pca.explained_variance_ratio_[:10],
    'Cumulative_Variance': cumulative_variance[:10]
})

print("First 10 Principal Components:")
print(variance_df.to_string(index=False))
print()

# Get feature loadings for first 3 components
loadings = pd.DataFrame(
    pca.components_[:3].T,
    columns=['PC1', 'PC2', 'PC3'],
    index=numeric_features
)

# Find features with highest loadings
print("Top 10 Features by Loading on PC1 (most important component):")
print("-" * 80)
pc1_loadings = loadings['PC1'].abs().sort_values(ascending=False).head(10)
print(pc1_loadings.to_string())
print()

# ============================================================================
# STEP 6: RANDOM FOREST FEATURE IMPORTANCE
# ============================================================================
print("=" * 80)
print("STEP 6: RANDOM FOREST FEATURE IMPORTANCE")
print("=" * 80)
print()

# Prepare data for Random Forest
X_rf = df[numeric_features].fillna(0)
y_rf = df[target_variable]

print("Training Random Forest model for feature importance...")
print(f"Features: {len(numeric_features)}")
print(f"Samples: {len(X_rf)}")
print()

# Train Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10)
rf.fit(X_rf, y_rf)

# Get feature importances
feature_importance = pd.DataFrame({
    'Feature': numeric_features,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

print("Top 15 Features by Random Forest Importance:")
print("-" * 80)
print(feature_importance.head(15).to_string(index=False))
print()

# ============================================================================
# STEP 7: FINAL FEATURE SELECTION
# ============================================================================
print("=" * 80)
print("STEP 7: FINAL FEATURE SELECTION")
print("=" * 80)
print()

# Combine insights from all methods
# Weight each method's ranking
def get_rank_score(feature_list, all_features):
    """Assign scores based on ranking (1 = best, decreasing from there)"""
    scores = {}
    for i, feature in enumerate(feature_list):
        scores[feature] = len(feature_list) - i
    return scores

# Method 1: Correlation ranking
corr_scores = get_rank_score(corr_df.head(15)['Feature'].tolist(), numeric_features)

# Method 2: Random Forest importance ranking
rf_scores = get_rank_score(feature_importance.head(15)['Feature'].tolist(), numeric_features)

# Method 3: PCA loadings ranking
pca_scores = get_rank_score(pc1_loadings.index.tolist(), numeric_features)

# Combine scores
combined_scores = {}
for feature in numeric_features:
    score = 0
    score += corr_scores.get(feature, 0) * 1.0  # Correlation weight
    score += rf_scores.get(feature, 0) * 1.2    # RF weight (slightly higher)
    score += pca_scores.get(feature, 0) * 0.8   # PCA weight
    combined_scores[feature] = score

# Sort by combined score
final_ranking = pd.DataFrame([
    {'Feature': k, 'Combined_Score': v} 
    for k, v in combined_scores.items()
]).sort_values('Combined_Score', ascending=False)

# Add individual rankings
final_ranking['Correlation_Rank'] = final_ranking['Feature'].apply(
    lambda x: corr_df[corr_df['Feature'] == x].index[0] + 1 if x in corr_df['Feature'].values else np.nan
)
final_ranking['RF_Importance_Rank'] = final_ranking['Feature'].apply(
    lambda x: feature_importance[feature_importance['Feature'] == x].index[0] + 1 if x in feature_importance['Feature'].values else np.nan
)

print("Top 20 Features - Combined Ranking:")
print("-" * 80)
print(final_ranking.head(20).to_string(index=False))
print()

# Select top features
top_features = final_ranking[final_ranking['Combined_Score'] > 0]['Feature'].tolist()

print(f" SELECTED FEATURES: {len(top_features)} features")
print()

# ============================================================================
# STEP 8: RECOMMENDED FEATURE SETS
# ============================================================================
print("=" * 80)
print("STEP 8: RECOMMENDED FEATURE SETS")
print("=" * 80)
print()

# Tier 1: Top 5 most important features
tier1_features = final_ranking.head(5)['Feature'].tolist()
print("TIER 1 - Essential Features (Top 5):")
print("-" * 80)
for i, feature in enumerate(tier1_features, 1):
    print(f"{i}. {feature}")
print()

# Tier 2: Top 10 features
tier2_features = final_ranking.head(10)['Feature'].tolist()
print("TIER 2 - Core Features (Top 10):")
print("-" * 80)
for i, feature in enumerate(tier2_features, 1):
    print(f"{i}. {feature}")
print()

# Tier 3: Top 15 features
tier3_features = final_ranking.head(15)['Feature'].tolist()
print("TIER 3 - Extended Features (Top 15):")
print("-" * 80)
for i, feature in enumerate(tier3_features, 1):
    print(f"{i}. {feature}")
print()

# ============================================================================
# STEP 9: SAVE RESULTS
# ============================================================================
print("=" * 80)
print("STEP 9: SAVING RESULTS")
print("=" * 80)
print()

# Save rankings to CSV
final_ranking.to_csv('feature_engineering/feature_ranking.csv', index=False)
print(" Feature ranking saved to: feature_engineering/feature_ranking.csv")

# Save recommended feature sets
with open('feature_engineering/recommended_features.txt', 'w') as f:
    f.write("RECOMMENDED FEATURES FOR YOUTUBE DISLIKES PREDICTION\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Target Variable: {target_variable}\n\n")
    f.write("TIER 1 - Essential Features (Top 5):\n")
    f.write("-" * 80 + "\n")
    for i, feature in enumerate(tier1_features, 1):
        f.write(f"{i}. {feature}\n")
    f.write("\n")
    f.write("TIER 2 - Core Features (Top 10):\n")
    f.write("-" * 80 + "\n")
    for i, feature in enumerate(tier2_features, 1):
        f.write(f"{i}. {feature}\n")
    f.write("\n")
    f.write("TIER 3 - Extended Features (Top 15):\n")
    f.write("-" * 80 + "\n")
    for i, feature in enumerate(tier3_features, 1):
        f.write(f"{i}. {feature}\n")

print(" Recommended features saved to: feature_engineering/recommended_features.txt")
print()

# ============================================================================
# STEP 10: GENERATE VISUALIZATIONS
# ============================================================================
print("=" * 80)
print("STEP 10: GENERATING VISUALIZATIONS")
print("=" * 80)
print()

# Create figure with subplots
fig = plt.figure(figsize=(20, 12))

# Plot 1: Target variable distribution comparison
ax1 = plt.subplot(2, 3, 1)
plt.hist(df['dislikes'], bins=50, alpha=0.7, label='dislikes', edgecolor='black')
plt.xlabel('Dislikes')
plt.ylabel('Frequency')
plt.title('Distribution of Raw Dislikes')
plt.legend()

ax2 = plt.subplot(2, 3, 2)
plt.hist(df['log_dislikes'], bins=50, alpha=0.7, color='orange', label='log_dislikes', edgecolor='black')
plt.xlabel('Log Dislikes')
plt.ylabel('Frequency')
plt.title('Distribution of Log-Transformed Dislikes')
plt.legend()

# Plot 3: Top 10 correlations
ax3 = plt.subplot(2, 3, 3)
top_corr = corr_df.head(10)
colors = ['green' if x > 0 else 'red' for x in top_corr['Correlation']]
plt.barh(range(len(top_corr)), top_corr['Correlation'], color=colors)
plt.yticks(range(len(top_corr)), top_corr['Feature'])
plt.xlabel('Pearson Correlation')
plt.title('Top 10 Features by Correlation with log_dislikes')
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
plt.gca().invert_yaxis()

# Plot 4: Top 10 RF importance
ax4 = plt.subplot(2, 3, 4)
top_rf = feature_importance.head(10)
plt.barh(range(len(top_rf)), top_rf['Importance'], color='steelblue')
plt.yticks(range(len(top_rf)), top_rf['Feature'])
plt.xlabel('Importance')
plt.title('Top 10 Features by Random Forest Importance')
plt.gca().invert_yaxis()

# Plot 5: PCA explained variance
ax5 = plt.subplot(2, 3, 5)
plt.plot(range(1, min(21, len(cumulative_variance)+1)), cumulative_variance[:20], marker='o')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% variance')
plt.axhline(y=0.90, color='orange', linestyle='--', label='90% variance')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA Cumulative Explained Variance')
plt.legend()
plt.grid(True)

# Plot 6: Combined ranking
ax6 = plt.subplot(2, 3, 6)
top_combined = final_ranking.head(10)
plt.barh(range(len(top_combined)), top_combined['Combined_Score'], color='purple')
plt.yticks(range(len(top_combined)), top_combined['Feature'])
plt.xlabel('Combined Score')
plt.title('Top 10 Features by Combined Ranking')
plt.gca().invert_yaxis()

plt.tight_layout()
plt.savefig('feature_engineering/feature_selection_analysis.png', dpi=300, bbox_inches='tight')
print(" Visualizations saved to: feature_engineering/feature_selection_analysis.png")
print()

# Create correlation heatmap for top features
plt.figure(figsize=(14, 12))
top_features_heatmap = tier3_features + [target_variable]
corr_matrix = df[top_features_heatmap].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix: Top 15 Features + Target Variable')
plt.tight_layout()
plt.savefig('feature_engineering/correlation_heatmap.png', dpi=300, bbox_inches='tight')
print(" Correlation heatmap saved to: feature_engineering/correlation_heatmap.png")
print()

print("=" * 80)
print("FEATURE SELECTION ANALYSIS COMPLETE")
print("=" * 80)
print()
print("Summary:")
print(f"  - Target Variable: {target_variable}")
print(f"  - Total Available Features: {len(available_features)}")
print(f"  - Recommended Tier 1 Features: {len(tier1_features)}")
print(f"  - Recommended Tier 2 Features: {len(tier2_features)}")
print(f"  - Recommended Tier 3 Features: {len(tier3_features)}")
print()
print("Output Files:")
print("  1. feature_ranking.csv - Complete feature ranking")
print("  2. recommended_features.txt - Recommended feature sets")
print("  3. feature_selection_analysis.png - Visualization dashboard")
print("  4. correlation_heatmap.png - Feature correlation matrix")
print()
print("Next Steps:")
print("  - Use recommended features to train regression models")
print("  - Compare model performance across different feature tiers")
print("  - Generate detailed report with findings")
print()
