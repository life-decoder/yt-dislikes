# Feature Engineering for YouTube Dislikes Prediction

This directory contains comprehensive feature selection analysis for predicting YouTube video dislikes.

## 📁 Directory Contents

### Analysis Scripts
- **`feature_selection_analysis.py`** - Main analysis script implementing:
  - Data leakage detection
  - Correlation analysis
  - Multicollinearity (VIF) analysis
  - Principal Component Analysis (PCA)
  - Random Forest feature importance
  - Combined ranking methodology

### Output Files

#### Data Files
- **`feature_ranking.csv`** - Complete ranking of all 15 available features by combined score
  - Includes correlation rank and Random Forest importance rank
  - Scored using weighted combination methodology

- **`recommended_features.txt`** - Three-tier feature recommendations:
  - **Tier 1:** Top 5 essential features (minimal model)
  - **Tier 2:** Top 10 core features (recommended)
  - **Tier 3:** Top 14 extended features (comprehensive)

#### Visualizations
- **`feature_selection_analysis.png`** - 6-panel dashboard showing:
  1. Distribution of raw dislikes
  2. Distribution of log-transformed dislikes
  3. Top 10 features by correlation
  4. Top 10 features by Random Forest importance
  5. PCA cumulative explained variance
  6. Top 10 features by combined ranking

- **`correlation_heatmap.png`** - Correlation matrix of top 15 features + target variable
  - Color-coded heatmap with correlation coefficients
  - Useful for identifying multicollinearity patterns

#### Logs
- **`analysis_output.txt`** - Complete console output from analysis run
  - Detailed statistics for each analysis step
  - VIF values for all features
  - PCA component breakdowns

## 🎯 Key Findings

### Target Variable
**Recommended:** `dislikes` (raw values)
- Alternative: log-transform during model training if needed
- Raw values provide more interpretable predictions

### Data Leakage Features (EXCLUDED)
10 features were identified and excluded to prevent data leakage:
1. `dislikes` (target)
2. `log_dislikes` (target)
3. `like_dislike_score` (contains dislikes)
4. `view_dislike_ratio` (contains dislikes)
5. `dislike_like_ratio` (contains dislikes)
6. `engagement_rate` (contains dislikes)
7. `log_view_dislike_ratio` (contains dislikes)
8. `log_dislike_like_ratio` (contains dislikes)
9. `log_like_dislike_score` (contains dislikes)
10. `log_engagement_rate` (contains dislikes)

### Recommended Features

#### Tier 1 - Essential (Top 5)
```python
essential_features = [
    'view_count',
    'log_view_count',
    'likes',
    'comment_count',
    'log_likes'
]
```

**Important:** Use either raw OR log versions, not both (to avoid multicollinearity)

**For Tree-Based Models (XGBoost, Random Forest):**
```python
features = ['view_count', 'likes', 'comment_count']
```

**For Linear Models:**
```python
features = ['log_view_count', 'log_likes', 'log_comment_count']
```

#### Tier 2 - Core (Top 10) - RECOMMENDED
```python
core_features = [
    'view_count',        # or 'log_view_count'
    'likes',             # or 'log_likes'
    'comment_count',     # or 'log_comment_count'
    'avg_pos',           # Positive sentiment
    'avg_neg',           # Negative sentiment
    'avg_compound',      # Overall sentiment
    'avg_neu',           # Neutral sentiment
    'comment_sample_size',
    'no_comments',
    'view_like_ratio'
]
```

### Correlation Analysis
Top correlated features with dislikes:
1. **view_count:** 0.69 (strong positive)
2. **likes:** 0.67 (strong positive)
3. **comment_count:** 0.42 (moderate positive)
4. **log_view_count:** 0.31 (moderate positive)

### Multicollinearity (VIF) Issues
- **Severe:** Log features (VIF > 200)
- **High:** Sentiment features (VIF > 25)
- **Low:** Raw engagement metrics (VIF < 5)

**Solution:** Choose one representation (raw OR log), not both

### PCA Results
- **7 components** explain 90% of variance
- **8 components** explain 95% of variance
- Confirms significant redundancy among 15 features
- Optimal subset: 5-10 features

## 🚀 Quick Start

### Run the Analysis
```bash
python feature_engineering/feature_selection_analysis.py
```

### Load Recommended Features
```python
import pandas as pd

# Load feature rankings
rankings = pd.read_csv('feature_engineering/feature_ranking.csv')
print(rankings.head(10))

# Use Tier 2 features (recommended)
tier2_features = [
    'view_count',
    'likes',
    'comment_count',
    'avg_compound',
    'comment_sample_size',
    'no_comments',
    'view_like_ratio'
]

# Load dataset and select features
df = pd.read_csv('yt_dataset_en_v3.csv')
X = df[tier2_features]
y = df['dislikes']
```

## 📊 Expected Model Performance

| Model Type | Features | Expected R² | Expected RMSE |
|-----------|----------|-------------|---------------|
| Baseline | Top 3 | 0.70-0.75 | ~15,000 |
| Recommended | Tier 2 (10) | 0.75-0.82 | ~13,000 |
| Comprehensive | Tier 3 (14) | 0.80-0.85 | ~12,000 |

## ⚠️ Critical Reminders

### DO NOT USE:
- Features with "dislike" in the name
- `engagement_rate` (includes dislikes)
- `like_dislike_score` (includes dislikes)
- Both raw AND log versions of the same feature

### DO USE:
- Either raw features (`view_count`, `likes`, `comment_count`)
- OR log features (`log_view_count`, `log_likes`, `log_comment_count`)
- Sentiment features (`avg_*`)
- Metadata (`age`, `no_comments`)

## 🔄 Next Steps

1. **Train Baseline Model**
   ```python
   from sklearn.ensemble import RandomForestRegressor
   
   model = RandomForestRegressor(n_estimators=100, random_state=42)
   model.fit(X_train[['view_count', 'likes', 'comment_count']], y_train)
   ```

2. **Train Enhanced Model**
   ```python
   import xgboost as xgb
   
   model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6)
   model.fit(X_train[tier2_features], y_train)
   ```

3. **Evaluate Performance**
   ```python
   from sklearn.metrics import mean_squared_error, r2_score
   
   y_pred = model.predict(X_test)
   rmse = np.sqrt(mean_squared_error(y_test, y_pred))
   r2 = r2_score(y_test, y_pred)
   
   print(f"RMSE: {rmse:.2f}")
   print(f"R²: {r2:.4f}")
   ```

## 📖 Additional Resources

- **Detailed Report:** `../FEATURE_SELECTION_REPORT.md`
- **Dataset Documentation:** `../DATASET_FIELDS_DOCUMENTATION.md`
- **Model Training:** `../xgboost/train_model.py` (example)

## 🔧 Dependencies

```
pandas >= 2.3.2
numpy >= 1.26.4
scikit-learn >= 1.7.2
statsmodels >= 0.14.5
matplotlib >= 3.10.7
seaborn >= 0.13.2
```

## 📝 Notes

- Analysis performed on 30,867 English YouTube videos
- Data collected before December 13, 2021 (when YouTube hid public dislikes)
- Random seed: 42 (for reproducibility)
- All visualizations generated automatically by analysis script

---

**Last Updated:** October 10, 2025  
**Analysis Version:** 1.0
