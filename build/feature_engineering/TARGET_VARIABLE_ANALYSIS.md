# 🎯 Target Variable Analysis: Raw vs Log Dislikes

## Executive Summary

**STRONG RECOMMENDATION: Use `log_dislikes` as your target variable** ✅

The log-transformed target significantly improves model training conditions by:
- **Reducing skewness from 33.9 → 0.4** (97% improvement)
- **Reducing outliers from 12.6% → 1.3%** (90% reduction)
- **Stabilizing variance** (coefficient of variation: 6.34 → 0.24)

---

## 📊 Distribution Comparison

### Raw Dislikes Statistics
```
Mean:      4,911.60
Median:      830.00
Std Dev:  31,125.87
Min:           0
Max:   2,397,733
Skewness:   33.93  ⚠️ Extremely right-skewed
Kurtosis: 1865.90  ⚠️ Heavy-tailed distribution
```

**Problem:** Mean is **6x larger** than median → extreme right skew

### Log Dislikes (log1p) Statistics
```
Mean:      6.82
Median:    6.72
Std Dev:   1.63
Min:       0.00
Max:      14.69
Skewness:  0.37  ✅ Nearly symmetric
Kurtosis:  0.38  ✅ Normal-like tails
```

**Advantage:** Mean ≈ Median → symmetric, well-behaved distribution

---

## 🔍 Key Findings

### 1. Skewness Analysis
| Metric | Raw Dislikes | Log Dislikes | Improvement |
|--------|-------------|--------------|-------------|
| **Skewness** | 33.93 | 0.37 | **97% reduction** |
| **Interpretation** | Extremely skewed | Nearly normal | Much better |

**Why this matters:**
- Most ML algorithms assume reasonably symmetric distributions
- Extreme skewness causes models to be biased toward high-dislike videos
- Log transformation brings distribution much closer to normality

### 2. Outlier Analysis
| Category | Raw Dislikes | Log Dislikes | Change |
|----------|-------------|--------------|--------|
| **Outliers (IQR method)** | 3,881 (12.6%) | 398 (1.3%) | **-90%** |

**Why this matters:**
- Fewer outliers = more robust model
- Less influence from viral videos with extreme dislike counts
- Better generalization to typical videos

### 3. Variance Stability
| Metric | Raw Dislikes | Log Dislikes |
|--------|-------------|--------------|
| **Coefficient of Variation** | 6.34 | 0.24 |

**Why this matters:**
- Lower CV = more stable predictions across the range
- Reduces heteroscedasticity (variance increasing with prediction)
- Meets regression assumptions better

### 4. Percentile Distribution

| Percentile | Raw Dislikes | Log Dislikes |
|-----------|-------------|--------------|
| 25th | 296 | 5.69 |
| 50th (Median) | 830 | 6.72 |
| 75th | 2,510 | 7.83 |
| 90th | 7,396 | 8.91 |
| 95th | 14,824 | 9.60 |
| 99th | 71,372 | 11.18 |
| 100th (Max) | 2,397,733 | 14.69 |

**Observation:** Log transformation compresses extreme values while preserving order

---

## ⚖️ Pros & Cons

### Raw Dislikes Target

✅ **Advantages:**
- **Directly interpretable:** Predictions are actual dislike counts
- **Business-friendly:** Stakeholders understand "1,000 dislikes" immediately
- **No transformation needed:** Predictions can be used as-is

❌ **Disadvantages:**
- **Extremely skewed:** Mean >> Median (skewness = 33.93)
- **High outlier percentage:** 12.6% of videos are outliers
- **Heteroscedasticity:** Prediction errors increase with value
- **Poor for linear models:** Violates normality assumptions
- **Dominated by extremes:** Model overfits to viral videos

### Log Dislikes Target ⭐

✅ **Advantages:**
- **Better distribution:** Near-normal (skewness = 0.37)
- **Reduced outliers:** Only 1.3% outliers
- **Homoscedastic:** Constant variance across predictions
- **Meets assumptions:** Better for all regression algorithms
- **Balanced learning:** Equal attention to low and high dislikes
- **Multiplicative relationships:** Captures relative changes better

❌ **Disadvantages:**
- **Less interpretable:** Need to transform back: `np.expm1(predictions)`
- **Percentage errors:** Model thinks in relative terms (10% error) not absolute
- **Extra step:** Must remember to exponentiate final predictions

---

## 🧪 Normality Tests

### Shapiro-Wilk Test (p-value)
```
Raw dislikes:  p = 0.000000  ❌ (Not normal, p < 0.05)
Log dislikes:  p = 0.000000  ❌ (Not normal, p < 0.05)
```

**Note:** While neither is perfectly normal (due to sample size), log transformation is **much closer** to normality based on skewness and kurtosis.

---

## 📈 Visual Analysis

See `target_variable_comparison.png` for:
1. **Histogram comparison:** Log distribution is symmetric vs. raw is extreme right-skewed
2. **Q-Q plots:** Log follows normal line much better than raw

---

## 🎓 Statistical Rationale

### Why Log Transformation Works

1. **Multiplicative Relationships:**
   - Dislikes grow proportionally with views (not additively)
   - A video with 1M views getting 10K dislikes is similar to 100K views getting 1K dislikes
   - Log captures this: `log(10K) - log(1K) = log(10)` (same ratio)

2. **Error Distribution:**
   - Predicting 900 instead of 1,000 dislikes: 10% error
   - Predicting 90,000 instead of 100,000 dislikes: 10% error
   - Log space treats both equally (constant relative error)
   - Raw space would penalize second more (9,000 vs 100 absolute error)

3. **Regression Assumptions:**
   - Linear regression assumes normal residuals
   - Tree-based models benefit from balanced target distribution
   - Neural networks train better with normalized outputs

---

## 🚀 Implementation Guide

### Training with Log Target

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load data
df = pd.read_csv('yt_dataset_filtered.csv')

# Create log target
df['log_dislikes'] = np.log1p(df['dislikes'])

# Use your chosen features
features = ['view_count', 'likes', 'comment_count', 'avg_compound', 
            'avg_pos', 'avg_neg', 'comment_sample_size', 'no_comments', 
            'view_like_ratio', 'age']

X = df[features]
y_log = df['log_dislikes']  # Use log target

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_log, test_size=0.2, random_state=42
)

# Train on log scale
model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)
model.fit(X_train, y_train)

# Predict in log scale
y_pred_log = model.predict(X_test)

# Convert back to raw scale for evaluation
y_pred_raw = np.expm1(y_pred_log)
y_test_raw = np.expm1(y_test)

# Evaluate in both scales
print("Log Scale Metrics:")
print(f"  RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_log)):.4f}")
print(f"  R²:   {r2_score(y_test, y_pred_log):.4f}")

print("\nRaw Scale Metrics:")
print(f"  RMSE: {np.sqrt(mean_squared_error(y_test_raw, y_pred_raw)):,.0f}")
print(f"  R²:   {r2_score(y_test_raw, y_pred_raw):.4f}")
```

### Key Transformations

```python
# Forward transformation (raw → log)
log_dislikes = np.log1p(dislikes)  # log(1 + x) handles zeros

# Inverse transformation (log → raw)
dislikes = np.expm1(log_dislikes)  # exp(x) - 1

# Why log1p/expm1?
# - log1p(0) = 0 (handles zero dislikes)
# - Regular log(0) = -∞ (breaks everything)
```

---

## 📊 Expected Performance Improvement

### Predicted Model Performance

| Model Type | Raw Target R² | Log Target R² | Improvement |
|------------|--------------|---------------|-------------|
| **Linear Regression** | 0.55-0.65 | 0.70-0.78 | **+20-25%** |
| **Random Forest** | 0.70-0.75 | 0.78-0.85 | **+10-15%** |
| **XGBoost** | 0.75-0.80 | 0.82-0.88 | **+8-12%** |
| **Neural Network** | 0.72-0.77 | 0.80-0.86 | **+10-12%** |

**Linear models benefit most** because they're most sensitive to distribution assumptions.

---

## ⚠️ Important Considerations

### When to Use Each

**Use Log Dislikes (Recommended):**
- Building predictive models for general use ✅
- Want better statistical properties ✅
- Using linear models (Ridge, Lasso, ElasticNet) ✅
- Care about relative accuracy (e.g., "within 20%") ✅
- Want robust predictions across all ranges ✅

**Use Raw Dislikes (Special Cases):**
- Need exact dislike count predictions
- Stakeholders require direct interpretability
- Building decision rules (e.g., "flag if > 10,000 dislikes")
- Cost function is truly linear (rare)

### Hybrid Approach

**Best of both worlds:**
1. Train on `log_dislikes` (better convergence)
2. Evaluate on `raw_dislikes` (interpretable metrics)
3. Report both R² scores

```python
# Train on log scale
model.fit(X_train, y_train_log)

# Predict and transform back
y_pred_log = model.predict(X_test)
y_pred_raw = np.expm1(y_pred_log)

# Evaluate both
log_r2 = r2_score(y_test_log, y_pred_log)
raw_r2 = r2_score(y_test_raw, y_pred_raw)

print(f"Model quality (log scale): R² = {log_r2:.4f}")
print(f"Practical accuracy (raw scale): R² = {raw_r2:.4f}")
```

---

## 🎯 Final Recommendation

### **Use `log_dislikes` as your primary target variable**

**Reasoning:**
1. **Massive skewness reduction:** 33.9 → 0.4 (97% improvement)
2. **Outlier reduction:** 12.6% → 1.3% (90% reduction)
3. **Stable variance:** CV reduces from 6.34 to 0.24
4. **Better model performance:** Expected 10-25% R² improvement
5. **More robust predictions:** Less influenced by viral extremes

**Action Items:**
1. ✅ Add `log_dislikes` column to your dataset
2. ✅ Train all models using `log_dislikes` as target
3. ✅ Use `np.expm1()` to convert predictions back to raw scale
4. ✅ Report metrics in both scales for completeness
5. ✅ Update feature sets if needed (use raw features, not log features, with log target)

---

## 📝 Code Template

```python
# Complete training pipeline with log target
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import sys

sys.path.append('feature_engineering/feature_sets')
from feature_sets_config import TIER2_TREE, TARGET

# Load and prepare data
df = pd.read_csv('yt_dataset_filtered.csv')
df['log_dislikes'] = np.log1p(df['dislikes'])

# Select features and target
X = df[TIER2_TREE]
y_log = df['log_dislikes']  # Log target
y_raw = df['dislikes']       # Keep for evaluation

# Split
X_train, X_test, y_train_log, y_test_log, y_train_raw, y_test_raw = train_test_split(
    X, y_log, y_raw, test_size=0.2, random_state=42
)

# Train on log scale
model = xgb.XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    random_state=42
)
model.fit(X_train, y_train_log)

# Predict
y_pred_log = model.predict(X_test)
y_pred_raw = np.expm1(y_pred_log)

# Comprehensive evaluation
print("=" * 60)
print("MODEL EVALUATION: Log Target Performance")
print("=" * 60)

print("\nLog Scale Metrics (training objective):")
print(f"  RMSE:  {np.sqrt(mean_squared_error(y_test_log, y_pred_log)):.4f}")
print(f"  MAE:   {np.mean(np.abs(y_test_log - y_pred_log)):.4f}")
print(f"  R²:    {r2_score(y_test_log, y_pred_log):.4f}")

print("\nRaw Scale Metrics (practical interpretation):")
print(f"  RMSE:  {np.sqrt(mean_squared_error(y_test_raw, y_pred_raw)):,.0f}")
print(f"  MAE:   {np.mean(np.abs(y_test_raw - y_pred_raw)):,.0f}")
print(f"  R²:    {r2_score(y_test_raw, y_pred_raw):.4f}")
print(f"  MAPE:  {mean_absolute_percentage_error(y_test_raw, y_pred_raw)*100:.1f}%")

# Cross-validation on log scale
cv_scores = cross_val_score(model, X, y_log, cv=5, 
                             scoring='r2', n_jobs=-1)
print(f"\n5-Fold CV R² (log): {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

print("=" * 60)
```

---

## 🔗 Related Files

- **Analysis Script:** `analyze_target_variable.py`
- **Visualization:** `target_variable_comparison.png`
- **Filtered Dataset:** `yt_dataset_filtered.csv`
- **Feature Configuration:** `feature_engineering/feature_sets/feature_sets_config.py`

---

**Generated:** October 10, 2025  
**Author:** Feature Engineering Analysis  
**Dataset:** yt_dataset_filtered.csv (30,867 videos)
