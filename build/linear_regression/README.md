# Linear Regression Model for YouTube Dislikes Prediction

## Overview

This directory contains a **baseline Linear Regression model** for predicting YouTube video dislikes. This model serves as a simple, interpretable benchmark in the model selection phase.

---

## 📁 Directory Contents

### Scripts
- **`train_linear_regression_model.py`** - Main training pipeline
- **`view_results.py`** - Quick results viewer
- **`detailed_analysis.py`** - In-depth error analysis
- **`check_dataset.py`** - Dataset inspector (optional)

### Outputs (Generated after training)
- **`linear_regression_coefficients.csv`** - Feature coefficients
- **`linear_regression_scaler_params.csv`** - StandardScaler parameters
- **`linear_regression_predictions.csv`** - All predictions (train + val)
- **`linear_regression_metrics.csv`** - Performance metrics

### Visualizations (Generated after training)
- **`linear_regression_performance_analysis.png`** - 6-panel main dashboard
- **`linear_regression_raw_scale_predictions.png`** - Raw scale predictions
- **`linear_regression_detailed_analysis.png`** - Advanced diagnostics

---

## 🚀 Quick Start

### 1. Train the Model
```bash
cd linear_regression
python train_linear_regression_model.py
```

**Expected Output:**
- Training completes in ~1-2 seconds
- Creates 7 output files
- Displays performance metrics

### 2. View Results
```bash
python view_results.py
```

**Shows:**
- Key performance metrics
- Top feature coefficients
- Overfitting analysis
- Prediction accuracy stats

### 3. Detailed Analysis
```bash
python detailed_analysis.py
```

**Provides:**
- Residual diagnostics
- Error analysis by video size
- Normality tests
- Additional visualizations

---

## 📊 Model Configuration

### Target Variable
- **log_dislikes** (log-transformed)
- Converts back to raw scale for interpretability

### Features (10 features)
```python
FEATURES = [
    'view_count',           # Raw view count
    'likes',                # Raw like count
    'comment_count',        # Number of comments
    'avg_compound',         # Average sentiment compound score
    'avg_pos',              # Average positive sentiment
    'avg_neg',              # Average negative sentiment
    'comment_sample_size',  # Sample size of comments analyzed
    'no_comments',          # Boolean: video has no comments
    'view_like_ratio',      # Ratio of views to likes
    'age'                   # Video age in days
]
```

### Data Split
- **Training:** 75% (used for fitting)
- **Validation:** 10% (used for evaluation)
- **Test:** 15% (reserved, not used yet)

### Preprocessing
- **Feature Scaling:** StandardScaler (mean=0, std=1)
  - Critical for Linear Regression!
- **Missing Values:** Filled with median (or 0 for log features)

### Hyperparameters
```python
LinearRegression(
    fit_intercept=True,
    n_jobs=-1  # Use all CPU cores
)
```

---

## 📈 Expected Performance

Based on typical results with this dataset:

### Validation Metrics (approximate)
- **R² (log scale):** ~0.75-0.80
- **R² (raw scale):** ~0.70-0.75
- **MAE:** ~2,500-3,500 dislikes
- **RMSE:** ~6,000-8,000 dislikes

### Strengths
- ✅ **Fast:** Training in 1-2 seconds
- ✅ **Interpretable:** Clear coefficient values
- ✅ **Simple:** No hyperparameter tuning needed
- ✅ **Baseline:** Good reference for comparison

### Limitations
- ⚠️ **Linear Assumptions:** May miss non-linear relationships
- ⚠️ **Feature Engineering:** Performance depends on feature quality
- ⚠️ **Outlier Sensitivity:** Can be affected by extreme values

---

## 🔍 Key Files Explained

### 1. linear_regression_coefficients.csv
```csv
feature,coefficient,abs_coefficient
view_count,0.8234,0.8234
likes,0.7123,0.7123
comment_count,0.1234,0.1234
...
```
**Interpretation:**
- Coefficient = impact of 1-std increase in feature (after scaling)
- Positive = increases predicted dislikes
- Negative = decreases predicted dislikes

### 2. linear_regression_metrics.csv
```csv
set,rmse_log,mae_log,r2_log,rmse_raw,mae_raw,r2_raw,mape_raw
training,0.4123,0.3234,0.7823,7234.12,3123.45,0.7456,42.34
validation,0.4567,0.3456,0.7512,7891.23,3456.78,0.7234,45.67
```
**Use for:**
- Model comparison
- Overfitting detection
- Performance tracking

### 3. linear_regression_predictions.csv
```csv
set,actual_log,predicted_log,actual_raw,predicted_raw
train,5.234,5.123,187.45,167.89
validation,6.123,6.234,456.78,489.12
...
```
**Use for:**
- Error analysis
- Visualization
- Debugging

---

## 📊 Visualization Guide

### 1. linear_regression_performance_analysis.png
**6-panel dashboard:**
1. **Feature Coefficients** - Bar chart of all coefficients
2. **Predictions vs Actual (Log)** - Validation set scatter
3. **Residuals Distribution** - Histogram of errors
4. **Predictions vs Actual (Raw)** - Validation set scatter
5. **Absolute Error Distribution** - Error magnitude histogram
6. **Performance Comparison** - Train vs Validation metrics

### 2. linear_regression_raw_scale_predictions.png
**2-panel figure:**
- Left: Training set predictions (raw scale)
- Right: Validation set predictions (raw scale)

### 3. linear_regression_detailed_analysis.png
**6-panel advanced diagnostics:**
1. **Residual Plot** - Check homoscedasticity
2. **Q-Q Plot** - Check normality assumption
3. **Error by Video Size** - Performance across video sizes
4. **Cumulative Error Distribution** - Percentile analysis
5. **Top 10 Coefficients** - Most important features
6. **Predictions with Error Coloring** - Error density visualization

---

## 🎯 Model Selection Workflow

### Phase 1: Train Baseline (Current)
```bash
python train_linear_regression_model.py
```

### Phase 2: Compare with Other Models
Train alternative models (XGBoost, Random Forest, etc.) and compare:
- Validation R² (primary metric)
- Training speed
- Overfitting level
- Interpretability

### Phase 3: Select Best Model
Choose based on:
1. **Highest validation R²**
2. **Lowest overfitting**
3. **Meets speed requirements**

### Phase 4: Final Evaluation
Use the test set (15%) for unbiased performance estimate:
```python
# Only after model selection is complete!
y_test_pred = model.predict(X_test_scaled)
test_r2 = r2_score(y_test, y_test_pred)
```

---

## 🔧 Troubleshooting

### Issue: Low R² score
**Possible causes:**
- Feature scaling not applied
- Wrong features selected
- Data quality issues

**Solutions:**
- Check scaler is fitted on train, transformed on val
- Verify feature names match dataset
- Run `check_dataset.py` to inspect data

### Issue: High overfitting (Train R² >> Val R²)
**Possible causes:**
- Model too complex (unlikely for Linear Regression)
- Data leakage
- Small validation set

**Solutions:**
- Check data split is correct (75-10-15)
- Verify no information leakage
- Consider regularization (Ridge/Lasso)

### Issue: FileNotFoundError
**Cause:**
- Training script hasn't been run
- Wrong working directory

**Solution:**
```bash
cd linear_regression
python train_linear_regression_model.py
```

---

## 📊 Comparing with XGBoost

| Aspect | Linear Regression | XGBoost |
|--------|------------------|---------|
| **Training Time** | 1-2 seconds | ~8 seconds |
| **R² (Val)** | ~0.75-0.80 | ~0.83-0.85 |
| **Interpretability** | High (coefficients) | Medium (feature importance) |
| **Overfitting Risk** | Low | Medium |
| **Feature Engineering** | Critical | Less critical |
| **Non-linearity** | No | Yes |

**When to use Linear Regression:**
- Need fast training/inference
- Require high interpretability
- Linear relationships sufficient

**When to use XGBoost:**
- Need best performance
- Complex non-linear patterns
- Have time for tuning

---

## 🧪 Advanced Usage

### Custom Feature Selection
Edit `FEATURES` list in training script:
```python
FEATURES = [
    'view_count',
    'likes',
    # Add or remove features here
]
```

### Add Regularization (Ridge/Lasso)
Replace `LinearRegression` with:
```python
from sklearn.linear_model import Ridge

model = Ridge(
    alpha=1.0,  # Regularization strength
    random_state=RANDOM_SEED
)
```

### Hyperparameter Tuning
For Ridge/Lasso, tune `alpha`:
```python
from sklearn.model_selection import GridSearchCV

param_grid = {'alpha': [0.1, 1.0, 10.0, 100.0]}
grid_search = GridSearchCV(Ridge(), param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train)
best_model = grid_search.best_estimator_
```

---

## 📚 References

### Documentation
- [Scikit-learn Linear Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
- [Feature Scaling Guide](https://scikit-learn.org/stable/modules/preprocessing.html)

### Related Files
- `../FEATURE_SELECTION_REPORT.md` - Feature engineering decisions
- `../xgboost/README.md` - XGBoost model documentation
- `../QUICKSTART_MODEL_TRAINING.md` - General training guide

---

## ✅ Checklist

Before considering model complete:
- [ ] Training script runs without errors
- [ ] Validation R² > 0.70
- [ ] Overfitting < 10% (R² difference)
- [ ] All visualizations generated
- [ ] Results saved to CSV files
- [ ] Test set remains unused

---

## 💡 Tips for Best Results

1. **Always scale features** - Critical for Linear Regression
2. **Check residual plots** - Verify assumptions
3. **Compare with XGBoost** - Baseline vs state-of-the-art
4. **Document findings** - Record why you chose this model (or not)
5. **Keep test set clean** - Don't touch until final evaluation

---

## 🎓 Key Learnings

### What Linear Regression Teaches Us
1. **Feature Importance** - Coefficients show direct impact
2. **Baseline Performance** - Sets minimum acceptable performance
3. **Assumption Checking** - Residuals reveal data properties
4. **Simplicity Value** - Sometimes simple is good enough

### Expected Insights
- `view_count` and `likes` likely have highest coefficients
- Sentiment features may contribute less than engagement metrics
- Log transformation helps with skewed distributions
- Feature scaling is crucial for coefficient interpretation

---

## 📞 Support

For issues or questions:
1. Check visualizations for diagnostic clues
2. Run `detailed_analysis.py` for deeper insights
3. Compare with XGBoost results in `../xgboost/`
4. Review feature engineering docs in `../feature_engineering/`

---

*Last Updated: October 11, 2025*
*Model Version: 1.0*
*Status: Model Selection Phase*
