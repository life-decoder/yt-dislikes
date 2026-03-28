# XGBoost Model Training Results

## Model Selection Phase - YouTube Dislikes Prediction

**Date:** October 11, 2025  
**Model:** XGBoost Regressor  
**Purpose:** Model selection phase (test set reserved for final evaluation)

---

## 📊 Dataset Split

| Dataset    | Samples | Percentage |
|------------|---------|------------|
| **Training**   | 23,150  | 75.0%      |
| **Validation** | 3,086   | 10.0%      |
| **Test**       | 4,631   | 15.0%      |
| **Total**      | 30,867  | 100.0%     |

⚠️ **Note:** Test set was NOT used during training and is reserved for final model evaluation.

---

## 🎯 Target Variable

**Selected:** `log_dislikes` (log-transformed dislikes)

**Rationale:**
- Recommended by feature engineering analysis
- Reduces skewness from 33.93 to 0.37 (-97% improvement)
- Reduces outliers from 12.6% to 1.3% (-90% improvement)
- Better model convergence and stability
- More stable variance across ranges (homoscedasticity)

---

## 🔧 Features Used

**Feature Set:** Tier 2 Tree-Based (10 features)

Optimized for tree-based models like XGBoost, selected based on:
- Random Forest feature importance
- Correlation analysis
- VIF (multicollinearity) testing
- Data leakage prevention

### Feature List (by importance):

| Rank | Feature              | Importance | Description                    |
|------|---------------------|------------|--------------------------------|
| 1    | view_count          | 0.5055     | Video view count               |
| 2    | likes               | 0.3036     | Number of likes                |
| 3    | comment_count       | 0.0568     | Number of comments             |
| 4    | view_like_ratio     | 0.0423     | Ratio of views to likes        |
| 5    | no_comments         | 0.0176     | Binary: has comments or not    |
| 6    | comment_sample_size | 0.0175     | Sample size of comments        |
| 7    | avg_compound        | 0.0159     | Average compound sentiment     |
| 8    | avg_neg             | 0.0146     | Average negative sentiment     |
| 9    | avg_pos             | 0.0132     | Average positive sentiment     |
| 10   | age                 | 0.0129     | Video age in days              |

**Key Insight:** Top 2 features (view_count and likes) account for ~80% of total importance.

---

## ⚙️ Model Hyperparameters

```python
XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    objective='reg:squarederror',
    n_jobs=-1
)
```

---

## 📈 Performance Metrics

### Log Scale Performance (Training on log-transformed target)

| Metric | Training Set | Validation Set | Difference |
|--------|--------------|----------------|------------|
| **RMSE** | 0.5285     | 0.6485         | +0.1200    |
| **MAE**  | 0.4028     | 0.4890         | +0.0862    |
| **R²**   | 0.8944     | 0.8369         | -0.0576    |

### Raw Scale Performance (Predictions converted back)

| Metric | Training Set | Validation Set |
|--------|--------------|----------------|
| **RMSE** | 15,202 dislikes | 8,092 dislikes |
| **MAE**  | 1,803 dislikes  | 1,793 dislikes |
| **R²**   | 0.7783         | 0.8206         |

---

## 🎓 Model Analysis

### ✅ Strengths

1. **Strong Validation R²: 0.8369**
   - Model explains ~84% of variance in log-transformed dislikes
   - High predictive power on unseen validation data

2. **Good Generalization**
   - R² difference between train/val: 0.0576 (5.76%)
   - Only slight overfitting detected
   - Model performs well on unseen data

3. **Feature Importance Clarity**
   - Clear hierarchy: view_count (50.5%) > likes (30.4%)
   - Interpretable model with engagement metrics as key drivers

4. **Consistent MAE**
   - Similar MAE on training (1,803) and validation (1,793)
   - Suggests stable predictions across dataset splits

### ⚠️ Areas for Improvement

1. **Slight Overfitting**
   - 5.76% drop in R² from training to validation
   - Could benefit from:
     - Increased regularization (lower `learning_rate`)
     - More aggressive subsampling
     - Higher `min_child_weight`

2. **Residual Distribution**
   - Check Q-Q plots in error analysis for normality
   - May have heteroscedasticity in predictions

3. **Feature Diversity**
   - Model heavily relies on 2 features (80% importance)
   - Could explore feature engineering for more balanced importance

---

## 📊 Generated Visualizations

### 1. **xgboost_performance_analysis.png**
6-panel comprehensive dashboard:
- Training history (learning curves)
- Feature importance (top 10)
- Actual vs Predicted (Training - Log Scale)
- Actual vs Predicted (Validation - Log Scale)
- Residuals distribution (Validation)
- Residual plot (Validation)

### 2. **xgboost_raw_scale_predictions.png**
- Raw scale predictions (Training)
- Raw scale predictions (Validation)
- Both with log-log scale for better visualization

### 3. **xgboost_error_analysis.png**
4-panel error analysis:
- Absolute error vs prediction (Training)
- Absolute error vs prediction (Validation)
- Q-Q plot for residuals (Training)
- Q-Q plot for residuals (Validation)

---

## 💾 Output Files

All files saved in `xgboost/` directory:

1. **xgboost_model.json** - Trained XGBoost model (JSON format)
2. **xgboost_metrics.csv** - Performance metrics summary
3. **xgboost_feature_importance.csv** - Feature importance rankings
4. **xgboost_predictions.csv** - All predictions (train + val)
5. **xgboost_performance_analysis.png** - Main visualization dashboard
6. **xgboost_raw_scale_predictions.png** - Raw scale predictions
7. **xgboost_error_analysis.png** - Detailed error analysis

---

## 🎯 Recommendations for Model Selection

### This Model is Suitable If:
- ✅ You need strong baseline performance (R² > 0.83)
- ✅ You value interpretability (clear feature importance)
- ✅ You want fast inference (tree-based model)
- ✅ You need predictions on raw scale with good accuracy

### Consider Alternative Models If:
- ⚠️ You need even better generalization (try ensemble methods)
- ⚠️ You want to reduce overfitting further (try regularization)
- ⚠️ You need predictions for extreme values (may need additional features)

---

## 🔄 Next Steps

### For Model Selection Phase:
1. **Train alternative models** (Random Forest, LightGBM, CatBoost, Neural Networks)
2. **Compare validation performance** across all models
3. **Analyze prediction patterns** for different video types
4. **Select best model** based on validation metrics

### For Final Evaluation:
1. **Only after model selection is complete**
2. **Use test set (4,631 samples)** for final performance assessment
3. **Report test metrics** as unbiased estimate of real-world performance
4. **Generate final prediction analysis** and confidence intervals

---

## 📝 Notes

- Model training took ~8 seconds
- No missing values in final dataset (handled during preprocessing)
- Random seed: 42 (for reproducibility)
- All predictions include both log and raw scale values
- MAPE metric had numerical issues (likely due to very small actual values)

---

## 🏆 Conclusion

**Strong baseline model achieved!** XGBoost with Tier 2 Tree-Based features delivers:
- **83.7% R²** on validation set (log scale)
- **82.1% R²** on validation set (raw scale)
- Minimal overfitting (5.76% drop)
- Clear interpretability through feature importance

This model serves as a solid benchmark for comparing other model architectures during the model selection phase.
