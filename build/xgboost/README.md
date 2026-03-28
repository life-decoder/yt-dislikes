# XGBoost Model for YouTube Dislikes Prediction

## 📁 Directory Overview

This directory contains the XGBoost model implementation for the YouTube dislikes prediction project, part of the **model selection phase**.

---

## 🎯 Project Status

**Phase:** Model Selection  
**Purpose:** Baseline model training and evaluation for comparison with other algorithms  
**Test Set:** Reserved (not used yet) for final evaluation after model selection

---

## 📊 Quick Results

| Metric | Validation Set |
|--------|---------------|
| **R² (log scale)** | 0.8369 (83.7%) |
| **R² (raw scale)** | 0.8206 (82.1%) |
| **MAE** | 1,793 dislikes |
| **Median Error** | 290 dislikes |
| **Within ±1,000** | 75.7% of predictions |
| **Within ±2,500** | 88.2% of predictions |

---

## 📂 Files in This Directory

### Scripts
- **`train_xgboost_model.py`** - Main training script
  - Implements 75-10-15 train/val/test split
  - Trains XGBoost with Tier 2 Tree-Based features
  - Generates all visualizations and metrics
  - ~8 seconds runtime

- **`view_results.py`** - Quick results summary
  - Displays key metrics
  - Shows prediction accuracy distribution
  - Overfitting analysis

- **`detailed_analysis.py`** - In-depth error analysis
  - Performance by video size category
  - Prediction confidence intervals
  - Cumulative accuracy curves

- **`check_dataset.py`** - Dataset inspection utility

### Model & Data
- **`xgboost_model.json`** - Trained XGBoost model (JSON format, 2.3 MB)
- **`xgboost_predictions.csv`** - All predictions (train + validation, 26,236 rows)
- **`xgboost_metrics.csv`** - Performance metrics summary
- **`xgboost_feature_importance.csv`** - Feature importance rankings

### Visualizations
- **`xgboost_performance_analysis.png`** - Main 6-panel dashboard
  - Training history (learning curves)
  - Feature importance (top 10)
  - Actual vs Predicted (train & validation)
  - Residual distribution
  - Residual plot

- **`xgboost_raw_scale_predictions.png`** - Raw scale predictions
  - Training set (log-log scale)
  - Validation set (log-log scale)

- **`xgboost_error_analysis.png`** - Error analysis (4 panels)
  - Absolute error vs prediction (train & validation)
  - Q-Q plots for normality check

- **`xgboost_detailed_analysis.png`** - Advanced analysis (6 panels)
  - Error distribution by video size
  - Percentage error by size
  - Sample distribution
  - Prediction scatter with confidence
  - Error vs predicted (zoomed)
  - Cumulative accuracy curve

### Documentation
- **`TRAINING_RESULTS.md`** - Comprehensive training report
- **`README.md`** - This file

---

## 🚀 How to Use

### 1. Train the Model
```bash
cd xgboost
python train_xgboost_model.py
```

**Expected Output:**
- Model training with progress updates
- 7 output files generated
- ~8 seconds runtime

### 2. View Quick Summary
```bash
python view_results.py
```

**Shows:**
- Performance metrics table
- Top 5 feature importance
- Sample predictions
- Error statistics
- Accuracy distribution
- Overfitting analysis

### 3. Run Detailed Analysis
```bash
python detailed_analysis.py
```

**Generates:**
- `xgboost_detailed_analysis.png`
- Performance breakdown by video size
- R² scores for each category

---

## 🔧 Model Configuration

### Features Used (10 total)
Based on **Tier 2 Tree-Based** feature set:

1. `view_count` (50.6% importance) ⭐⭐⭐
2. `likes` (30.4% importance) ⭐⭐⭐
3. `comment_count` (5.7% importance) ⭐
4. `view_like_ratio` (4.2% importance)
5. `no_comments` (1.8% importance)
6. `comment_sample_size` (1.8% importance)
7. `avg_compound` (1.6% importance)
8. `avg_neg` (1.5% importance)
9. `avg_pos` (1.3% importance)
10. `age` (1.3% importance)

**Key Insight:** Top 2 features account for 81% of model importance!

### Target Variable
- **`log_dislikes`** (log-transformed)
- Predictions converted back to raw scale for interpretability
- Log transformation improves:
  - Distribution normality (skewness: 33.93 → 0.37)
  - Outlier handling (12.6% → 1.3%)
  - Model convergence and stability

### Hyperparameters
```python
n_estimators = 200
max_depth = 6
learning_rate = 0.1
subsample = 0.8
colsample_bytree = 0.8
objective = 'reg:squarederror'
random_state = 42
```

### Data Split
- **Training:** 23,150 samples (75%)
- **Validation:** 3,086 samples (10%)
- **Test:** 4,631 samples (15%) - **RESERVED**

---

## 📈 Performance Analysis

### Overall Performance
✅ **Strong baseline achieved!**
- Validation R² = 0.837 (explains 83.7% of variance)
- Slight overfitting: 5.76% drop from training
- Good generalization to unseen data

### Performance by Video Size

| Size Category | Samples | Median Error | % Error | R² |
|--------------|---------|--------------|---------|-----|
| **Tiny (0-100)** | 240 | 37 dislikes | 64.8% | 0.33 |
| **Small (100-500)** | 875 | 94 dislikes | 38.0% | 0.27 |
| **Medium (500-1K)** | 572 | 246 dislikes | 35.1% | 0.07 |
| **Large (1K-5K)** | 972 | 696 dislikes | 36.3% | 0.26 |
| **Very Large (5K-10K)** | 198 | 2,177 dislikes | 32.5% | 0.02 |
| **Huge (10K+)** | 228 | 8,458 dislikes | 41.9% | 0.77 |

**Key Findings:**
- ✅ Better performance on larger videos (10K+ dislikes)
- ⚠️ Higher percentage errors on tiny videos (< 100 dislikes)
- ✅ Median errors consistently below 50% for all categories

### Accuracy Distribution
- **62.3%** of predictions within ±500 dislikes
- **75.7%** of predictions within ±1,000 dislikes
- **88.2%** of predictions within ±2,500 dislikes
- **93.4%** of predictions within ±5,000 dislikes

---

## ✅ Strengths

1. **High Predictive Power**
   - R² = 0.837 on validation set
   - Robust performance across different video sizes

2. **Good Generalization**
   - Only 5.76% overfitting (train-val R² difference)
   - Consistent MAE across train/val sets

3. **Interpretable**
   - Clear feature importance hierarchy
   - view_count and likes are dominant predictors

4. **Fast Training & Inference**
   - Training: ~8 seconds
   - Inference: milliseconds per prediction

---

## ⚠️ Limitations

1. **Slight Overfitting**
   - 5.76% drop in R² from training to validation
   - Could benefit from more regularization

2. **Feature Imbalance**
   - Heavy reliance on top 2 features (81% importance)
   - Other features contribute minimally

3. **Variable Performance**
   - Lower R² for medium-sized videos (500-10K)
   - Higher percentage errors on very small videos

4. **Extreme Value Handling**
   - Max error: 233K dislikes (on validation)
   - May struggle with viral videos

---

## 🎯 Recommendations

### For Model Selection Phase
Compare this baseline with:
- **Random Forest** (ensemble comparison)
- **LightGBM** (faster gradient boosting)
- **CatBoost** (handles categoricals better)
- **Neural Networks** (deep learning approach)

### Potential Improvements
1. **Hyperparameter Tuning**
   - Grid search or Bayesian optimization
   - Try deeper trees (max_depth > 6)
   - Experiment with learning rates

2. **Feature Engineering**
   - Interaction features (view_count × likes)
   - Time-based features (day of week, hour)
   - Channel-level features

3. **Regularization**
   - Lower learning_rate (e.g., 0.05)
   - Increase min_child_weight
   - Add L1/L2 penalties

4. **Ensemble Methods**
   - Stack with other models
   - Weighted averaging of predictions

---

## 🔄 Next Steps

### Immediate Actions
1. ✅ Train alternative models (RF, LightGBM, Neural Net)
2. ✅ Compare validation metrics across all models
3. ✅ Analyze prediction errors and patterns
4. ✅ Select best performing model

### After Model Selection
1. ⏳ Evaluate selected model on **test set** (4,631 samples)
2. ⏳ Generate final performance report
3. ⏳ Calculate confidence intervals
4. ⏳ Deploy model for production use

---

## 📚 References

- **Feature Engineering:** `../feature_engineering/`
- **Dataset:** `../yt_dataset_v4.csv`
- **Feature Selection Report:** `../FEATURE_SELECTION_REPORT.md`
- **Target Variable Decision:** `../feature_engineering/TARGET_VARIABLE_DECISION.md`

---

## 🛠️ Dependencies

```
pandas >= 2.0.0
numpy >= 1.24.0
xgboost >= 3.0.0
scikit-learn >= 1.3.0
matplotlib >= 3.7.0
seaborn >= 0.12.0
scipy >= 1.10.0
```

All dependencies are installed in the virtual environment.

---

## 📝 Notes

- Model trained on October 11, 2025
- Random seed: 42 (for reproducibility)
- All predictions include both log and raw scale values
- Test set remains **completely untouched** until final evaluation

---

## 🏆 Conclusion

**XGBoost achieves excellent baseline performance** with minimal hyperparameter tuning:
- **83.7% R²** on validation (log scale)
- **82.1% R²** on validation (raw scale)
- **75.7%** of predictions within ±1,000 dislikes

This model serves as a strong benchmark for comparing other algorithms during the model selection phase. The test set remains reserved for unbiased final evaluation.

---

*For questions or issues, refer to the main project documentation.*
