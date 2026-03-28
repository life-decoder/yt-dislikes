# Model Comparison Template

This file will help track performance across different models during the selection phase.

## Models Trained

### 1. XGBoost ✅
- **Status:** Complete
- **Validation R² (log):** 0.8369
- **Validation R² (raw):** 0.8206
- **Validation MAE:** 1,793 dislikes
- **Training Time:** ~8 seconds
- **Overfitting:** 5.76% (slight)
- **Strengths:** Fast, interpretable, good baseline
- **Weaknesses:** Slight overfitting, feature imbalance

### 2. Random Forest ⏳
- **Status:** Not trained yet
- **Expected R²:** ~0.82-0.85
- **Expected Training Time:** ~15-30 seconds
- **Notes:** Good for comparison, less prone to overfitting

### 3. LightGBM ⏳
- **Status:** Not trained yet
- **Expected R²:** ~0.83-0.86
- **Expected Training Time:** ~5-10 seconds
- **Notes:** Faster than XGBoost, handles large datasets well

### 4. CatBoost ⏳
- **Status:** Not trained yet
- **Expected R²:** ~0.83-0.85
- **Expected Training Time:** ~10-20 seconds
- **Notes:** Handles categorical features naturally

### 5. Neural Network ⏳
- **Status:** Not trained yet
- **Expected R²:** ~0.80-0.84
- **Expected Training Time:** ~30-60 seconds
- **Notes:** May capture non-linear patterns better

---

## Comparison Metrics

| Model | Val R² (log) | Val R² (raw) | MAE | Within ±1K | Training Time | Overfitting |
|-------|-------------|-------------|-----|------------|---------------|-------------|
| **XGBoost** | 0.8369 | 0.8206 | 1,793 | 75.7% | 8s | 5.76% |
| Random Forest | - | - | - | - | - | - |
| LightGBM | - | - | - | - | - | - |
| CatBoost | - | - | - | - | - | - |
| Neural Net | - | - | - | - | - | - |

---

## Selection Criteria

### Priority 1: Validation Performance
- **Primary:** R² (log scale) - higher is better
- **Secondary:** MAE (raw scale) - lower is better
- **Tertiary:** Within ±1,000 accuracy - higher is better

### Priority 2: Generalization
- Train-Val R² difference < 10% (avoid overfitting)
- Consistent performance across video size categories

### Priority 3: Practical Considerations
- Training time (for retraining)
- Inference speed (for production)
- Model interpretability
- Deployment complexity

---

## Decision Framework

1. **If performance difference < 2%:**
   - Choose simpler/faster model
   - Prioritize interpretability

2. **If one model clearly outperforms (> 5% better R²):**
   - Choose best performer
   - Accept complexity trade-off

3. **If tie:**
   - Consider ensemble approach
   - Stack multiple models

---

## Current Leader: XGBoost

**Why:**
- Strong baseline (83.7% R²)
- Fast training and inference
- Highly interpretable
- Good generalization

**To Beat XGBoost:**
- Need > 0.86 validation R² (> 2% improvement)
- Or significantly better generalization (< 3% overfitting)
- Or much faster training/inference

---

## Next Steps

1. Train Random Forest (similar to XGBoost but more robust)
2. Train LightGBM (faster alternative)
3. Train CatBoost (if categorical features matter)
4. Train Neural Network (if need non-linearity)
5. Compare all models using this template
6. Select winner for test set evaluation

---

*Update this file as you train each model!*
