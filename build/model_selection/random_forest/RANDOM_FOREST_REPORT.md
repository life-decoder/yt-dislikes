# Random Forest Regression — Report

Dataset: `model_selection/yt_dataset_v5.csv`

Target: `log_dislikes` (log of dislikes). `dislikes` column was excluded from features.

## Splits
- Train: 75%
- Validation: 10% (used for model selection / early evaluation)
- Test: 15% (held out, not used here)

## Validation results
### Train metrics
- RMSE: 0.2314
- MAE: 0.1712
- R2: 0.9796

### Validation metrics
- RMSE: 0.6056
- MAE: 0.4577
- R2: 0.8589

## Plots
- target_distribution: `target_distribution_train_val.png`
- pred_vs_actual_train: `pred_vs_actual_train.png`
- pred_vs_actual_val: `pred_vs_actual_val.png`
- residuals_train: `residuals_train.png`
- residuals_val: `residuals_val.png`
- feature_importances: `feature_importances.png`

Model saved to: `rf_model.pkl`
