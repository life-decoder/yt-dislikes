# XGBoost Regression Model Report

## Data Split
Train: 75%
Validation: 10%
Test: 15%

## Hyperparameter Optimization
Best Parameters: {'subsample': 0.8, 'n_estimators': 200, 'max_depth': 7, 'learning_rate': 0.1, 'colsample_bytree': 1.0}

## Cross-Validation
CV RMSE (mean): 0.3734
CV RMSE (std): 0.0065

## Validation Results
RMSE: 0.6016
R2: 0.8673

## Test Results
RMSE: 0.6216
R2: 0.8548

## Visualizations
- Feature Importance: feature_importance.png
- Actual vs Predicted (Validation): actual_vs_pred_val.png
- Actual vs Predicted (Test): actual_vs_pred_test.png
- Error Distribution (Validation): error_dist_val.png
- Error Distribution (Test): error_dist_test.png
