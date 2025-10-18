import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- Main Script ---
if __name__ == "__main__":
    # 1. Load the Dataset
    try:
        df = pd.read_csv('../yt_dataset_v4.csv')
        print(f"Dataset 'yt_dataset_v4.csv' loaded successfully. Original shape: {df.shape}")
        # IMPORTANT: Assuming the CSV is already sorted by date as stated.
    except FileNotFoundError:
        print("Error: The file 'yt_dataset_en_v4.csv' was not found.")
        exit()

    features_basic = [
        'view_count', 'likes', 'comment_count', 'view_like_ratio', 'avg_pos', 'avg_neu', 'avg_neg', 'avg_compound'
    ]
    target_basic = 'dislikes'
    features_log = [
        'log_view_count', 'log_likes', 'log_comment_count', 'avg_pos', 'avg_neu', 'avg_neg', 'avg_compound'
    ]
    target_log = 'log_dislikes'
    
    # --- ROBUST DATA CLEANING ---
    df.replace('#NUM!', np.nan, inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    all_cols_to_use = list(set(features_basic + [target_basic] + features_log + [target_log]))
    for col in all_cols_to_use:
        # This loop will now work correctly because `col` will be a valid string name
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)
    print(f"Dataset cleaned. Shape after cleaning: {df.shape}")

    # ===================================================================
    # --- 75-10-15 Chronological Split ---
    # ===================================================================
    n_rows = len(df)
    train_end_index = int(n_rows * 0.75)
    validation_end_index = int(n_rows * (0.75 + 0.10)) # 85% mark
    
    train_df = df.iloc[:train_end_index]
    validation_df = df.iloc[train_end_index:validation_end_index]
    test_df = df.iloc[validation_end_index:]
    
    print("\n--- Chronological Data Split ---")
    print(f"Training set size:   {len(train_df)} ({len(train_df)/n_rows:.0%})")
    print(f"Validation set size: {len(validation_df)} ({len(validation_df)/n_rows:.0%})")
    print(f"Test set size:       {len(test_df)} ({len(test_df)/n_rows:.0%})")
    
    # --- Model A: Basic Approach (Predicting Raw Dislikes) ---
    print("\n--- Training Model A (Basic Approach) ---")

    X_train_b, y_train_b = train_df[features_basic], train_df[target_basic]
    X_test_b, y_test_b = test_df[features_basic], test_df[target_basic]

    pipeline_A = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ])

    print("Fitting Model A on the training set...")
    pipeline_A.fit(X_train_b, y_train_b)

    print("Evaluating Model A on the test set...")
    y_pred_A = pipeline_A.predict(X_test_b)
    
    r2_A = r2_score(y_test_b, y_pred_A)
    mae_A = mean_absolute_error(y_test_b, y_pred_A)
    
    print("Model A evaluation complete.")

    # --- Model B: Improved Approach (Predicting Log Dislikes) ---
    print("\n--- Training Model B (Improved Log-Transformed Approach) ---")
    
    X_train_l, y_train_l = train_df[features_log], train_df[target_log]
    X_test_l, y_test_l = test_df[features_log], test_df[target_log]

    pipeline_B = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ])

    print("Fitting Model B on the training set...")
    pipeline_B.fit(X_train_l, y_train_l)

    print("Evaluating Model B on the test set...")
    y_pred_log_B = pipeline_B.predict(X_test_l)

    y_pred_B_original_scale = np.expm1(y_pred_log_B)
    y_test_original_scale = np.expm1(y_test_l)

    r2_B = r2_score(y_test_original_scale, y_pred_B_original_scale)
    mae_B = mean_absolute_error(y_test_original_scale, y_pred_B_original_scale)
    
    print("Model B evaluation complete.")

    # --- Final Summary of Results ---
    print("\n\n--- EXPERIMENT RESULTS SUMMARY ---")
    results = {
        "Model": ["Model A (Basic)", "Model B (Log-Transformed)"],
        "R-squared (R²)": [r2_A, r2_B],
        "Mean Absolute Error (MAE)": [mae_A, mae_B]
    }
    results_df = pd.DataFrame(results).set_index("Model")
    results_df['R-squared (R²)'] = results_df['R-squared (R²)'].map('{:.4f}'.format)
    results_df['Mean Absolute Error (MAE)'] = results_df['Mean Absolute Error (MAE)'].map('{:.1f}'.format)
    print(results_df)

    # --- Visualize the Results to Enhance the Report ---
    print("\n\n--- Generating Visualizations ---")

    # --- Plot 1: Predicted vs. Actual Values ---
    # This plot is the best way to visually assess regression performance.
    # A perfect model would have all points on the 45-degree line.

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Comparison of Model Performance: Predicted vs. Actual Dislikes', fontsize=16)

    # Plot for Model A (Basic)
    sns.scatterplot(x=y_test_b, y=y_pred_A, alpha=0.5, ax=ax1)
    ax1.set_title('Model A (Basic Approach)')
    ax1.set_xlabel('Actual Dislikes')
    ax1.set_ylabel('Predicted Dislikes')
    # Add a line for perfect predictions
    min_val = min(y_test_b.min(), y_pred_A.min())
    max_val = max(y_test_b.max(), y_pred_A.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)

    # Plot for Model B (Log-Transformed)
    sns.scatterplot(x=y_test_original_scale, y=y_pred_B_original_scale, alpha=0.5, ax=ax2)
    ax2.set_title('Model B (Log-Transformed Approach)')
    ax2.set_xlabel('Actual Dislikes')
    ax2.set_ylabel('Predicted Dislikes')
    # Add a line for perfect predictions
    min_val_b = min(y_test_original_scale.min(), y_pred_B_original_scale.min())
    max_val_b = max(y_test_original_scale.max(), y_pred_B_original_scale.max())
    ax2.plot([min_val_b, max_val_b], [min_val_b, max_val_b], 'r--', lw=2)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])


    # --- Plot : Distribution of Residuals (Errors) using Subplots ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Distribution of Prediction Errors (Residuals)', fontsize=16)

    # Calculate residuals
    residuals_A = y_test_b - y_pred_A
    residuals_B = y_test_original_scale - y_pred_B_original_scale

    # Plot for Model A (Basic)
    sns.histplot(residuals_A, color='skyblue', kde=True, ax=ax1, bins=50)
    ax1.set_title(f'Model A (Basic) Errors\nMAE: {mae_A:.0f}')
    ax1.set_xlabel('Error (Actual - Predicted)')
    ax1.set_ylabel('Frequency')
    ax1.axvline(0, color='r', linestyle='--')

    # Plot for Model B (Log-Transformed)
    sns.histplot(residuals_B, color='green', kde=True, ax=ax2, bins=50)
    ax2.set_title(f'Model B (Log-Transformed) Errors\nMAE: {mae_B:.0f}')
    ax2.set_xlabel('Error (Actual - Predicted)')
    ax2.set_ylabel('') # Hide y-label for cleaner look
    ax2.axvline(0, color='r', linestyle='--')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])


    # --- Plot 3: Feature Importance for the Best Model (Model B) ---
    # This plot shows what features the improved model relied on most.

    # Extract feature importances from the pipeline
    importances = pipeline_B.named_steps['regressor'].feature_importances_
    feature_names = X_train_l.columns

    # Create a DataFrame for easier plotting
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance_df)
    plt.title('Feature Importance for Best Model (Model B)')
    plt.xlabel('Importance')
    plt.ylabel('Feature')


    # Show all the plots
    plt.show()

    print("\nVisualizations generated. Close the plot windows to end the script.")