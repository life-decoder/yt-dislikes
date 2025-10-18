import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

# === Create output directory ===
os.makedirs("mlp_outputs", exist_ok=True)

# === Load Dataset ===
df = pd.read_csv("yt_dataset_v4.csv")
df = df.drop(columns=["video_id", "channel_id", "published_at"], errors="ignore")
df = df.apply(pd.to_numeric, errors='coerce').dropna()
df = df[df["dislikes"] < df["dislikes"].quantile(0.99)]  # remove top 1% outliers

# === Prepare Features ===
feature_cols = [col for col in df.columns if col not in ["dislikes", "log_dislikes"]]
X = df[feature_cols]
y_raw = df["dislikes"]
y_log = df["log_dislikes"]

# === Train/Val/Test Split (75/10/15) ===
X_train, X_temp, y_train_r, y_temp_r, y_train_l, y_temp_l = train_test_split(
    X, y_raw, y_log, test_size=0.25, random_state=42
)
X_val, X_test, y_val_r, y_test_r, y_val_l, y_test_l = train_test_split(
    X_temp, y_temp_r, y_temp_l, test_size=0.6, random_state=42
)

# === Scale Features ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# === Helper Function for Training & Evaluation ===
def train_and_evaluate(y_train, y_test, transform_name):
    mlp = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42
    )
    mlp.fit(X_train_scaled, y_train)
    y_pred = mlp.predict(X_test_scaled)
    y_train_pred = mlp.predict(X_train_scaled)
    
    # Back-transform if log
    if transform_name == "log":
        y_true = np.expm1(y_test)
        y_pred_original = np.expm1(y_pred)
        y_train_true = np.expm1(y_train)
        y_train_pred_original = np.expm1(y_train_pred)
    else:
        y_true = y_test
        y_pred_original = y_pred
        y_train_true = y_train
        y_train_pred_original = y_train_pred

    # === Metrics ===
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_true, y_pred_original)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred_original))
    mape = np.mean(np.abs((y_true - y_pred_original) / y_true)) * 100
    train_r2 = r2_score(y_train, y_train_pred)
    
    print(f"\n📊 {transform_name.upper()} MODEL RESULTS")
    print(f"Train R²: {train_r2:.4f}")
    print(f"Test R² Score: {r2:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"Mean % Error (MAPE): {mape:.2f}%")

    # === Save model ===
    model_filename = f"mlp_outputs/mlp_model_{transform_name}.pkl"
    joblib.dump(mlp, model_filename)
    print(f"✓ Model saved: {model_filename}")

    # === Save predictions ===
    predictions_df = pd.DataFrame({
        'dataset': ['train']*len(y_train_true) + ['test']*len(y_true),
        'actual': np.concatenate([y_train_true, y_true]),
        'predicted': np.concatenate([y_train_pred_original, y_pred_original])
    })
    pred_filename = f"mlp_outputs/predictions_{transform_name}.csv"
    predictions_df.to_csv(pred_filename, index=False)
    print(f"✓ Predictions saved: {pred_filename}")

    # === Plots ===
    plot_filenames = []
    
    # Actual vs Predicted
    plt.figure(figsize=(8,6))
    plt.scatter(y_true, y_pred_original, alpha=0.5, color='blue')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.title(f'Actual vs Predicted ({transform_name.upper()})')
    plt.xlabel('Actual Dislikes')
    plt.ylabel('Predicted Dislikes')
    plt.grid(True)
    plot_file = f"mlp_outputs/actual_vs_pred_{transform_name}.png"
    plt.savefig(plot_file, dpi=300)
    plot_filenames.append(plot_file)
    plt.close()
    
    # Residuals Distribution
    residuals = y_true - y_pred_original
    plt.figure(figsize=(8,6))
    plt.hist(residuals, bins=30, color='skyblue', edgecolor='black')
    plt.title(f'Residual Distribution ({transform_name.upper()})')
    plt.xlabel('Error (Actual - Predicted)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plot_file = f"mlp_outputs/residuals_{transform_name}.png"
    plt.savefig(plot_file, dpi=300)
    plot_filenames.append(plot_file)
    plt.close()
    
    # Residuals vs Predicted
    plt.figure(figsize=(8,6))
    plt.scatter(y_pred_original, residuals, alpha=0.5, color='orange')
    plt.axhline(0, color='red', linestyle='--')
    plt.title(f'Residuals vs Predicted ({transform_name.upper()})')
    plt.xlabel('Predicted Dislikes')
    plt.ylabel('Residual')
    plt.grid(True)
    plot_file = f"mlp_outputs/residuals_vs_pred_{transform_name}.png"
    plt.savefig(plot_file, dpi=300)
    plot_filenames.append(plot_file)
    plt.close()
    
    # Absolute Error per Sample
    abs_error = np.abs(residuals)
    plt.figure(figsize=(10,5))
    plt.plot(abs_error[:100], color='purple')
    plt.title(f'Absolute Error per Sample ({transform_name.upper()})')
    plt.xlabel('Sample Index')
    plt.ylabel('Absolute Error')
    plt.grid(True)
    plot_file = f"mlp_outputs/abs_error_{transform_name}.png"
    plt.savefig(plot_file, dpi=300)
    plot_filenames.append(plot_file)
    plt.close()
    
    # Percentage Error Distribution
    perc_error = (residuals / y_true) * 100
    plt.figure(figsize=(8,6))
    plt.hist(perc_error, bins=30, color='lightcoral', edgecolor='black')
    plt.title(f'Percentage Error Distribution ({transform_name.upper()})')
    plt.xlabel('Percentage Error (%)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plot_file = f"mlp_outputs/perc_error_{transform_name}.png"
    plt.savefig(plot_file, dpi=300)
    plot_filenames.append(plot_file)
    plt.close()
    
    print(f"✓ Plots saved: {plot_filenames}")

    return {
        "r2": r2,
        "mae": mae,
        "rmse": rmse,
        "mape": mape
    }

# === Train and Evaluate Both Models ===
metrics_raw = train_and_evaluate(y_train_r, y_test_r, "raw")
metrics_log = train_and_evaluate(y_train_l, y_test_l, "log")

# === Compare Metrics Bar Chart ===
metrics_df = pd.DataFrame([metrics_raw, metrics_log], index=["Raw", "Log"])
plt.figure(figsize=(10,6))
metrics_df[["mae", "rmse", "mape"]].plot(kind='bar')
plt.title("Model Error Comparison (Raw vs Log)")
plt.ylabel("Error Value / %")
plt.grid(True)
plt.savefig("mlp_outputs/error_comparison.png", dpi=300)
plt.close()
print("✓ Comparison plot saved: mlp_outputs/error_comparison.png")
