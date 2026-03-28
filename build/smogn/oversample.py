# Script: smogn_synthetic_data.py
# Purpose: Apply SMOGN to generate synthetic data for the imbalanced 'dislikes' target variable
#          in the YouTube dataset and save the result to a new CSV file.

import pandas as pd
from sklearn.preprocessing import StandardScaler
from smogn import smoter

# --- Configuration ---
# Input file path
input_file_path = 'yt_dataset_v5.csv'  # Update this path if necessary
# Output file path for the combined original + synthetic data
output_file_path = 'yt_dataset_v5_smogn_synthetic.csv'

# Name of the target variable (the one we want to balance for regression)
target_variable = 'dislikes'

# SMOGN hyperparameters (adjust these based on your specific needs)
# Typical values for 'prop': 1.0 (generate synthetic data up to the average density of the majority)
# Typical value for 'pert': 0.02 or 0.01 (percentage of range to perturb minority instances)
# Typical value for 'k': 5 (number of nearest neighbors)
# Typical value for 'threshold': 0.95 (threshold for determining extreme values)
# Typical value for 'replace': False (do not replace values during perturbation)
smogn_params = {
    'y': target_variable,
    'pert': 0.02,     # Perturbation rate
    'k': 5,           # Number of nearest neighbors
    'replace': False   # Whether to replace values during perturbation (optional)
}

# --- Load Data ---
print(f"Loading data from {input_file_path}...")
try:
    df = pd.read_csv(input_file_path)
    print(f"Data loaded successfully. Shape: {df.shape}")
except FileNotFoundError:
    print(f"Error: File {input_file_path} not found.")
    exit(1)
except Exception as e:
    print(f"Error loading data: {e}")
    exit(1)

# --- Prepare Features and Target ---
# Identify features (all columns except the target)
feature_columns = [col for col in df.columns if col != target_variable]
X = df[feature_columns]
y = df[target_variable]

print(f"Features used: {len(feature_columns)}")
print(f"Target variable: {target_variable}")

# --- Optional: Standardize Features (SMOGN might work better with standardized features) ---
# Uncomment the following lines if you want to standardize features before applying SMOGN.
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# X_scaled_df = pd.DataFrame(X_scaled, columns=feature_columns)

# If standardizing, use X_scaled_df instead of X in the smoter call below
# X_for_smogn = X_scaled_df
X_for_smogn = X # Using original features as SMOGN documentation doesn't strictly require standardization

# --- Apply SMOGN ---
print("\nApplying SMOGN (SMOTE for Regression)...")
try:
    # Combine features and target into a single DataFrame for SMOGN
    df_for_smogn = pd.concat([X_for_smogn, y], axis=1)

    # Generate synthetic data using SMOGN
    synthetic_data = smoter(df_for_smogn, **smogn_params)

    print(f"Original data shape: {df.shape}")
    print(f"Synthetic data shape: {synthetic_data.shape}")

    # Combine original and synthetic data
    combined_data = pd.concat([df, synthetic_data], ignore_index=True)
    print(f"Combined data shape (Original + Synthetic): {combined_data.shape}")

    # --- Save Combined Data ---
    print(f"\nSaving combined data to {output_file_path}...")
    combined_data.to_csv(output_file_path, index=False)
    print(f"Synthetic data generation and saving completed successfully!")
    print(f"Saved combined dataset to: {output_file_path}")

except Exception as e:
    print(f"Error during SMOGN application or saving: {e}")
    import traceback
    traceback.print_exc()
