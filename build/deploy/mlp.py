#pip install tensorflow optuna

"""
from google.colab import drive
drive.mount('/content/drive')
"""
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
import optuna
from optuna.samplers import TPESampler
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import os
import warnings
warnings.filterwarnings('ignore')

print("TensorFlow version:", tf.__version__)

# Detect TPU
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU:', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()

print("Number of replicas:", strategy.num_replicas_in_sync)

# Reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# Exact column names from your CSV
columns = [
    'video_id', 'channel_id', 'published_at', 'dislikes', 'log_dislikes',
    'age', 'avg_compound', 'avg_neg', 'avg_neu', 'avg_pos',
    'comment_count', 'comment_sample_size', 'likes', 'log_comment_count',
    'log_likes', 'log_view_count', 'no_comments', 'view_count',
    'view_like_ratio', 'log_view_like_ratio', 'duration', 'genre_id'
]

# Load data
df = pd.read_csv('/content/yt_dataset_v5.csv', header=None, names=columns)

# Remove accidental header row
df = df[df['video_id'] != 'video_id'].copy()

# Convert numeric columns
numeric_cols = [
    'dislikes', 'log_dislikes', 'age', 'avg_compound', 'avg_neg', 'avg_neu', 'avg_pos',
    'comment_count', 'comment_sample_size', 'likes', 'log_comment_count',
    'log_likes', 'log_view_count', 'no_comments', 'view_count',
    'view_like_ratio', 'log_view_like_ratio', 'duration', 'genre_id'
]
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

# Clean target
df = df.dropna(subset=['dislikes', 'log_dislikes'])
df = df[df['dislikes'] > 0].copy()

# Features
feature_cols = [
    'age', 'avg_compound', 'avg_neg', 'avg_neu', 'avg_pos',
    'comment_count', 'likes', 'log_comment_count',
    'log_likes', 'log_view_count', 'no_comments', 'view_count',
    'view_like_ratio', 'log_view_like_ratio', 'duration', 'genre_id'
]

X = df[feature_cols].fillna(df[feature_cols].median()).values
y = df['log_dislikes'].values

# Hold-out test set
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

# Scale features
scaler = StandardScaler()
X_train_full = scaler.fit_transform(X_train_full)
X_test = scaler.transform(X_test)


# 2. Define Model-Building Function
def create_model(n_layers, units, dropout, input_dim, lr):
    model = keras.Sequential()
    model.add(layers.Input(shape=(input_dim,)))
    for i in range(n_layers):
        model.add(layers.Dense(units[i], activation='relu'))
        model.add(layers.Dropout(dropout))
    model.add(layers.Dense(1))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss='mse',
        metrics=['mae']
    )
    return model

# 3. Optuna Objective with Cross-Validation
def objective(trial):
    # Suggest hyperparameters
    n_layers = trial.suggest_int("n_layers", 1, 4)
    units = [trial.suggest_int(f"units_l{i}", 64, 512, step=64) for i in range(n_layers)]
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])

    # Adjust batch size for TPU (must be divisible by 8)
    if tpu:
        batch_size = ((batch_size + 7) // 8) * 8

    # K-Fold CV
    kfold = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = []

    for train_idx, val_idx in kfold.split(X_train_full):
        X_tr, X_val = X_train_full[train_idx], X_train_full[val_idx]
        y_tr, y_val = y_train_full[train_idx], y_train_full[val_idx]

        # Build model inside strategy scope
        with strategy.scope():
            model = create_model(n_layers, units, dropout, X_tr.shape[1], lr)

        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )

        history = model.fit(
            X_tr, y_tr,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=0
        )

        val_loss = min(history.history['val_loss'])
        cv_scores.append(val_loss)

    return np.mean(cv_scores)


# 4. Run Optuna Optimization
study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=RANDOM_STATE))
print("Starting Optuna hyperparameter tuning...")
study.optimize(objective, n_trials=30)

print("Best trial:")
print(study.best_params)

# 5. Train Final Model
best_params = study.best_params
n_layers = best_params["n_layers"]
units = [best_params[f"units_l{i}"] for i in range(n_layers)]
dropout = best_params["dropout"]
lr = best_params["lr"]
batch_size = best_params["batch_size"]

if tpu:
    batch_size = ((batch_size + 7) // 8) * 8

with strategy.scope():
    final_model = create_model(n_layers, units, dropout, X_train_full.shape[1], lr)

early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=15, restore_best_weights=True
)

history = final_model.fit(
    X_train_full, y_train_full,
    validation_split=0.1,
    epochs=200,
    batch_size=batch_size,
    callbacks=[early_stopping],
    verbose=1
)

# Evaluate
y_pred_train = final_model.predict(X_train_full, batch_size=batch_size).flatten()
y_pred_test = final_model.predict(X_test, batch_size=batch_size).flatten()

train_mse = mean_squared_error(y_train_full, y_pred_train)
test_mse = mean_squared_error(y_test, y_pred_test)
train_r2 = r2_score(y_train_full, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

print(f"\nFinal Model Performance:")
print(f"Train MSE: {train_mse:.4f}, R²: {train_r2:.4f}")
print(f"Test MSE: {test_mse:.4f}, R²: {test_r2:.4f}")

residuals = y_test - y_pred_test


# 6. Visualizations
sns.set(style="whitegrid")
plt.rcParams.update({'font.size': 12})

# Define output directory for plots
plot_output_dir = '/content/output/plots'
os.makedirs(plot_output_dir, exist_ok=True)

# Create a wrapper function for the Keras model to be compatible with scikit-learn's permutation_importance
class KerasRegressorWrapper:
    def __init__(self, model):
        self.model = model

    def predict(self, X):
        return self.model.predict(X).flatten()

    def score(self, X, y):
        # Use negative mean squared error as the scoring metric
        y_pred = self.predict(X)
        return -mean_squared_error(y, y_pred)

    def fit(self, X, y):
        # This is a dummy fit method to satisfy scikit-learn's requirement
        pass

# Wrap the final model
wrapped_model = KerasRegressorWrapper(final_model)

# 6.1 Feature Importance (Permutation)
print("Computing permutation feature importance...")
perm_importance = permutation_importance(
    wrapped_model, X_test, y_test,
    scoring='neg_mean_squared_error',  # Explicitly set scoring
    n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1
)

importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': perm_importance.importances_mean
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(data=importance_df, x='importance', y='feature', orient='h')
plt.title('Permutation Feature Importance')
plt.xlabel('Importance (Negative MSE)')
plt.tight_layout()
plt.savefig(os.path.join(plot_output_dir, 'feature_importance_tf.png'), dpi=150)
plt.show()

# Define output directory for plots
plot_output_dir = '/content/output/plots'
os.makedirs(plot_output_dir, exist_ok=True)

# 6.2 Residuals vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_pred_test, residuals, alpha=0.6)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicted log(Dislikes)')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted')
plt.tight_layout()
plt.savefig(os.path.join(plot_output_dir, 'residuals_tf.png'), dpi=150)
plt.show()

# Define output directory for plots
plot_output_dir = '/content/output/plots'
os.makedirs(plot_output_dir, exist_ok=True)

# 6.3 Actual vs Predicted
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred_test, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual log(Dislikes)')
plt.ylabel('Predicted log(Dislikes)')
plt.title(f'Actual vs Predicted (R² = {test_r2:.3f})')
plt.tight_layout()
plt.savefig(os.path.join(plot_output_dir, 'regression_tf.png'), dpi=150)
plt.show()

# Define output directory for plots
plot_output_dir = '/content/output/plots'
os.makedirs(plot_output_dir, exist_ok=True)

plt.figure(figsize=(8, 6))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('Q–Q Plot of Residuals')
plt.tight_layout()
plt.savefig(os.path.join(plot_output_dir, 'qq_tf.png'), dpi=150)
plt.show()

# Define output directory for the model
model_output_dir = '/content/output/model'
os.makedirs(model_output_dir, exist_ok=True)

# 7. Save Model
final_model.save(os.path.join(model_output_dir, 'youtube_dislikes_tpu_model.keras'))
print(f"\nModel saved as '{os.path.join(model_output_dir, 'youtube_dislikes_tpu_model.keras')}'")

import os
import shutil

source_dir = '/content/output'
destination_dir = '/content/drive/My Drive/YT_Dislikes/mlp_tf'

# Create the destination directory if it doesn't exist
os.makedirs(destination_dir, exist_ok=True)

# Use shutil.copytree for recursive copying
try:
    # Remove the destination directory if it already exists to avoid errors with copytree
    if os.path.exists(destination_dir):
        shutil.rmtree(destination_dir)
    shutil.copytree(source_dir, destination_dir)
    print(f"Recursively copied contents of '{source_dir}' to '{destination_dir}'")
except FileNotFoundError:
    print(f"Error: Source directory '{source_dir}' not found.")
except Exception as e:
    print(f"Error copying directory: {e}")

print("Output files copying process completed.")