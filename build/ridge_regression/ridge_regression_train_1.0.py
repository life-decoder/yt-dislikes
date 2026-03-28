import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data(csv_path):
    """Load the dataset and prepare features."""
    print("📊 Loading dataset...")
    df = pd.read_csv(csv_path)
    
    # Convert published_at to datetime for time-based splitting
    if 'published_at' in df.columns:
        df['published_at'] = pd.to_datetime(df['published_at'])
        df = df.sort_values('published_at').reset_index(drop=True)
        print(f"📅 Date range: {df['published_at'].min()} to {df['published_at'].max()}")
    
    print(f"📊 Loaded dataset with {len(df)} videos")
    
    # Check for missing values
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print(f"⚠️  Missing values found:")
        for col, missing_count in missing_values[missing_values > 0].items():
            print(f"   {col}: {missing_count} missing values ({missing_count/len(df)*100:.2f}%)")
    
    return df

def create_time_based_split(df, train_ratio=0.75, val_ratio=0.10):
    """Split data chronologically based on published_at."""
    print("\n" + "="*50)
    print("🎯 TIME-BASED DATA SPLIT")
    print("="*50)
    
    n_total = len(df)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val
    
    # Split chronologically
    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train:n_train + n_val]
    test_df = df.iloc[n_train + n_val:]
    
    print(f"✅ Time-based split completed:")
    print(f"   🏋️  Training set: {len(train_df)} videos")
    print(f"   ⚖️  Validation set: {len(val_df)} videos")
    print(f"   🧪 Test set: {len(test_df)} videos")
    
    return train_df, val_df, test_df

def handle_missing_values(X_train, X_val, X_test, strategy='mean'):
    """Handle missing values in the dataset."""
    print(f"\n🔧 Handling missing values with strategy: '{strategy}'")
    
    # Create imputer
    imputer = SimpleImputer(strategy=strategy)
    
    # Fit on training data and transform all sets
    X_train_imputed = imputer.fit_transform(X_train)
    X_val_imputed = imputer.transform(X_val)
    X_test_imputed = imputer.transform(X_test)
    
    # Convert back to DataFrames to preserve column names
    X_train_clean = pd.DataFrame(X_train_imputed, columns=X_train.columns, index=X_train.index)
    X_val_clean = pd.DataFrame(X_val_imputed, columns=X_val.columns, index=X_val.index)
    X_test_clean = pd.DataFrame(X_test_imputed, columns=X_test.columns, index=X_test.index)
    
    print(f"✅ Missing values handled:")
    print(f"   🏋️  Training: {X_train.isnull().sum().sum()} → {X_train_clean.isnull().sum().sum()} missing")
    print(f"   ⚖️  Validation: {X_val.isnull().sum().sum()} → {X_val_clean.isnull().sum().sum()} missing")
    print(f"   🧪 Test: {X_test.isnull().sum().sum()} → {X_test_clean.isnull().sum().sum()} missing")
    
    return X_train_clean, X_val_clean, X_test_clean, imputer

def prepare_features(train_df, val_df, test_df):
    """Prepare features for ridge regression."""
    print("\n" + "="*50)
    print("🔧 FEATURE PREPARATION")
    print("="*50)
    
    # Available features from your dataset (excluding target and metadata)
    available_features = [
        'age', 'avg_compound', 'avg_neg', 'avg_neu', 'avg_pos',
        'comment_count', 'comment_sample_size', 'likes', 
        'log_comment_count', 'log_likes', 'log_view_count',
        'no_comments', 'view_count', 'view_like_ratio'
    ]
    
    # Check which features are actually available
    features_to_use = [col for col in available_features if col in train_df.columns]
    
    print(f"✅ Using {len(features_to_use)} features:")
    for feature in features_to_use:
        missing_train = train_df[feature].isnull().sum()
        missing_val = val_df[feature].isnull().sum()
        print(f"   📍 {feature:<20} | Missing: train={missing_train}, val={missing_val}")
    
    # Target variable
    target_column = 'log_dislikes'
    
    # Check for missing values in target
    missing_target_train = train_df[target_column].isnull().sum()
    missing_target_val = val_df[target_column].isnull().sum()
    missing_target_test = test_df[target_column].isnull().sum()
    
    if missing_target_train > 0 or missing_target_val > 0 or missing_target_test > 0:
        print(f"⚠️  Missing values in target '{target_column}':")
        print(f"   Training: {missing_target_train}, Validation: {missing_target_val}, Test: {missing_target_test}")
        # Remove rows with missing target values
        train_df = train_df.dropna(subset=[target_column])
        val_df = val_df.dropna(subset=[target_column])
        test_df = test_df.dropna(subset=[target_column])
        print(f"✅ Removed rows with missing target values")
    
    # Prepare feature matrices
    X_train = train_df[features_to_use]
    X_val = val_df[features_to_use]
    X_test = test_df[features_to_use]
    
    # Target variables
    y_train = train_df[target_column]
    y_val = val_df[target_column]
    y_test = test_df[target_column]
    
    print(f"\n📊 Data shapes after handling missing targets:")
    print(f"   🏋️  Training: {X_train.shape[1]} features, {X_train.shape[0]} samples")
    print(f"   ⚖️  Validation: {X_val.shape[1]} features, {X_val.shape[0]} samples")
    print(f"   🧪 Test: {X_test.shape[1]} features, {X_test.shape[0]} samples")
    print(f"   🎯 Target: {target_column}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, features_to_use, target_column

def train_ridge_regression(X_train, X_val, y_train, y_val, alpha_values=None):
    """Train ridge regression with hyperparameter tuning."""
    print("\n" + "="*50)
    print("🎯 RIDGE REGRESSION TRAINING")
    print("="*50)
    
    if alpha_values is None:
        alpha_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    
    # Handle missing values in features
    X_train_clean, X_val_clean, _, imputer = handle_missing_values(X_train, X_val, X_val)  # We only need train/val for training
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_clean)
    X_val_scaled = scaler.transform(X_val_clean)
    
    best_alpha = None
    best_val_mse = float('inf')
    best_model = None
    results = []
    
    print("🔍 Tuning hyperparameters...")
    for alpha in alpha_values:
        # Train model
        model = Ridge(alpha=alpha, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Predict on validation set
        y_val_pred = model.predict(X_val_scaled)
        val_mse = mean_squared_error(y_val, y_val_pred)
        val_mae = mean_absolute_error(y_val, y_val_pred)
        val_r2 = r2_score(y_val, y_val_pred)
        
        results.append({
            'alpha': alpha,
            'val_mse': val_mse,
            'val_mae': val_mae,
            'val_r2': val_r2
        })
        
        print(f"   α={alpha:6.3f} | MSE={val_mse:.4f} | MAE={val_mae:.4f} | R²={val_r2:.4f}")
        
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_alpha = alpha
            best_model = model
    
    print(f"\n✅ Best hyperparameters:")
    print(f"   🏆 Best alpha: {best_alpha}")
    print(f"   📊 Best validation MSE: {best_val_mse:.4f}")
    
    return best_model, scaler, imputer, best_alpha, results

def evaluate_model(model, scaler, imputer, X_test, y_test, set_name="Test"):
    """Evaluate the model on test set."""
    print(f"\n" + "="*50)
    print(f"📊 {set_name.upper()} SET EVALUATION")
    print("="*50)
    
    # Handle missing values and scale
    X_test_clean = imputer.transform(X_test)
    X_test_scaled = scaler.transform(X_test_clean)
    
    y_pred = model.predict(X_test_scaled)
    
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"📈 {set_name} Set Performance:")
    print(f"   📏 Mean Squared Error (MSE): {mse:.4f}")
    print(f"   📏 Mean Absolute Error (MAE): {mae:.4f}")
    print(f"   📊 R² Score: {r2:.4f}")
    
    # Calculate RMSE
    rmse = np.sqrt(mse)
    print(f"   📏 Root Mean Squared Error (RMSE): {rmse:.4f}")
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'predictions': y_pred,
        'actual': y_test
    }

def plot_feature_importance(model, feature_names):
    """Plot feature importance."""
    print("\n" + "="*50)
    print("📊 FEATURE IMPORTANCE")
    print("="*50)
    
    importance = np.abs(model.coef_)
    feature_imp_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    print("🎯 Feature importances (absolute coefficients):")
    for i, row in feature_imp_df.iterrows():
        print(f"   {row['feature']:<20}: {row['importance']:.4f}")
    
    # Plot
    plt.figure(figsize=(12, 8))
    sns.barplot(data=feature_imp_df, x='importance', y='feature', palette='viridis')
    plt.title('Feature Importances - Ridge Regression (log_dislikes prediction)')
    plt.xlabel('Absolute Coefficient Value')
    plt.tight_layout()
    plt.savefig('feature_importance_log_dislikes.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return feature_imp_df

def plot_predictions(y_true, y_pred, set_name="Test"):
    """Plot actual vs predicted values."""
    plt.figure(figsize=(10, 8))
    
    plt.scatter(y_true, y_pred, alpha=0.6, s=30, color='blue')
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2, label='Perfect Prediction')
    
    plt.xlabel('Actual log_dislikes')
    plt.ylabel('Predicted log_dislikes')
    plt.title(f'Actual vs Predicted log_dislikes ({set_name} Set)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'actual_vs_predicted_log_dislikes_{set_name.lower()}.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_actual_vs_predicted_dislikes(test_df, y_pred_log, set_name="Test"):
    """Plot actual dislikes vs predicted dislikes (converted from log scale)."""
    print(f"\n" + "="*50)
    print(f"📊 ACTUAL VS PREDICTED DISLIKES")
    print("="*50)
    
    # Convert log predictions back to original scale
    y_pred_actual = np.exp(y_pred_log)
    y_actual_actual = np.exp(test_df['log_dislikes'].values)
    
    # Calculate metrics in original scale
    mse_actual = mean_squared_error(y_actual_actual, y_pred_actual)
    mae_actual = mean_absolute_error(y_actual_actual, y_pred_actual)
    r2_actual = r2_score(y_actual_actual, y_pred_actual)
    
    print(f"📈 Performance in Original Dislikes Scale:")
    print(f"   📏 Mean Squared Error (MSE): {mse_actual:.2f}")
    print(f"   📏 Mean Absolute Error (MAE): {mae_actual:.2f}")
    print(f"   📊 R² Score: {r2_actual:.4f}")
    print(f"   📏 RMSE: {np.sqrt(mse_actual):.2f}")
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Scatter plot with transparency to handle overlapping points
    plt.scatter(y_actual_actual, y_pred_actual, alpha=0.6, s=20, color='steelblue', edgecolors='white', linewidth=0.5)
    
    # Perfect prediction line
    max_val = max(y_actual_actual.max(), y_pred_actual.max())
    plt.plot([0, max_val], [0, max_val], 'r--', alpha=0.8, linewidth=2, label='Perfect Prediction')
    
    # Add regression line to show trend
    z = np.polyfit(y_actual_actual, y_pred_actual, 1)
    p = np.poly1d(z)
    plt.plot(y_actual_actual, p(y_actual_actual), "g--", alpha=0.8, linewidth=2, label='Regression Line')
    
    plt.xlabel('Actual Dislikes', fontsize=12)
    plt.ylabel('Predicted Dislikes', fontsize=12)
    plt.title(f'Actual vs Predicted Dislikes ({set_name} Set)\nR² = {r2_actual:.3f}, MAE = {mae_actual:.1f}', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Set reasonable axis limits
    plt.xlim(0, min(max_val * 1.1, 50000))  # Cap at 50k for better visualization
    plt.ylim(0, min(max_val * 1.1, 50000))
    
    plt.tight_layout()
    plt.savefig(f'actual_vs_predicted_dislikes_{set_name.lower()}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Additional plot focusing on lower range for better detail
    plt.figure(figsize=(12, 8))
    plt.scatter(y_actual_actual, y_pred_actual, alpha=0.7, s=25, color='coral', edgecolors='white', linewidth=0.3)
    plt.plot([0, 5000], [0, 5000], 'r--', alpha=0.8, linewidth=2, label='Perfect Prediction')
    
    plt.xlabel('Actual Dislikes', fontsize=12)
    plt.ylabel('Predicted Dislikes', fontsize=12)
    plt.title(f'Actual vs Predicted Dislikes ({set_name} Set) - Zoomed View\n(0-5000 range)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 5000)
    plt.ylim(0, 5000)
    
    plt.tight_layout()
    plt.savefig(f'actual_vs_predicted_dislikes_zoomed_{set_name.lower()}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'mse_actual': mse_actual,
        'mae_actual': mae_actual,
        'r2_actual': r2_actual,
        'rmse_actual': np.sqrt(mse_actual)
    }

def plot_residuals(y_true, y_pred, set_name="Test"):
    """Plot residuals."""
    residuals = y_true - y_pred
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_pred, residuals, alpha=0.6, color='red')
    plt.axhline(y=0, color='black', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title(f'Residual Plot ({set_name} Set)')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.hist(residuals, bins=50, alpha=0.7, color='green', edgecolor='black')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title(f'Residual Distribution ({set_name} Set)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'residuals_plot_{set_name.lower()}.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_model(model, scaler, imputer, feature_names, alpha, metrics, filename='ridge_regression_log_dislikes.pkl'):
    """Save the trained model and metadata."""
    model_data = {
        'model': model,
        'scaler': scaler,
        'imputer': imputer,
        'feature_names': feature_names,
        'alpha': alpha,
        'training_date': datetime.now(),
        'metrics': metrics
    }
    
    joblib.dump(model_data, filename)
    print(f"💾 Model saved to: {filename}")

def main():
    """Main function to run the ridge regression pipeline."""
    print("=== RIDGE REGRESSION FOR log_dislikes PREDICTION ===")
    print("=" * 55)
    
    # Configuration
    CSV_PATH = "yt_dataset_v4.csv"
    TRAIN_RATIO = 0.75
    VAL_RATIO = 0.10
    
    try:
        # Step 1: Load and prepare data
        df = load_and_prepare_data(CSV_PATH)
        
        # Step 2: Create time-based split
        train_df, val_df, test_df = create_time_based_split(df, TRAIN_RATIO, VAL_RATIO)
        
        # Step 3: Prepare features
        X_train, X_val, X_test, y_train, y_val, y_test, feature_names, target_column = prepare_features(
            train_df, val_df, test_df
        )
        
        # Step 4: Train ridge regression
        model, scaler, imputer, best_alpha, tuning_results = train_ridge_regression(X_train, X_val, y_train, y_val)
        
        # Step 5: Evaluate on validation set
        val_metrics = evaluate_model(model, scaler, imputer, X_val, y_val, "Validation")
        
        # Step 6: Evaluate on test set
        test_metrics = evaluate_model(model, scaler, imputer, X_test, y_test, "Test")
        
        # Step 7: Plot feature importance
        feature_imp_df = plot_feature_importance(model, feature_names)
        
        # Step 8: Plot predictions and residuals
        plot_predictions(y_test, test_metrics['predictions'], "Test")
        plot_residuals(y_test, test_metrics['predictions'], "Test")
        
        # Step 9: NEW - Plot actual vs predicted dislikes (converted from log scale)
        actual_dislikes_metrics = plot_actual_vs_predicted_dislikes(test_df, test_metrics['predictions'], "Test")
        
        # Step 10: Save model
        all_metrics = {
            'validation': val_metrics,
            'test': test_metrics,
            'actual_dislikes': actual_dislikes_metrics,
            'tuning': tuning_results,
            'best_alpha': best_alpha
        }
        save_model(model, scaler, imputer, feature_names, best_alpha, all_metrics)
        
        print("\n" + "="*50)
        print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print("="*50)
        print(f"📊 Final Test Performance (log scale):")
        print(f"   📏 MSE: {test_metrics['mse']:.4f}")
        print(f"   📏 RMSE: {test_metrics['rmse']:.4f}")
        print(f"   📏 MAE: {test_metrics['mae']:.4f}")
        print(f"   📊 R²: {test_metrics['r2']:.4f}")
        print(f"\n📊 Final Test Performance (actual dislikes):")
        print(f"   📏 MSE: {actual_dislikes_metrics['mse_actual']:.2f}")
        print(f"   📏 RMSE: {actual_dislikes_metrics['rmse_actual']:.2f}")
        print(f"   📏 MAE: {actual_dislikes_metrics['mae_actual']:.2f}")
        print(f"   📊 R²: {actual_dislikes_metrics['r2_actual']:.4f}")
        print(f"\n💾 Model saved: ridge_regression_log_dislikes.pkl")
        print(f"📈 Plots saved:")
        print(f"   - feature_importance_log_dislikes.png")
        print(f"   - actual_vs_predicted_log_dislikes_test.png")
        print(f"   - actual_vs_predicted_dislikes_test.png")
        print(f"   - actual_vs_predicted_dislikes_zoomed_test.png")
        print(f"   - residuals_plot_test.png")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()