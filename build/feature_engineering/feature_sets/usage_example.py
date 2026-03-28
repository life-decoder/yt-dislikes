
# Example: Load filtered dataset and train a model

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Import feature configuration
import sys
sys.path.append('feature_engineering/feature_sets')
from feature_sets_config import TIER2_TREE, TARGET

# Load filtered dataset
df = pd.read_csv('yt_dataset_filtered.csv')

# Select features and target
X = df[TIER2_TREE]
y = df[TARGET]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Handle categorical features if needed
if 'desc_lang' in X_train.columns:
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    X_train['desc_lang'] = le.fit_transform(X_train['desc_lang'])
    X_test['desc_lang'] = le.transform(X_test['desc_lang'])

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:,.2f}")
print(f"R² Score: {r2:.4f}")
