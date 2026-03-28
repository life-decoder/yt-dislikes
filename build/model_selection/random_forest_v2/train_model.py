# d:\Coding\Machine learning\YT dislikes\random_forest_regression.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# Load the dataset (assuming it's named 'youtube_data.csv' in the directory)
data = pd.read_csv('yt_dataset_v6.csv')

# Define features, excluding 'dislikes', 'log_dislikes', 'percentage_dislikes', 'log_percentage_dislikes', and non-numeric fields
features = [col for col in data.columns if col not in ['dislikes', 'log_dislikes', 'percentage_dislikes', 'log_percentage_dislikes'] and data[col].dtype in ['int64', 'float64']]
X = data[features]
y = data['log_percentage_dislikes']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'MSE: {mse}, R2: {r2}')

# Visualizations
# Regression plot (Predicted vs Actual)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual log_percentage_dislikes')
plt.ylabel('Predicted log_percentage_dislikes')
plt.title('Regression Plot')
plt.show()

# Residual plot
residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_pred, y=residuals)
plt.axhline(0, color='r', linestyle='--')
plt.xlabel('Predicted log_percentage_dislikes')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()

# Q-Q plot for residuals
plt.figure(figsize=(8, 6))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('Q-Q Plot of Residuals')
plt.show()

# Feature importance plot
importances = model.feature_importances_
feature_names = X.columns
plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=feature_names)
plt.xlabel('Importance')
plt.ylabel('Features')
plt.title('Feature Importances')
plt.show()