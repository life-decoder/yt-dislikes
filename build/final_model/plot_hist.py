# d:\Coding\Machine learning\YT dislikes\analyze_dislikes.py
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('d:\\Coding\\Machine learning\\YT dislikes\\final_model\\yt_dataset_v5_percentage.csv')

# Compute percentage_dislikes and log_percentage_dislikes
df['percentage_dislikes'] = df['dislikes'] / (df['likes'] + df['dislikes']) * 100
df['log_percentage_dislikes'] = np.log(df['percentage_dislikes'] + 1)  # Add 1 to avoid log(0)

# after dataframe `df` with columns 'percentage_dislikes' and 'log_percentage_dislikes' is ready:
plots_dir = r"d:\Coding\Machine learning\YT dislikes\plots"
os.makedirs(plots_dir, exist_ok=True)

# Percentage dislikes histogram + KDE
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x='percentage_dislikes', kde=True, bins=50, color='C0')
plt.title('Percentage Dislikes')
plt.xlabel('percentage_dislikes')
plt.ylabel('count')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'percentage_dislikes_hist.png'), dpi=150)
plt.close()

# Log percentage dislikes histogram + KDE
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x='log_percentage_dislikes', kde=True, bins=50, color='C1')
plt.title('Log Percentage Dislikes')
plt.xlabel('log_percentage_dislikes')
plt.ylabel('count')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'log_percentage_dislikes_hist.png'), dpi=150)
plt.close()