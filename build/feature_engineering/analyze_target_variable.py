import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Load data
df = pd.read_csv('yt_dataset_filtered.csv')

# Calculate log_dislikes
df['log_dislikes'] = np.log1p(df['dislikes'])

print('=' * 70)
print('TARGET VARIABLE COMPARISON: dislikes vs log_dislikes')
print('=' * 70)

print('\n1. DISTRIBUTION STATISTICS')
print('-' * 70)
print('Raw Dislikes:')
print(f'  Mean:     {df["dislikes"].mean():,.2f}')
print(f'  Median:   {df["dislikes"].median():,.2f}')
print(f'  Std Dev:  {df["dislikes"].std():,.2f}')
print(f'  Min:      {df["dislikes"].min():,.0f}')
print(f'  Max:      {df["dislikes"].max():,.0f}')
print(f'  Skewness: {df["dislikes"].skew():.2f}')
print(f'  Kurtosis: {df["dislikes"].kurtosis():.2f}')

print('\nLog Dislikes (log1p):')
print(f'  Mean:     {df["log_dislikes"].mean():.2f}')
print(f'  Median:   {df["log_dislikes"].median():.2f}')
print(f'  Std Dev:  {df["log_dislikes"].std():.2f}')
print(f'  Min:      {df["log_dislikes"].min():.2f}')
print(f'  Max:      {df["log_dislikes"].max():.2f}')
print(f'  Skewness: {df["log_dislikes"].skew():.2f}')
print(f'  Kurtosis: {df["log_dislikes"].kurtosis():.2f}')

print('\n2. NORMALITY TESTS')
print('-' * 70)
# Shapiro-Wilk test (sample due to size)
sample_size = min(5000, len(df))
sample = df.sample(n=sample_size, random_state=42)

_, p_raw = stats.shapiro(sample['dislikes'])
_, p_log = stats.shapiro(sample['log_dislikes'])

print('Shapiro-Wilk test (p-value):')
print(f'  Raw dislikes:     {p_raw:.6f} (Normal if p > 0.05)')
print(f'  Log dislikes:     {p_log:.6f} (Normal if p > 0.05)')

print('\n3. EXTREME VALUES ANALYSIS')
print('-' * 70)
q1_raw = df['dislikes'].quantile(0.25)
q3_raw = df['dislikes'].quantile(0.75)
iqr_raw = q3_raw - q1_raw
outliers_raw = ((df['dislikes'] < q1_raw - 1.5*iqr_raw) | (df['dislikes'] > q3_raw + 1.5*iqr_raw)).sum()

q1_log = df['log_dislikes'].quantile(0.25)
q3_log = df['log_dislikes'].quantile(0.75)
iqr_log = q3_log - q1_log
outliers_log = ((df['log_dislikes'] < q1_log - 1.5*iqr_log) | (df['log_dislikes'] > q3_log + 1.5*iqr_log)).sum()

print('Outliers (IQR method):')
print(f'  Raw dislikes:     {outliers_raw:,} ({100*outliers_raw/len(df):.1f}%)')
print(f'  Log dislikes:     {outliers_log:,} ({100*outliers_log/len(df):.1f}%)')

print('\n4. PERCENTILE BREAKDOWN')
print('-' * 70)
percentiles = [0, 25, 50, 75, 90, 95, 99, 100]
print(f'{"Percentile":<12} {"Raw Dislikes":>15} {"Log Dislikes":>15}')
print('-' * 44)
for p in percentiles:
    raw_val = df['dislikes'].quantile(p/100)
    log_val = df['log_dislikes'].quantile(p/100)
    print(f'{p}th{"":<9} {raw_val:>15,.0f} {log_val:>15.2f}')

print('\n5. INTERPRETATION')
print('=' * 70)
print('Raw Dislikes:')
print('  + Interpretable: Direct prediction of actual dislike count')
print('  + Business value: Stakeholders understand raw numbers')
print('  - Highly skewed: Mean >> Median (right-skewed distribution)')
print('  - Outliers: Many extreme values affecting model')
print('  - Heteroscedasticity: Variance increases with predicted values')
print('')
print('Log Dislikes:')
print('  + Better distribution: More symmetric, closer to normal')
print('  + Reduced outliers: Extreme values compressed')
print('  + Homoscedasticity: More constant variance')
print('  + Better for linear models: Meets assumptions better')
print('  - Less interpretable: Need to exponentiate predictions')
print('  - Percentage errors: Model predicts relative, not absolute')

print('\n6. RECOMMENDATION')
print('=' * 70)

# Calculate coefficient of variation
cv_raw = df['dislikes'].std() / df['dislikes'].mean()
cv_log = df['log_dislikes'].std() / df['log_dislikes'].mean()

if df['log_dislikes'].skew() < 1 and df['dislikes'].skew() > 5:
    print('STRONG RECOMMENDATION: Use log_dislikes as target')
    print('')
    print('Reasons:')
    print(f'  1. Skewness reduced from {df["dislikes"].skew():.1f} to {df["log_dislikes"].skew():.1f}')
    print(f'  2. Coefficient of variation: {cv_log:.2f} vs {cv_raw:.2f}')
    print(f'  3. Better for regression assumptions')
    print('')
    print('To get actual dislikes: predictions = np.expm1(log_predictions)')
else:
    print('MODERATE RECOMMENDATION: Try both and compare')

print('=' * 70)

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Raw dislikes histogram
axes[0, 0].hist(df['dislikes'], bins=100, edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('Dislikes')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title(f'Raw Dislikes Distribution\nSkewness: {df["dislikes"].skew():.2f}')
axes[0, 0].axvline(df['dislikes'].mean(), color='red', linestyle='--', label=f'Mean: {df["dislikes"].mean():,.0f}')
axes[0, 0].axvline(df['dislikes'].median(), color='green', linestyle='--', label=f'Median: {df["dislikes"].median():,.0f}')
axes[0, 0].legend()

# Log dislikes histogram
axes[0, 1].hist(df['log_dislikes'], bins=100, edgecolor='black', alpha=0.7, color='orange')
axes[0, 1].set_xlabel('Log(Dislikes + 1)')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title(f'Log Dislikes Distribution\nSkewness: {df["log_dislikes"].skew():.2f}')
axes[0, 1].axvline(df['log_dislikes'].mean(), color='red', linestyle='--', label=f'Mean: {df["log_dislikes"].mean():.2f}')
axes[0, 1].axvline(df['log_dislikes'].median(), color='green', linestyle='--', label=f'Median: {df["log_dislikes"].median():.2f}')
axes[0, 1].legend()

# Q-Q plot for raw dislikes
stats.probplot(sample['dislikes'], dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('Q-Q Plot: Raw Dislikes')
axes[1, 0].grid(True, alpha=0.3)

# Q-Q plot for log dislikes
stats.probplot(sample['log_dislikes'], dist="norm", plot=axes[1, 1])
axes[1, 1].set_title('Q-Q Plot: Log Dislikes')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('target_variable_comparison.png', dpi=300, bbox_inches='tight')
print('\nVisualization saved: target_variable_comparison.png')
