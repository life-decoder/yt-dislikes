import pandas as pd

df = pd.read_csv('../yt_dataset_v4.csv')
print(f'Shape: {df.shape}')
print(f'\nColumns:\n{list(df.columns)}')
print(f'\nFirst few rows:')
print(df.head())
print(f'\nData types:')
print(df.dtypes)
print(f'\nMissing values:')
print(df.isnull().sum())
