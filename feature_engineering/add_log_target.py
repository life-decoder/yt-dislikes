"""
Add log_dislikes column to filtered dataset for immediate use
"""
import pandas as pd
import numpy as np

# Load filtered dataset
df = pd.read_csv('yt_dataset_filtered.csv')

# Add log_dislikes column
df['log_dislikes'] = np.log1p(df['dislikes'])

# Reorder columns: identifiers, targets, then features
column_order = [
    # Identifiers
    'video_id', 'channel_id', 'published_at',
    
    # Targets (both raw and log)
    'dislikes', 'log_dislikes',
    
    # Features (alphabetically)
    'age', 'avg_compound', 'avg_neg', 'avg_neu', 'avg_pos',
    'comment_count', 'comment_sample_size', 'likes',
    'log_comment_count', 'log_likes', 'log_view_count',
    'no_comments', 'view_count', 'view_like_ratio'
]

df = df[column_order]

# Save
output_path = 'yt_dataset_filtered_with_log.csv'
df.to_csv(output_path, index=False)

print("=" * 70)
print("DATASET WITH LOG TARGET CREATED")
print("=" * 70)
print(f"\nOutput file: {output_path}")
print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"\nColumns:")
print(f"  - Identifiers: 3")
print(f"  - Targets: 2 (dislikes, log_dislikes)")
print(f"  - Features: 15")
print(f"\nTarget variable statistics:")
print(f"  dislikes:     mean={df['dislikes'].mean():,.2f}, median={df['dislikes'].median():,.2f}")
print(f"  log_dislikes: mean={df['log_dislikes'].mean():.2f}, median={df['log_dislikes'].median():.2f}")
print("\nReady for model training with log target!")
print("=" * 70)
