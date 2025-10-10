# Filter and Combine Data Script

## Overview
This script processes the YouTube dislike dataset and combines it with comment sentiment data to create a comprehensive dataset for analysis. It computes various metrics and ratios for each video record.

## Features

### Input Files
- **YouTube Dislike Dataset** (`youtube_dislike_dataset.csv`): Contains video metadata including likes, dislikes, views, etc.
- **Comments Sentiment Data** (`comments_sentiment_all.csv`): Contains sentiment analysis scores for video comments

### Output Fields

The script generates a CSV file with the following fields:

#### Basic Fields
- `video_id`: YouTube video ID
- `channel_id`: YouTube channel ID
- `published_at`: Publication date/time
- `view_count`: Number of views
- `likes`: Number of likes
- `dislikes`: Number of dislikes
- `comment_count`: Number of comments (from metadata)

#### Computed Ratio Fields
- `like_dislike_score`: Likes / (Likes + Dislikes) - smoothed by adding 1 to denominator if 0
- `ld_score_ohe`: One-Hot Encoded categorical version of like_dislike_score
  - `-1`: score < 0.33 (dislike-heavy)
  - `0`: 0.33 ≤ score ≤ 0.67 (balanced)
  - `1`: score > 0.67 (like-heavy)
- `view_like_ratio`: view_count / like_count (smoothed)
- `view_dislike_ratio`: view_count / dislike_count (smoothed)
- `dislike_like_ratio`: dislike_count / like_count (smoothed)
- `no_comments`: Binary flag (1 if no comments in dataset, 0 otherwise)

#### Comment Sentiment Fields
- `avg_pos`: Average positive sentiment score from comments
- `avg_neu`: Average neutral sentiment score from comments
- `avg_neg`: Average negative sentiment score from comments
- `avg_compound`: Average compound sentiment score from comments
- `comment_sample_size`: Number of comments used for sentiment calculation

### Division by Zero Handling
All ratio calculations add 1 to the denominator if it would be 0, preventing division by zero errors.

## Usage

### Basic Usage
```bash
python filter_and_combine_data.py
```

This uses default file paths:
- Input: `youtube_dislike_dataset.csv`
- Comments: `comments_sentiment_all.csv`
- Output: `filtered_combined_dataset.csv`

### Process All Records (Explicit)
```bash
python filter_and_combine_data.py --all
```

### Custom File Paths
```bash
python filter_and_combine_data.py --input dataset.csv --comments sentiment.csv --output output.csv
```

### Process Specific Row Range
```bash
# Process rows 0 to 10,000
python filter_and_combine_data.py --start-row 0 --end-row 10000

# Process from row 50,000 to end
python filter_and_combine_data.py --start-row 50000

# Process from beginning to row 100,000
python filter_and_combine_data.py --end-row 100000
```

### Use Parallel Processing
```bash
# Use 4 threads for parallel processing
python filter_and_combine_data.py --threads 4

# Use 8 threads with larger batch size
python filter_and_combine_data.py --threads 8 --batch-size 5000
```

### Combine Multiple Options
```bash
# Process specific range with parallel processing
python filter_and_combine_data.py --start-row 10000 --end-row 50000 --threads 4 --output chunk1.csv

# Process large dataset with optimal settings
python filter_and_combine_data.py --all --threads 8 --batch-size 10000 --output full_dataset.csv
```

## Command-Line Arguments

### Input/Output Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--input` | `youtube_dislike_dataset.csv` | Path to YouTube dislike dataset |
| `--comments` | `comments_sentiment_all.csv` | Path to comments sentiment data |
| `--output` | `filtered_combined_dataset.csv` | Path for output file |

### Row Range Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--all` | (flag) | Process all records (default behavior) |
| `--start-row` | None | Starting row index (0-based, inclusive) |
| `--end-row` | None | Ending row index (0-based, exclusive) |

**Note**: You can specify `--start-row` and/or `--end-row` to process a subset of the dataset. This is useful for:
- Processing data in chunks
- Testing on a small subset
- Resuming interrupted processing
- Parallel processing across multiple machines

### Performance Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--batch-size` | `1000` | Records to process before writing to disk |
| `--threads` | `1` | Number of threads for parallel processing |

**Performance Tips**:
- Increase `--batch-size` (e.g., 5000-10000) for large datasets to reduce I/O overhead
- Use `--threads` (e.g., 4-8) to speed up processing on multi-core systems
- Balance threads and batch size based on available memory

## Processing Details

1. **Loading Data**: Loads comments sentiment data and video metadata
2. **Row Filtering**: Applies start-row/end-row filters if specified
3. **Parallel/Sequential Processing**: 
   - Single-threaded (default): Processes records sequentially
   - Multi-threaded: Uses ThreadPoolExecutor for parallel processing
4. **Sentiment Aggregation**: For each video, finds all its comments and computes average sentiment
5. **Metric Calculation**: Computes all ratio and derived metrics
6. **Incremental Writing**: Writes results to disk in batches to handle large datasets

### Parallel Processing

When `--threads > 1`, the script uses Python's `ThreadPoolExecutor` to process multiple videos concurrently. This can significantly speed up processing, especially for large datasets where sentiment lookups are I/O-bound.

**Best Practices**:
- Start with 4 threads and increase based on performance
- Monitor memory usage - more threads = more memory
- Optimal thread count is usually 4-8 for most systems

## Output Statistics

The script provides detailed statistics including:
- Dataset shape and column list
- Videos with/without comments
- Like/Dislike score distribution across categories
- Average sentiment scores with standard deviation
- Ratio metrics with mean and median values

## Performance Considerations

- Uses batch processing to handle large datasets efficiently
- Incrementally writes to disk to minimize memory usage
- Only loads necessary columns from input files
- Progress updates every batch for long-running operations
- Supports parallel processing for faster execution on multi-core systems
- Row range filtering allows processing data in chunks

### Processing Large Datasets in Chunks

For very large datasets, you can process data in chunks and combine later:

```bash
# Process first 100k records
python filter_and_combine_data.py --start-row 0 --end-row 100000 --output chunk1.csv --threads 4

# Process next 100k records
python filter_and_combine_data.py --start-row 100000 --end-row 200000 --output chunk2.csv --threads 4

# Process remaining records
python filter_and_combine_data.py --start-row 200000 --output chunk3.csv --threads 4

# Combine chunks (using pandas)
# python -c "import pandas as pd; pd.concat([pd.read_csv(f'chunk{i}.csv') for i in range(1,4)]).to_csv('combined.csv', index=False)"
```

This approach:
- Reduces memory requirements
- Allows parallel processing on multiple machines
- Enables recovery from interruptions
- Facilitates testing on smaller subsets

## Example Output Summary

```
Dataset shape: 546,827 rows × 18 columns

Basic Statistics:
  • Videos with comments: 234,156 (42.8%)
  • Videos without comments: 312,671 (57.2%)
  • Average comment sample size: 23.4

Like/Dislike Score Distribution (OHE):
  • Category -1 (< 0.33): 12,345 (2.3%)
  • Category  0 (0.33-0.67): 45,678 (8.4%)
  • Category  1 (> 0.67): 488,804 (89.3%)

Sentiment Averages:
  • Positive:  0.1234 (±0.0987)
  • Neutral:   0.7654 (±0.1234)
  • Negative:  0.1112 (±0.0876)
  • Compound:  0.2345 (±0.3456)
```

## Error Handling

- Handles missing values by converting to 0
- Returns neutral sentiment (neu=1.0, others=0.0) for videos without comments
- Smooths all denominators to prevent division by zero
- Continues processing even if individual records have issues

## Notes

- Videos without comments in the sentiment dataset will have `no_comments=1` and neutral sentiment scores
- All ratio calculations are smoothed to handle edge cases
- The script is designed to work with large datasets (millions of records)
- Output file is created incrementally to avoid memory issues
