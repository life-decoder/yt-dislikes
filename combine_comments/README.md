# Combine Comments Script

## Overview

The `combine_comments.py` script combines multiple CSV files containing comment sentiment data into a single consolidated dataset. It automatically excludes unnecessary columns (`cid`, `author`, `votes`, `text`) that are not needed for analysis, keeping only the essential sentiment-related data.

## Features

- **Automatic File Discovery**: Processes all CSV files in the specified input folder
- **Alphabetical Processing**: Files are processed in alphabetical order by filename
- **Column Filtering**: Automatically removes specified columns to reduce file size
- **Error Handling**: Continues processing even if individual files encounter errors
- **Progress Reporting**: Provides detailed feedback during the combination process
- **Flexible Configuration**: Customizable input folder and output file paths

## Requirements

```bash
pip install pandas
```

## Usage

### Basic Usage

Run the script with default settings:

```bash
python combine_comments.py
```

This will:
- Look for CSV files in `../comments_datasets/`
- Combine all files
- Save output to `combined_comments_sentiment.csv` in the current directory

### Custom Input/Output Paths

Import and use the function in your own script:

```python
from combine_comments import combine_comments

# Combine files from a custom folder
combine_comments(
    input_folder='path/to/your/csv/files',
    output_file='custom_output.csv'
)
```

### Example: Absolute Paths

```python
from combine_comments import combine_comments

combine_comments(
    input_folder=r'C:\Users\YourName\Documents\comments',
    output_file=r'C:\Users\YourName\Documents\all_comments.csv'
)
```

### Example: Relative Paths

```python
from combine_comments import combine_comments

combine_comments(
    input_folder='./data/raw',
    output_file='./data/processed/combined.csv'
)
```

## Input File Structure

The script expects CSV files with the following structure:

```
cid, author, votes, text, video_id, sentiment, ...
```

### Excluded Columns

The following columns are automatically removed during combination:
- `cid` - Comment ID (not needed for aggregated analysis)
- `author` - Author name (not needed for sentiment analysis)
- `votes` - Vote count (optional, can be modified if needed)
- `text` - Comment text (large field, excluded to reduce file size)

### Retained Columns

All other columns are retained, typically including:
- `video_id` - YouTube video identifier
- `sentiment` - Sentiment score or classification
- Any other sentiment-related metrics

## Output

The script produces:
- A single combined CSV file with all data merged
- Progress messages showing:
  - Number of files found
  - Processing status for each file
  - Columns dropped
  - Row counts
  - Final combined statistics

### Sample Output

```
Found 5 CSV files to combine:
  - 00001-comments_sentiment.csv
  - 07501-comments_sentiment.csv
  - 15001-comments_sentiment.csv
  - 22501-comments_sentiment.csv
  - 30001_comments_sentiment.csv

Processing file 1/5: 00001-comments_sentiment.csv
  Dropped columns: ['cid', 'author', 'votes', 'text']
  Rows: 12,543

...

✓ Successfully combined 5 files
✓ Total rows: 58,231
✓ Columns: ['video_id', 'sentiment', 'score']
✓ Output file: combined_comments_sentiment.csv
```

## Directory Structure

Expected workspace structure:

```
combine_comments/
│   combine_comments.py
│   README.md
│
└── ../comments_datasets/
    │   00001-comments_sentiment.csv
    │   07501-comments_sentiment.csv
    │   15001-comments_sentiment.csv
    └   ...
```

## Modifying Excluded Columns

To change which columns are excluded, edit the `exclude_columns` list in the function:

```python
# Current exclusions
exclude_columns = ['cid', 'author', 'votes', 'text']

# Example: Keep 'votes' column
exclude_columns = ['cid', 'author', 'text']

# Example: Exclude additional columns
exclude_columns = ['cid', 'author', 'votes', 'text', 'timestamp']
```

## Error Handling

- If no CSV files are found, the script will display a message and exit gracefully
- If an individual file fails to process, the error is logged and the script continues with remaining files
- If all files fail to process, a "No data to combine!" message is displayed

## Performance Notes

- Files are processed sequentially to maintain order
- Memory usage depends on the total size of all CSV files
- For very large datasets (>1GB), consider processing in batches
- The excluded columns significantly reduce output file size

## Integration with Pipeline

This script is typically used after comment scraping and before model training:

1. **Scrape Comments** → `scrape_comments/scrape_comments.py`
2. **Combine Comments** → `combine_comments/combine_comments.py` ← **(This script)**
3. **Filter & Merge** → `filter_merge_data/filter_and_combine_data.py`
4. **Train Model** → Use combined dataset for training

## Troubleshooting

### Issue: "No CSV files found"
- Check that the input folder path is correct
- Verify that CSV files exist in the folder
- Ensure file extensions are `.csv` (lowercase)

### Issue: "Error processing file"
- Check that CSV files are not corrupted
- Verify that all files have consistent column names
- Ensure files are not locked/open in another program

### Issue: Memory errors with large files
- Process files in smaller batches
- Increase available system memory
- Consider using chunked reading for very large files

## License

Part of the YouTube Dislikes Prediction project.
