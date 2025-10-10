# scrape_metadata.py Usage Guide

## Overview
The `scrape_metadata.py` script fetches YouTube metadata (including duration and genre) for a specified range of videos from your dislike dataset and combines it with the existing CSV data.

## Features
- **Batch Processing**: Process any range of rows from the dataset
- **Multi-threading**: Parallel processing with configurable thread count
- **Rate Limiting**: Configurable delay between requests to avoid API limits
- **Resume Support**: Resume interrupted processing without needing to specify start-row
- **Error Handling**: Gracefully handles errors and continues processing
- **Automatic Error Logging**: All errors are logged to a timestamped text file
- **Interruption Recovery**: Displays exact command to resume after interruption or error

## Command-Line Arguments

### Required Arguments
- `--end-row END_ROW`: Ending row index (1-based, inclusive)
- Either `--start-row` OR `--resume-row` (one must be specified)

### Optional Arguments
- `--start-row START_ROW`: Starting row index (1-based, inclusive). Not required if using --resume-row
- `--delay DELAY`: Delay in seconds between requests (default: 1.0)
- `--threads THREADS`: Number of threads for parallel processing (default: 1)
- `--resume-row RESUME_ROW`: Resume from this row, appending to existing output file. When specified, --start-row is optional
- `--input INPUT`: Input CSV file (default: youtube_dislike_dataset.csv)
- `--output OUTPUT`: Output CSV file (default: combined_metadata.csv)

## Usage Examples

### Basic Usage
Process rows 1-100 with default settings (1 thread, 1 second delay):
```bash
python scrape_metadata.py --start-row 1 --end-row 100
```

### Multi-threaded Processing
Process 1000 rows using 5 threads with a 2-second delay:
```bash
python scrape_metadata.py --start-row 1 --end-row 1000 --threads 5 --delay 2
```

### Custom Output File
Save results to a specific file:
```bash
python scrape_metadata.py --start-row 1 --end-row 100 --output batch1_metadata.csv
```

### Resume Processing (No Start-Row Needed!)
If processing was interrupted at row 500, simply resume from there:
```bash
python scrape_metadata.py --resume-row 500 --end-row 1000 --output combined_metadata.csv
```
Note: When using `--resume-row`, you don't need to specify `--start-row`.

### Processing Large Dataset in Batches
Process in chunks to manage memory and handle interruptions:
```bash
# First batch (rows 1-1000)
python scrape_metadata.py --start-row 1 --end-row 1000 --output metadata_batch.csv --threads 3

# Second batch (resuming from 1001)
python scrape_metadata.py --resume-row 1001 --end-row 2000 --output metadata_batch.csv --threads 3

# Third batch (resuming from 2001)
python scrape_metadata.py --resume-row 2001 --end-row 3000 --output metadata_batch.csv --threads 3
```

## Output Format

The output CSV will contain:
- **Original CSV columns**: Prefixed with `csv_` (e.g., `csv_video_id`, `csv_title`)
- **YouTube video metadata**: Prefixed with `aiotube_video_` including:
  - `aiotube_video_id`
  - `aiotube_video_title`
  - `aiotube_video_duration`
  - `aiotube_video_genre`
  - `aiotube_video_views`
  - `aiotube_video_likes`
  - `aiotube_video_upload_date`
  - And more...
- **YouTube channel metadata**: Prefixed with `aiotube_channel_` including:
  - `aiotube_channel_id`
  - `aiotube_channel_name`
  - `aiotube_channel_subscribers`
  - `aiotube_channel_verified`
  - And more...
- **row_index**: The original row number from the input CSV

## Performance Tips

1. **Thread Count**: Start with 3-5 threads. Too many threads may trigger rate limits.
2. **Delay**: Use at least 1 second delay. Increase if you encounter rate limiting errors.
3. **Batch Size**: Process 100-1000 rows at a time for better progress tracking.
4. **Resume**: Always use `--resume-row` when continuing interrupted processing.
5. **Check Error Logs**: Review the error log file after each run to identify systematic issues.

## Error Handling & Recovery

### Automatic Error Logging
All errors are automatically logged to a text file with the same name as your output file but with `_errors.txt` suffix:
- Output: `combined_metadata.csv`
- Error log: `combined_metadata_errors.txt`

The error log includes:
- Timestamp of each error
- Row number where the error occurred
- Video ID (if available)
- Detailed error message

### Interruption Recovery
If the script is interrupted (Ctrl+C) or encounters a fatal error, it will display:
- Number of successfully processed rows
- Number of errors encountered
- Last successful row number
- **Exact command to resume processing** from where it left off

Example output on interruption:
```
================================================================================
Process interrupted by user!
================================================================================

Processed: 245, Errors: 5
Last successful row: 250

To resume from the next row, use:
  python scrape_metadata.py --resume-row 251 --end-row 1000 --output combined_metadata.csv

Error log: combined_metadata_errors.txt
================================================================================
```

Simply copy and paste the displayed command to resume!

## Notes

- **Row indices are 1-based** (row 1 is the first data row after the header)
- The script automatically detects video IDs from various column names
- Multi-threading writes results as they complete (not in order)
- Resume functionality appends to existing file, maintaining all previous data
- Error logs are cumulative when resuming - all errors are tracked in one file
- Failed video fetches are logged and skipped
- The script continues processing even if individual videos fail
