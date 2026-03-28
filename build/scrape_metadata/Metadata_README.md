# scrape_metadata.py Usage Guide

## Overview
The `scrape_metadata.py` script fetches additional YouTube metadata (duration and genre) for videos in your dataset. It's designed to supplement your existing dataset with these two specific fields that may be missing.

## Features
- **Batch Processing**: Process any range of rows from the dataset
- **Process All Records**: Use `--all` flag to process entire dataset
- **Multi-threading**: Parallel processing with configurable thread count
- **Rate Limiting**: Configurable delay between requests to avoid API limits
- **Resume Support**: Resume interrupted processing from any row
- **Comprehensive Error Logging**: All errors (missing metadata, failed fetches, etc.) are logged
- **Interruption Recovery**: Clean exit with Ctrl+C showing exact resume command
- **Missing Data Tracking**: Logs when duration or genre fields are absent

## Command-Line Arguments

### Required Arguments
You must specify EITHER:
- `--end-row END_ROW`: Ending row index (1-based, inclusive), OR
- `--all`: Process all records until the end of the dataset

### Optional Arguments
- `--start-row START_ROW`: Starting row index (1-based, inclusive). Defaults to 1 if not specified
- `--resume-row RESUME_ROW`: Resume from this row, appending to existing output file
- `--delay DELAY`: Delay in seconds between requests (default: 1.0)
- `--threads THREADS`: Number of threads for parallel processing (default: 1)
- `--input INPUT`: Input CSV file (default: ../yt_dataset_en_v3.csv)
- `--output OUTPUT`: Output CSV file (default: additional_metadata.csv)

## Usage Examples

### Basic Usage
Process rows 1-100 with default settings (1 thread, 1 second delay):
```bash
python scrape_metadata.py --start-row 1 --end-row 100
```

### Process All Records
Process the entire dataset from beginning to end:
```bash
python scrape_metadata.py --all
```

### Process All Records Starting from a Specific Row
```bash
python scrape_metadata.py --start-row 100 --all
```

### Multi-threaded Processing
Process 1000 rows using 5 threads with a 2-second delay:
```bash
python scrape_metadata.py --start-row 1 --end-row 1000 --threads 5 --delay 2
```

### Custom Output File
Save results to a specific file:
```bash
python scrape_metadata.py --start-row 1 --end-row 100 --output my_metadata.csv
```

### Resume Processing After Interruption
If processing was interrupted at row 500, resume from there:
```bash
python scrape_metadata.py --resume-row 500 --end-row 1000 --output additional_metadata.csv
```

### Resume and Process Until End
```bash
python scrape_metadata.py --resume-row 500 --all
```

### Processing Large Dataset in Batches
Process in chunks to manage memory and handle interruptions:
```bash
# First batch (rows 1-1000)
python scrape_metadata.py --start-row 1 --end-row 1000 --output metadata.csv --threads 3

# Second batch (resuming from 1001)
python scrape_metadata.py --resume-row 1001 --end-row 2000 --output metadata.csv --threads 3

# Or continue processing everything remaining
python scrape_metadata.py --resume-row 2001 --all --output metadata.csv --threads 3
```

## Output Format

The output CSV will contain:
- **video_id**: The YouTube video ID
- **duration**: Video duration in seconds (or None if unavailable)
- **genre**: Video genre as a string (e.g., "Music", "Gaming", "Entertainment", or None if unavailable)
- **row_index**: The original row number from the input CSV (1-based)

Example output:
```csv
video_id,duration,genre,row_index
0l3-iufiywU,245,Music,1
JXzk8G9aXI8,180,Entertainment,2
F1MAuOsOlqw,320,Gaming,3
```

## Performance Tips

1. **Thread Count**: Start with 3-5 threads. Too many threads may trigger rate limits.
2. **Delay**: Use at least 1 second delay. Increase if you encounter rate limiting errors.
3. **Batch Size**: Process 100-1000 rows at a time for better progress tracking, or use `--all` for convenience.
4. **Resume**: Always use `--resume-row` when continuing interrupted processing.
5. **Check Error Logs**: Review the error log file after each run to identify systematic issues.

## Error Handling & Recovery

### Comprehensive Error Logging
All errors are automatically logged to a text file with the same name as your output file but with `_errors.txt` suffix:
- Output: `additional_metadata.csv`
- Error log: `additional_metadata_errors.txt`

The error log tracks:
- **Timestamp** of each error
- **Row number** where the error occurred
- **Video ID** (if available)
- **Detailed error message**

Types of errors logged:
- Invalid or missing video IDs
- Failed metadata fetches from YouTube
- Missing duration metadata
- Missing genre metadata
- Any unexpected processing errors

### Interruption Recovery (Ctrl+C)
If you interrupt the script with Ctrl+C, it will:
- Save all successfully processed videos to the output file
- Display detailed progress summary
- Show the exact command to resume from where you left off

Example output on interruption:
```
================================================================================
Process interrupted by user (Ctrl+C)
================================================================================

[PROGRESS SUMMARY]
  * Successfully processed: 245 videos
  * Errors encountered: 5
  * Last successful row: 250

[OUTPUT] Output saved to: additional_metadata.csv
[WARNING] Errors logged to: additional_metadata_errors.txt

[RESUME] To resume from where you left off, run:
  python scrape_metadata.py --resume-row 251 --end-row 1000 --output additional_metadata.csv

================================================================================
```

Simply copy and paste the displayed command to continue!

### Handling Missing Metadata
When duration or genre is unavailable:
- The field is set to `None` (NULL in the CSV)
- An error is logged to the error log file
- Processing continues with the next video
- The video ID and row_index are still saved

## Notes

- **Row indices are 1-based** (row 1 is the first data row after the header)
- **Genres are stored as strings** (e.g., "Music", "Gaming", "Entertainment", "Education")
- The script automatically detects video IDs from various column names
- Multi-threading writes results as they complete (not necessarily in order)
- Resume functionality appends to existing file, maintaining all previous data
- Error logs are cumulative when resuming - all errors are tracked in one file
- Failed video fetches still create output rows with NULL values
- The script continues processing even if individual videos fail
- Press **Ctrl+C at any time** to safely stop and save progress

## Dependencies

The script requires:
- `pandas`: For CSV handling
- `aiotube`: For fetching YouTube metadata

Install with:
```bash
pip install pandas aiotube
```
