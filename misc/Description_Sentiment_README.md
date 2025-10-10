# analyze_description_sentiment.py Usage Guide

## Overview
The `analyze_description_sentiment.py` script performs sentiment analysis on video descriptions from the YouTube dislike dataset using NLTK's VADER (Valence Aware Dictionary and sEntiment Reasoner) sentiment analyzer. The script outputs the video ID along with sentiment scores to a CSV file.

## Features
- **VADER Sentiment Analysis**: Uses the industry-standard NLTK VADER for sentiment scoring
- **Batch Processing**: Process any range of rows from the dataset
- **Memory Efficient**: Uses chunked reading for large datasets
- **Resume Support**: Resume interrupted processing from any row
- **Progress Tracking**: Real-time progress updates every 1,000 videos
- **Error Handling**: Gracefully handles missing or empty descriptions
- **Flexible Output**: Append or overwrite modes supported

## Sentiment Scores Output
The script outputs four sentiment scores for each video description:
- **neg**: Negative sentiment score (0.0 to 1.0)
- **neu**: Neutral sentiment score (0.0 to 1.0)
- **pos**: Positive sentiment score (0.0 to 1.0)
- **compound**: Compound score (-1.0 to 1.0) - normalized, weighted composite score

### Interpreting Compound Score
- **Positive sentiment**: compound score >= 0.05
- **Neutral sentiment**: -0.05 < compound score < 0.05
- **Negative sentiment**: compound score <= -0.05

## Requirements
Install required packages:
```bash
pip install nltk pandas
```

The script will automatically download the VADER lexicon on first run if not already present.

## Command-Line Arguments

### Input/Output Options
- `--csv CSV`: Input CSV file with video data (default: `youtube_dislike_dataset.csv`)
- `--out OUT`: Output CSV file (default: `description_sentiment.csv`)

### Processing Range Options
- `--start-row N`: Starting row number (1-based, after header). Default: 1
- `--end-row N`: Ending row number (1-based, inclusive). If not specified, process to end
- `--all`: Process all rows in the dataset
- `--resume-row N`: Resume from this row (implies `--append`)

### Other Options
- `--append`: Append to output file if it exists (default: overwrite)
- `--chunk-size N`: Number of rows to process in each chunk (default: 10,000)

## Usage Examples

### Process First 100 Videos
```bash
python analyze_description_sentiment.py --start-row 1 --end-row 100
```

### Process All Videos in Dataset
```bash
python analyze_description_sentiment.py --all
```

### Process Specific Range
```bash
python analyze_description_sentiment.py --start-row 1000 --end-row 5000
```

### Process Large Batch with Custom Output
```bash
python analyze_description_sentiment.py --start-row 1 --end-row 10000 --out batch1_descriptions.csv
```

### Resume Interrupted Processing
If processing was interrupted at row 5000:
```bash
python analyze_description_sentiment.py --resume-row 5000 --end-row 10000
```

### Process in Batches
Process a large dataset in manageable chunks:
```bash
# First batch (1-10,000)
python analyze_description_sentiment.py --start-row 1 --end-row 10000 --out descriptions_all.csv

# Second batch (10,001-20,000) - append to same file
python analyze_description_sentiment.py --resume-row 10001 --end-row 20000 --out descriptions_all.csv

# Third batch (20,001-30,000) - append to same file
python analyze_description_sentiment.py --resume-row 20001 --end-row 30000 --out descriptions_all.csv
```

## Output Format

The output CSV contains the following columns:
- **video_id**: YouTube video ID
- **neg**: Negative sentiment score (0.0 to 1.0)
- **neu**: Neutral sentiment score (0.0 to 1.0)
- **pos**: Positive sentiment score (0.0 to 1.0)
- **compound**: Compound sentiment score (-1.0 to 1.0)

Example output:
```csv
video_id,neg,neu,pos,compound
dQw4w9WgXcQ,0.000,0.823,0.177,0.6369
jNQXAC9IVRw,0.125,0.750,0.125,0.2263
9bZkp7q19f0,0.000,1.000,0.000,0.0000
```

## Handling Missing Descriptions
Videos with missing or empty descriptions are assigned neutral sentiment scores:
- neg: 0.0
- neu: 1.0
- pos: 0.0
- compound: 0.0

## Performance Tips

1. **Chunk Size**: The default chunk size of 10,000 works well for most systems. Reduce if experiencing memory issues.
2. **Progress Tracking**: Progress is displayed every 1,000 videos processed.
3. **Resume Feature**: Always use `--resume-row` to continue from where you left off if interrupted.
4. **Batch Processing**: For very large datasets, process in batches of 10,000-50,000 rows.

## Error Handling & Recovery

### Keyboard Interruption
If you interrupt processing (Ctrl+C), the script will:
- Save all processed results to the output file
- Display the exact resume command to continue processing

### Missing video_id
Videos without a valid video_id are skipped and counted in the summary.

### Processing Errors
If an error occurs during processing:
- Partial results are saved to the output file
- Error details are displayed
- Use `--resume-row` to continue from the last successful row

## Integration with Other Scripts

This script complements the other sentiment analysis tools in the project:
- `scrape_comments.py`: Analyzes sentiment of video comments
- `analyze_description_sentiment.py`: Analyzes sentiment of video descriptions (this script)

You can combine the results from both scripts to get a comprehensive view of sentiment for each video.

## Example Workflow

```bash
# 1. Analyze descriptions for the first 1000 videos
python analyze_description_sentiment.py --start-row 1 --end-row 1000 --out desc_sentiment.csv

# 2. Check the output
head -n 5 desc_sentiment.csv

# 3. Continue processing the next batch
python analyze_description_sentiment.py --resume-row 1001 --end-row 2000 --out desc_sentiment.csv

# 4. Load and analyze results in Python
# import pandas as pd
# df = pd.read_csv('desc_sentiment.csv')
# print(df.describe())
# print(f"Positive descriptions: {(df['compound'] >= 0.05).sum()}")
# print(f"Negative descriptions: {(df['compound'] <= -0.05).sum()}")
# print(f"Neutral descriptions: {((df['compound'] > -0.05) & (df['compound'] < 0.05)).sum()}")
```

## About VADER Sentiment Analysis

VADER (Valence Aware Dictionary and sEntiment Reasoner) is a lexicon and rule-based sentiment analysis tool specifically attuned to sentiments expressed in social media. It is particularly effective for:
- Short texts like social media posts
- Texts with informal language, emoticons, and slang
- Mixed sentiment expressions

The compound score is the most useful metric for single-dimensional sentiment measurement, as it normalizes the scores into a single value between -1 (most negative) and +1 (most positive).

## Troubleshooting

### VADER Lexicon Not Found
If you see an error about the VADER lexicon:
```bash
python -c "import nltk; nltk.download('vader_lexicon')"
```

### Memory Issues
If processing fails due to memory constraints:
- Reduce `--chunk-size` to 5000 or lower
- Process smaller batches (e.g., 5,000-10,000 rows at a time)

### Encoding Issues
The script uses UTF-8 encoding. If you encounter encoding errors with the input CSV, ensure your file is UTF-8 encoded.

## References
- VADER Sentiment Analysis: https://github.com/cjhutto/vaderSentiment
- NLTK Documentation: https://www.nltk.org/
