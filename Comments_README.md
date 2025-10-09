# YouTube Comment Scraper with Sentiment Analysis

A Python tool that scrapes comments from YouTube videos and performs VADER sentiment analysis on each comment. Perfect for analyzing audience reactions, content performance, and understanding viewer sentiment at scale.

## Features

- 📊 **Batch Processing**: Process single or multiple videos from a CSV file
- 🎯 **Sentiment Analysis**: Uses NLTK's VADER for accurate sentiment scoring
- 🔄 **Resume Capability**: Gracefully handle interruptions and resume from where you left off
- 🧵 **Multi-threading**: Optional parallel processing with configurable workers
- ⏱️ **Rate Limiting**: Built-in delay between requests to avoid API throttling
- 🛡️ **Error Handling**: Fail-fast on fetch errors with clear recovery instructions
- ⌨️ **Graceful Interruption**: Ctrl+C support with automatic resume information

## Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Install Dependencies

```bash
pip install -r requirements.txt
```

Or install packages individually:

```bash
pip install youtube-comment-downloader nltk
```

The program will automatically download the VADER lexicon on first run if not present.

## Usage

### Basic Usage

Process a single video (default):
```bash
python scrape.py
```

### Common Examples

**Process 10 videos with 200 comments each:**
```bash
python scrape.py --limit 10 --max-comments 200
```

**Process all videos in the CSV:**
```bash
python scrape.py --all
```

**Use custom input/output files:**
```bash
python scrape.py --csv my_videos.csv --out my_output.csv
```

**Multi-threaded processing with rate limiting:**
```bash
python scrape.py --workers 4 --delay 2.0 --all
```

**Resume from a specific row (after interruption):**
```bash
python scrape.py --resume-row 50
```

**Start from row 25 and append to existing output:**
```bash
python scrape.py --start-row 25 --append
```

### Advanced Row Control & Resume Options

**Process 50 videos starting from row 100:**
```bash
python scrape.py --start-row 100 --limit 50
```
This starts at row 100 (1-based, after the header) and processes the next 50 videos.

**Process a specific range of rows (e.g., rows 200-299):**
```bash
python scrape.py --start-row 200 --end-row 299
```
Processes exactly rows 200 through 299 (inclusive). This is more precise than using `--limit`.

**Process videos from row 200 to 299 and append to existing file:**
```bash
python scrape.py --start-row 200 --end-row 299 --append
```
Use `--append` to preserve existing data when continuing a previous run.

**Resume after an error at row 456:**
```bash
python scrape.py --resume-row 456
```
This automatically sets `--start-row 456` and `--append` mode to continue safely.

**Resume from row 456 and process up to row 500:**
```bash
python scrape.py --resume-row 456 --end-row 500
```
Combines resume functionality with an end boundary for precise range control.

**Process remaining videos after row 500:**
```bash
python scrape.py --start-row 501 --all
```
The `--all` flag ignores `--limit` and processes everything from row 501 onwards.

## Command-Line Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--all` | flag | `False` | Process all videos in the CSV file |
| `--limit` | int | `1` | Number of videos to process (ignored if `--all` is used) |
| `--max-comments` | int | `50` | Maximum number of comments to fetch per video |
| `--csv` | string | `youtube_dislike_dataset.csv` | Path to the input CSV containing video IDs |
| `--out` | string | `comments_sentiment_all.csv` | Path to the output CSV file |
| `--append` | flag | `False` | Append to output file if it exists (default overwrites) |
| `--workers` | int | `1` | Number of worker threads to use for parallel processing |
| `--delay` | float | `1.0` | Delay in seconds between requests (helps avoid rate limiting) |
| `--start-row` | int | `1` | 1-based CSV data row to start from (after header) |
| `--end-row` | int | `None` | 1-based CSV data row to end at (inclusive). Must be >= start-row |
| `--resume-row` | int | `None` | Shorthand for resuming: equivalent to `--start-row N --append` |

## Input Format

The input CSV should contain YouTube video IDs or URLs. The program automatically detects:

- **Column names**: `video_id`, `videoId`, `Video ID`, `video.id`, `yt_id`, `youtube_id`, `id`, `Id`, `ID`, etc.
- **Video ID formats**:
  - Plain 11-character IDs: `dQw4w9WgXcQ`
  - Full URLs: `https://www.youtube.com/watch?v=dQw4w9WgXcQ`
  - Short URLs: `https://youtu.be/dQw4w9WgXcQ`

**Example CSV:**
```csv
video_id,title,views
dQw4w9WgXcQ,Example Video 1,1000000
jNQXAC9IVRw,Example Video 2,500000
```

## Output Format

The output CSV contains the following columns:

| Column | Description |
|--------|-------------|
| `video_id` | YouTube video ID |
| `comment_index` | Sequential number of the comment (1-based) |
| `cid` | Comment ID from YouTube |
| `author` | Comment author's username |
| `time` | Timestamp of the comment |
| `votes` | Number of likes/votes on the comment |
| `text` | Comment text (newlines replaced with spaces) |
| `pos` | Positive sentiment score (0.0 to 1.0) |
| `neu` | Neutral sentiment score (0.0 to 1.0) |
| `neg` | Negative sentiment score (0.0 to 1.0) |
| `compound` | Overall sentiment score (-1.0 to 1.0) |

### Understanding Sentiment Scores

- **Positive (pos)**: Proportion of positive sentiment (0.0 = none, 1.0 = maximum)
- **Neutral (neu)**: Proportion of neutral sentiment
- **Negative (neg)**: Proportion of negative sentiment
- **Compound**: Normalized, weighted composite score
  - `>= 0.05`: Positive sentiment
  - `-0.05 to 0.05`: Neutral sentiment
  - `<= -0.05`: Negative sentiment

## Error Handling & Recovery

The script provides robust error handling and multiple ways to resume your work after interruptions or errors.

### Understanding Row Numbers

- **Row numbers are 1-based** and count data rows **after the header**
- Row 1 = first video in your CSV (the row immediately after the header)
- Row 50 = the 50th video in your CSV
- The script always displays the row number when processing each video

### Interruption (Ctrl+C)

Press `Ctrl+C` to gracefully stop the program. It will:
- Cancel pending tasks
- Save progress up to the last completed video
- Display exact command to resume

**Example output:**
```
^C
Interrupted by user (Ctrl+C). Stopping gracefully...
Processed 25 video(s) before interruption.
To resume, use: --start-row 30025 --append
```

### Fetch Errors

If a video fails to fetch (private, deleted, rate-limited, or unavailable):
- The program **stops immediately** to prevent wasting resources
- Displays the problematic row number
- Provides the exact command to resume from that row

**Example error output:**
```
[10] ABC123xyz (row 150): fetch_error: Video unavailable
Stopping due to fetch error. Re-run with --start-row 150 to resume from this row.
```

**How to resume after an error:**

**Option 1 - Using `--resume-row` (recommended):**
```bash
python scrape.py --resume-row 150
```
This is the easiest way - it automatically appends to your existing file.

**Option 2 - Using `--start-row` with `--append`:**
```bash
python scrape.py --start-row 150 --append
```
Equivalent to Option 1, but more explicit.

**Option 3 - Skip the problematic video:**
```bash
python scrape.py --start-row 151 --append
```
If you want to skip row 150 entirely, start from 151.

### Resume Strategies

**Strategy 1: Process in batches with error recovery**
```bash
# Process first 100 videos
python scrape.py --limit 100

# If error at row 45, resume from there
python scrape.py --resume-row 45 --limit 100

# Continue with next batch
python scrape.py --start-row 101 --limit 100 --append
```

**Strategy 2: Process all with periodic manual stops**
```bash
# Start processing all videos
python scrape.py --all

# Press Ctrl+C after some time, output shows:
# "To resume, use: --start-row 523 --append"

# Resume from where you left off
python scrape.py --start-row 523 --all
```

**Strategy 3: Targeted range processing with --end-row**
```bash
# Process exactly rows 1000-1099 (more precise than --limit)
python scrape.py --start-row 1000 --end-row 1099

# If error at row 1042, resume just that specific segment
python scrape.py --resume-row 1042 --end-row 1099

# Process in exact batches
python scrape.py --start-row 1 --end-row 500
python scrape.py --start-row 501 --end-row 1000 --append
python scrape.py --start-row 1001 --end-row 1500 --append
```

### Common Resume Scenarios

**Scenario: Script crashed at row 234**
```bash
python scrape.py --resume-row 234
```

**Scenario: Want to reprocess row 234 with different settings**
```bash
# First, manually remove row 234's data from output CSV
# Then resume from that row
python scrape.py --start-row 234 --max-comments 200 --append
```

**Scenario: Processed rows 1-500, want to continue with rest**
```bash
python scrape.py --start-row 501 --all
```

**Scenario: Testing specific problematic rows**
```bash
# Test just row 456
python scrape.py --start-row 456 --limit 1

# If it works, continue from there
python scrape.py --resume-row 456 --limit 100

# Or test a specific range
python scrape.py --start-row 456 --end-row 460
```

**Scenario: Split large job into manageable chunks**
```bash
# Process in chunks of 100 rows each
python scrape.py --start-row 1 --end-row 100
python scrape.py --start-row 101 --end-row 200 --append
python scrape.py --start-row 201 --end-row 300 --append
# etc...
```

## Best Practices

### 1. Start Small
Always test with a single video first:
```bash
python scrape.py --limit 1
```

### 2. Use Rate Limiting
Keep the default 1-second delay or increase it to avoid throttling:
```bash
python scrape.py --delay 2.0
```

### 3. Single Thread Recommended
Default `--workers 1` is safest for API stability. Only increase if you experience no issues:
```bash
python scrape.py --workers 1  # Safest
python scrape.py --workers 4  # Only if stable
```

### 4. Resume on Errors
Always use `--resume-row` or `--append` when resuming to preserve existing data:
```bash
# Simplest way to resume
python scrape.py --resume-row 100

# Or explicitly with append
python scrape.py --start-row 100 --append
```

**Important**: Using `--start-row` without `--append` will **overwrite** your output file!

### 5. Control Your Processing Range
Use `--end-row` for exact range control, or `--limit` for a count:
```bash
# Process exactly rows 200-249 (50 videos)
python scrape.py --start-row 200 --end-row 249 --append

# Or use --limit to process 50 videos starting from row 200
python scrape.py --start-row 200 --limit 50 --append

# Process all remaining videos from row 1000 onwards
python scrape.py --start-row 1000 --all
```

**Tip**: `--end-row` is more precise when you know exact row numbers; `--limit` is easier when you just want a count.

### 6. Monitor Progress
The program displays real-time progress:
```
[1] dQw4w9WgXcQ (row 1): saved 50 comments to comments_sentiment_all.csv
[2] jNQXAC9IVRw (row 2): saved 48 comments to comments_sentiment_all.csv
```

## Troubleshooting

### Issue: "youtube-comment-downloader is not installed"
**Solution:**
```bash
pip install youtube-comment-downloader
```

### Issue: Rate limiting / HTTP 429 errors
**Solution:** Increase delay and reduce workers:
```bash
python scrape.py --delay 3.0 --workers 1
```

### Issue: Some videos have no comments
**Expected behavior.** The program will process them but write 0 rows. Check the output for:
```
[5] ABC123xyz (row 150): saved 0 comments to comments_sentiment_all.csv
```

### Issue: CSV encoding problems
The program tries `utf-8-sig` and `utf-8` automatically. If issues persist, ensure your CSV is saved with UTF-8 encoding.

## Performance Tips

- **Batch size**: Process 100-500 videos at a time for manageable sessions
- **Comment limit**: 50-100 comments per video is usually sufficient for sentiment analysis
- **Delay**: 1-2 seconds works well; increase if you encounter rate limits
- **Workers**: Start with 1, only increase if stable and needed

## Example Workflow

### Full Processing Pipeline

1. **Test with one video:**
   ```bash
   python scrape.py --limit 1 --max-comments 50
   ```

2. **Process first batch:**
   ```bash
   python scrape.py --limit 100 --max-comments 50 --out batch1.csv
   ```

3. **Resume if interrupted:**
   ```bash
   python scrape.py --resume-row 50 --out batch1.csv
   ```

4. **Process remaining videos:**
   ```bash
   python scrape.py --start-row 101 --all --out batch2.csv
   ```

## License

This project uses the following open-source libraries:
- [youtube-comment-downloader](https://github.com/egbertbouman/youtube-comment-downloader)
- [NLTK](https://www.nltk.org/)

## Contributing

Contributions are welcome! Please ensure:
- Code follows existing style
- New features include documentation
- Error handling is comprehensive

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review command-line options with `python scrape.py --help`
3. Open an issue on the repository

---

**Note**: This tool is for educational and research purposes. Please respect YouTube's Terms of Service and rate limits when using this scraper.
