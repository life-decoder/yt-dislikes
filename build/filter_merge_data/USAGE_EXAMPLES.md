# Filter and Combine Data - Usage Examples

## Quick Reference

### Basic Commands

```bash
# Process all records with defaults
python filter_and_combine_data.py

# Process all records explicitly
python filter_and_combine_data.py --all

# Custom input/output paths
python filter_and_combine_data.py --input my_dataset.csv --output results.csv
```

### Row Range Processing

```bash
# First 10,000 records
python filter_and_combine_data.py --start-row 0 --end-row 10000

# Records 10,000 to 20,000
python filter_and_combine_data.py --start-row 10000 --end-row 20000

# From row 50,000 to end
python filter_and_combine_data.py --start-row 50000

# First 100,000 records only
python filter_and_combine_data.py --end-row 100000
```

### Parallel Processing

```bash
# Use 4 threads
python filter_and_combine_data.py --threads 4

# Use 8 threads with larger batch size
python filter_and_combine_data.py --threads 8 --batch-size 5000

# Maximum performance setup
python filter_and_combine_data.py --threads 8 --batch-size 10000
```

### Combined Options

```bash
# Process subset with parallel processing
python filter_and_combine_data.py \
    --start-row 0 \
    --end-row 50000 \
    --threads 4 \
    --batch-size 2000 \
    --output subset_results.csv

# Full processing with optimal settings
python filter_and_combine_data.py \
    --all \
    --input youtube_dislike_dataset.csv \
    --comments comments_sentiment_all.csv \
    --output full_results.csv \
    --threads 8 \
    --batch-size 10000
```

## Common Scenarios

### Testing on Small Sample

```bash
# Test with first 1,000 records
python filter_and_combine_data.py --end-row 1000 --output test_sample.csv
```

### Processing in Chunks (for very large datasets)

```bash
# Chunk 1: Records 0-250,000
python filter_and_combine_data.py --start-row 0 --end-row 250000 --output chunk1.csv --threads 4

# Chunk 2: Records 250,000-500,000
python filter_and_combine_data.py --start-row 250000 --end-row 500000 --output chunk2.csv --threads 4

# Chunk 3: Records 500,000-750,000
python filter_and_combine_data.py --start-row 500000 --end-row 750000 --output chunk3.csv --threads 4

# Chunk 4: Records 750,000 to end
python filter_and_combine_data.py --start-row 750000 --output chunk4.csv --threads 4
```

### Resuming Interrupted Processing

If processing was interrupted at row 150,000:

```bash
# Resume from where it left off
python filter_and_combine_data.py --start-row 150000 --output resume_results.csv --threads 4
```

### Optimizing for Speed vs Memory

```bash
# Speed optimized (uses more memory)
python filter_and_combine_data.py --threads 8 --batch-size 20000

# Memory optimized (slower but uses less memory)
python filter_and_combine_data.py --threads 2 --batch-size 500

# Balanced approach
python filter_and_combine_data.py --threads 4 --batch-size 5000
```

## All Available Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--input` | string | `youtube_dislike_dataset.csv` | Input dislike dataset path |
| `--comments` | string | `comments_sentiment_all.csv` | Comments sentiment data path |
| `--output` | string | `filtered_combined_dataset.csv` | Output file path |
| `--all` | flag | - | Explicitly process all records |
| `--start-row` | integer | None | Starting row (0-based, inclusive) |
| `--end-row` | integer | None | Ending row (0-based, exclusive) |
| `--batch-size` | integer | `1000` | Batch size for writing |
| `--threads` | integer | `1` | Number of parallel threads |

## Performance Guidelines

### Thread Count Recommendations

- **1-2 threads**: Low memory systems or small datasets
- **4 threads**: Good balance for most systems
- **8+ threads**: High-performance systems with large datasets

### Batch Size Recommendations

- **500-1000**: Conservative, low memory usage
- **2000-5000**: Balanced approach
- **10000+**: High performance, requires more memory

### Row Range Strategy

For datasets with 1+ million records:
- Test with `--end-row 1000` first
- Process in 100k-500k record chunks
- Use 4-8 threads per chunk
- Combine results afterward

## Troubleshooting

### Out of Memory Error

```bash
# Reduce batch size and threads
python filter_and_combine_data.py --threads 2 --batch-size 500
```

### Slow Processing

```bash
# Increase threads and batch size
python filter_and_combine_data.py --threads 8 --batch-size 10000
```

### Need to Process Subset

```bash
# Use row range
python filter_and_combine_data.py --start-row 10000 --end-row 20000
```

## Tips

1. **Start Small**: Test with `--end-row 1000` before processing full dataset
2. **Monitor Resources**: Watch CPU and memory usage to optimize thread count
3. **Chunk Large Datasets**: Break into manageable pieces for better control
4. **Save Intermediate Results**: Use descriptive output filenames for chunks
5. **Document Your Process**: Keep notes on row ranges processed
