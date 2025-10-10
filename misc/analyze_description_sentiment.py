import csv
import os
import argparse
import pandas as pd
from typing import Optional

# Sentiment analysis
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download VADER lexicon if not already present
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    print("Downloading VADER lexicon...")
    nltk.download('vader_lexicon', quiet=True)


def analyze_description_sentiment(video_id: str, description: str, sia: SentimentIntensityAnalyzer) -> dict:
    """
    Analyze sentiment of a video description using VADER.
    
    Args:
        video_id: YouTube video ID
        description: Video description text
        sia: SentimentIntensityAnalyzer instance
        
    Returns:
        Dictionary with video_id and sentiment scores (neg, neu, pos, compound)
    """
    # Handle missing or empty descriptions
    if pd.isna(description) or not str(description).strip():
        return {
            'video_id': video_id,
            'neg': 0.0,
            'neu': 1.0,
            'pos': 0.0,
            'compound': 0.0
        }
    
    # Calculate sentiment scores
    scores = sia.polarity_scores(str(description))
    
    return {
        'video_id': video_id,
        'neg': scores['neg'],
        'neu': scores['neu'],
        'pos': scores['pos'],
        'compound': scores['compound']
    }


def main():
    parser = argparse.ArgumentParser(
        description="Analyze video description sentiment using NLTK VADER",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process first 100 videos
  python analyze_description_sentiment.py --start-row 1 --end-row 100

  # Process all videos in the dataset
  python analyze_description_sentiment.py --all

  # Process specific range with custom input/output files
  python analyze_description_sentiment.py --start-row 1000 --end-row 2000 --csv my_dataset.csv --out my_output.csv

  # Resume processing from row 5000
  python analyze_description_sentiment.py --resume-row 5000 --end-row 10000
        """
    )
    
    parser.add_argument('--csv', type=str, default='youtube_dislike_dataset.csv',
                        help='Input CSV file with video data (default: youtube_dislike_dataset.csv)')
    parser.add_argument('--out', type=str, default='description_sentiment.csv',
                        help='Output CSV file (default: description_sentiment.csv)')
    parser.add_argument('--start-row', type=int, default=1,
                        help='Starting row number (1-based, after header). Default: 1')
    parser.add_argument('--end-row', type=int,
                        help='Ending row number (1-based, inclusive). If not specified, process to end.')
    parser.add_argument('--all', action='store_true',
                        help='Process all rows in the dataset')
    parser.add_argument('--append', action='store_true',
                        help='Append to output file if it exists (default: overwrite)')
    parser.add_argument('--resume-row', type=int,
                        help='Resume from this row (implies --append)')
    parser.add_argument('--chunk-size', type=int, default=10000,
                        help='Number of rows to process in each chunk (default: 10000)')
    
    args = parser.parse_args()
    
    # Handle resume-row
    if args.resume_row:
        args.start_row = args.resume_row
        args.append = True
        print(f"Resuming from row {args.resume_row} (appending to output)...")
    
    # Check if input file exists
    if not os.path.exists(args.csv):
        print(f"ERROR: Input file '{args.csv}' not found!")
        return
    
    # Initialize VADER sentiment analyzer
    print("Initializing VADER sentiment analyzer...")
    sia = SentimentIntensityAnalyzer()
    
    # Determine write mode
    write_mode = 'a' if args.append else 'w'
    write_header = not (args.append and os.path.exists(args.out))
    
    print(f"Reading from: {args.csv}")
    print(f"Writing to: {args.out}")
    print(f"Mode: {'Append' if args.append else 'Overwrite'}")
    
    # Open output file
    with open(args.out, write_mode, newline='', encoding='utf-8') as out_file:
        fieldnames = ['video_id', 'neg', 'neu', 'pos', 'compound']
        writer = csv.DictWriter(out_file, fieldnames=fieldnames)
        
        if write_header:
            writer.writeheader()
        
        # Process CSV in chunks
        total_processed = 0
        total_skipped = 0
        
        # Calculate which rows to read
        nrows = None
        
        if args.end_row:
            if args.end_row < args.start_row:
                print(f"ERROR: end-row ({args.end_row}) must be >= start-row ({args.start_row})")
                return
            nrows = args.end_row - args.start_row + 1
        
        print(f"\nProcessing rows {args.start_row} to {args.end_row if args.end_row else 'end'}...")
        print("="*70)
        
        try:
            # Read in chunks for memory efficiency
            # Build read_csv arguments conditionally
            read_csv_kwargs = {
                'chunksize': args.chunk_size,
                'dtype': {'video_id': str, 'description': str},
                'na_values': ['', 'NA', 'N/A', 'null', 'None'],
                'keep_default_na': True
            }
            
            # Only add skiprows if we're not starting from row 1
            if args.start_row > 1:
                read_csv_kwargs['skiprows'] = range(1, args.start_row)
            
            # Only add nrows if end_row is specified
            if nrows is not None:
                read_csv_kwargs['nrows'] = nrows
            
            chunk_iter = pd.read_csv(args.csv, **read_csv_kwargs)
            
            for chunk_num, chunk in enumerate(chunk_iter, start=1):
                chunk_start = args.start_row + (chunk_num - 1) * args.chunk_size
                
                for idx, row in chunk.iterrows():
                    video_id = row.get('video_id')
                    description = row.get('description')
                    
                    # Skip if video_id is missing
                    if pd.isna(video_id) or not str(video_id).strip():
                        total_skipped += 1
                        continue
                    
                    # Analyze sentiment
                    result = analyze_description_sentiment(video_id, description, sia)
                    writer.writerow(result)
                    
                    total_processed += 1
                    
                    # Progress update every 1000 videos
                    if total_processed % 1000 == 0:
                        current_row = chunk_start + (idx % args.chunk_size)
                        print(f"Processed {total_processed:,} videos (current row: ~{current_row:,})...")
            
            print("="*70)
            print(f"\nProcessing complete!")
            print(f"Total videos processed: {total_processed:,}")
            print(f"Total videos skipped: {total_skipped:,}")
            print(f"Output saved to: {args.out}")
            
        except KeyboardInterrupt:
            print("\n\nProcessing interrupted by user!")
            print(f"Processed {total_processed:,} videos before interruption.")
            print(f"Output saved to: {args.out}")
            print(f"\nTo resume, run:")
            resume_row = args.start_row + total_processed
            print(f"python analyze_description_sentiment.py --resume-row {resume_row} --end-row {args.end_row if args.end_row else '[END]'} --out {args.out}")
            
        except Exception as e:
            print(f"\nERROR: {e}")
            print(f"Processed {total_processed:,} videos before error.")
            print(f"Partial output saved to: {args.out}")
            raise


if __name__ == '__main__':
    main()
