import pandas as pd
import numpy as np
import argparse
from typing import Optional
import warnings
from datetime import datetime
import os

warnings.filterwarnings('ignore')


def categorize_like_dislike_score(score: float) -> int:
    """
    Convert Like_Dislike score to categorical value.
    
    Args:
        score: Like_Dislike ratio (0 to 1)
        
    Returns:
        -1 if score < 0.33
        0 if 0.33 <= score <= 0.67
        1 if score > 0.67
    """
    if score < 0.33:
        return -1
    elif score <= 0.67:
        return 0
    else:
        return 1


def compute_average_sentiment(comments_df: pd.DataFrame, video_id: str) -> dict:
    """
    Compute average sentiment scores for a specific video from its comments.
    
    Args:
        comments_df: DataFrame containing comment sentiment data
        video_id: YouTube video ID
        
    Returns:
        Dictionary with average sentiment scores (pos, neu, neg, compound)
    """
    # Filter comments for this video
    video_comments = comments_df[comments_df['video_id'] == video_id]
    
    if len(video_comments) == 0:
        # No comments found - return neutral sentiment
        return {
            'avg_pos': 0.0,
            'avg_neu': 1.0,
            'avg_neg': 0.0,
            'avg_compound': 0.0,
            'comment_sample_size': 0
        }
    
    # Calculate averages
    return {
        'avg_pos': video_comments['pos'].mean(),
        'avg_neu': video_comments['neu'].mean(),
        'avg_neg': video_comments['neg'].mean(),
        'avg_compound': video_comments['compound'].mean(),
        'comment_sample_size': len(video_comments)
    }


def process_video_record(row: pd.Series, comments_df: pd.DataFrame) -> dict:
    """
    Process a single video record and compute all derived fields.
    
    Args:
        row: Video metadata row from dislike dataset
        comments_df: DataFrame containing comment sentiment data
        
    Returns:
        Dictionary with all fields for the output CSV
    """
    video_id = row['video_id']
    
    # Extract basic fields
    channel_id = row['channel_id']
    published_at = row['published_at']
    view_count = row['view_count']
    likes = row['likes']
    dislikes = row['dislikes']
    comment_count = row['comment_count']
    
    # Handle missing values - convert to 0
    view_count = 0 if pd.isna(view_count) else int(view_count)
    likes = 0 if pd.isna(likes) else int(likes)
    dislikes = 0 if pd.isna(dislikes) else int(dislikes)
    comment_count = 0 if pd.isna(comment_count) else int(comment_count)
    
    # Compute average sentiment from comments
    sentiment = compute_average_sentiment(comments_df, video_id)
    
    # Compute derived metrics
    # Like_Dislike Score: Likes / (Likes + Dislikes)
    # Add 1 to denominator if it would be 0
    like_dislike_denominator = likes + dislikes
    if like_dislike_denominator == 0:
        like_dislike_denominator = 1
    like_dislike_score = likes / like_dislike_denominator
    
    # LD Score OHE: Categorical version of Like_Dislike Score
    ld_score_ohe = categorize_like_dislike_score(like_dislike_score)
    
    # View_Like Ratio: view_count / like_count
    # Add 1 to denominator if it's 0
    like_count_safe = likes if likes > 0 else 1
    view_like_ratio = view_count / like_count_safe
    
    # View_Dislike Ratio: view_count / dislike_count
    # Add 1 to denominator if it's 0
    dislike_count_safe = dislikes if dislikes > 0 else 1
    view_dislike_ratio = view_count / dislike_count_safe
    
    # Dislike_Like Ratio: dislike_count / like_count
    # Add 1 to denominator if it's 0
    dislike_like_ratio = dislikes / like_count_safe
    
    # No_Comments Binary: 1 if video has no comments in comments dataset, 0 otherwise
    no_comments = 1 if sentiment['comment_sample_size'] == 0 else 0
    
    return {
        'video_id': video_id,
        'channel_id': channel_id,
        'published_at': published_at,
        'view_count': view_count,
        'likes': likes,
        'dislikes': dislikes,
        'comment_count': comment_count,
        'like_dislike_score': like_dislike_score,
        'ld_score_ohe': ld_score_ohe,
        'view_like_ratio': view_like_ratio,
        'view_dislike_ratio': view_dislike_ratio,
        'dislike_like_ratio': dislike_like_ratio,
        'no_comments': no_comments,
        'avg_pos': sentiment['avg_pos'],
        'avg_neu': sentiment['avg_neu'],
        'avg_neg': sentiment['avg_neg'],
        'avg_compound': sentiment['avg_compound'],
        'comment_sample_size': sentiment['comment_sample_size']
    }


def filter_and_combine_data(
    dislike_dataset_path: str = '..\\youtube_dislike_dataset.csv',
    comments_sentiment_path: str = '..\\combined_comments_all.csv',
    output_path: str = 'filtered_combined_dataset.csv',
    batch_size: int = 1000,
    start_row: Optional[int] = None,
    end_row: Optional[int] = None,
    num_threads: int = 1,
    resume_mode: bool = False
):
    """
    Process YouTube dislike dataset and combine with comment sentiment data.
    
    Args:
        dislike_dataset_path: Path to the YouTube dislike dataset CSV
        comments_sentiment_path: Path to the combined comments sentiment CSV
        output_path: Path for the output CSV file
        batch_size: Number of records to process before writing to disk
        start_row: Starting row index (0-based, None = from beginning)
        end_row: Ending row index (exclusive, None = to end)
        num_threads: Number of threads for parallel processing
        resume_mode: If True, append to existing output file instead of overwriting
    """
    # Create error log file path
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    error_log_path = f'error_log_{timestamp}.txt'
    error_count = 0
    
    def log_error(message: str):
        """Write error message to log file."""
        nonlocal error_count
        error_count += 1
        with open(error_log_path, 'a', encoding='utf-8') as f:
            timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"[{timestamp_str}] {message}\n")
    
    print("=" * 80)
    print("YouTube Dislike Dataset Filtering and Combining Script")
    print("=" * 80)
    print(f"\nError log file: {error_log_path}")
    
    # Check resume mode
    output_file_exists = os.path.exists(output_path)
    if resume_mode:
        if output_file_exists:
            print(f"\n⚠ RESUME MODE: Will append to existing file: {output_path}")
        else:
            print(f"\n⚠ RESUME MODE: Output file does not exist, will create new file: {output_path}")
    else:
        if output_file_exists:
            print(f"\n⚠ Output file exists and will be OVERWRITTEN: {output_path}")
    
    # Load comments sentiment data
    print(f"\n[1/4] Loading comments sentiment data from: {comments_sentiment_path}")
    try:
        comments_df = pd.read_csv(
            comments_sentiment_path,
            usecols=['video_id', 'pos', 'neu', 'neg', 'compound']
        )
        print(f"  ✓ Loaded {len(comments_df):,} comment records")
        print(f"  ✓ Unique videos with comments: {comments_df['video_id'].nunique():,}")
    except Exception as e:
        error_msg = f"Error loading comments data: {e}"
        print(f"  ✗ {error_msg}")
        log_error(error_msg)
        return
    
    # Load dislike dataset metadata
    print(f"\n[2/4] Loading dislike dataset from: {dislike_dataset_path}")
    try:
        # Load full dataset first
        dislike_df = pd.read_csv(
            dislike_dataset_path,
            usecols=['video_id', 'channel_id', 'published_at', 'view_count', 
                    'likes', 'dislikes', 'comment_count']
        )
        total_available = len(dislike_df)
        print(f"  ✓ Loaded {total_available:,} video records")
        
        # Apply row filtering if specified
        if start_row is not None or end_row is not None:
            # Convert from 1-based to 0-based indexing
            start_idx = (start_row - 1) if start_row is not None else 0
            end_idx = end_row if end_row is not None else len(dislike_df)
            
            # Validate indices
            if start_row is not None and (start_row < 1 or start_row > len(dislike_df)):
                error_msg = f"Invalid start_row: {start_row}. Must be between 1 and {len(dislike_df)}"
                print(f"  ✗ {error_msg}")
                log_error(error_msg)
                return
            if end_row is not None and (end_row < 1 or end_row > len(dislike_df)):
                error_msg = f"Invalid end_row: {end_row}. Must be between 1 and {len(dislike_df)}"
                print(f"  ✗ {error_msg}")
                log_error(error_msg)
                return
            if start_row is not None and end_row is not None and start_row > end_row:
                error_msg = f"Invalid range: start_row ({start_row}) must be <= end_row ({end_row})"
                print(f"  ✗ {error_msg}")
                log_error(error_msg)
                return
            
            dislike_df = dislike_df.iloc[start_idx:end_idx].reset_index(drop=True)
            actual_start = start_row if start_row is not None else 1
            actual_end = end_row if end_row is not None else len(dislike_df)
            print(f"  ✓ Filtered to rows {actual_start:,} to {actual_end:,} (inclusive, {len(dislike_df):,} records)")
        
    except Exception as e:
        error_msg = f"Error loading dislike dataset: {e}"
        print(f"  ✗ {error_msg}")
        log_error(error_msg)
        return
    
    # Process records
    print(f"\n[3/4] Processing video records and computing metrics...")
    print(f"  Processing in batches of {batch_size:,} records")
    if num_threads > 1:
        print(f"  Using {num_threads} threads for parallel processing")
    
    processed_records = []
    total_records = len(dislike_df)
    
    # Determine if we should write header based on resume mode and file existence
    should_write_header = not (resume_mode and output_file_exists)
    
    if num_threads > 1:
        # Parallel processing with ThreadPoolExecutor
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading
        
        # Thread lock for batch writing
        write_lock = threading.Lock()
        completed_count = 0
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Submit all tasks
            futures = {
                executor.submit(process_video_record, row, comments_df): (idx, row) 
                for idx, row in dislike_df.iterrows()
            }
            
            # Process completed tasks
            for future in as_completed(futures):
                idx, row = futures[future]
                try:
                    processed_record = future.result()
                    
                    # Thread-safe append to processed_records
                    with write_lock:
                        processed_records.append(processed_record)
                        completed_count += 1
                        current_count = completed_count
                        
                        # Check if we should write a batch
                        should_write = (len(processed_records) >= batch_size) or (current_count == total_records)
                        
                        if should_write:
                            print(f"  Progress: {current_count:,}/{total_records:,} records ({current_count/total_records*100:.1f}%)")
                            
                            # Write batch to file
                            try:
                                batch_df = pd.DataFrame(processed_records)
                                if should_write_header:
                                    # First batch in non-resume mode - create new file with header
                                    batch_df.to_csv(output_path, index=False, mode='w')
                                    print(f"    → Created output file: {output_path}")
                                    should_write_header = False  # Don't write header again
                                else:
                                    # Subsequent batches or resume mode - append without header
                                    batch_df.to_csv(output_path, index=False, mode='a', header=False)
                                    print(f"    → Appended batch to file")
                                
                                # Clear processed records to free memory
                                processed_records.clear()
                            except Exception as e:
                                error_msg = f"Error writing batch to file: {e}"
                                print(f"    ✗ {error_msg}")
                                log_error(error_msg)
                        elif current_count % 100 == 0:
                            # Progress update without writing
                            print(f"  Progress: {current_count:,}/{total_records:,} records ({current_count/total_records*100:.1f}%)")
                            
                except Exception as e:
                    with write_lock:
                        completed_count += 1
                        video_id = row.get('video_id', 'UNKNOWN')
                        error_msg = f"Error processing record at index {idx} (video_id: {video_id}): {e}"
                        print(f"  ✗ {error_msg}")
                        log_error(error_msg)
    else:
        # Sequential processing (original single-threaded approach)
        for idx in range(len(dislike_df)):
            row = dislike_df.iloc[idx]
            
            # Process the record
            try:
                processed_record = process_video_record(row, comments_df)
                processed_records.append(processed_record)
            except Exception as e:
                video_id = row.get('video_id', 'UNKNOWN')
                error_msg = f"Error processing record at index {idx} (video_id: {video_id}): {e}"
                print(f"  ✗ {error_msg}")
                log_error(error_msg)
                continue
            
            # Progress update and batch writing
            current_count = idx + 1
            if current_count % batch_size == 0 or current_count == total_records:
                print(f"  Progress: {current_count:,}/{total_records:,} records ({current_count/total_records*100:.1f}%)")
                
                # Write batch to file
                try:
                    batch_df = pd.DataFrame(processed_records)
                    if should_write_header:
                        # First batch in non-resume mode - create new file with header
                        batch_df.to_csv(output_path, index=False, mode='w')
                        print(f"    → Created output file: {output_path}")
                        should_write_header = False  # Don't write header again
                    else:
                        # Subsequent batches or resume mode - append without header
                        batch_df.to_csv(output_path, index=False, mode='a', header=False)
                        print(f"    → Appended batch to file")
                except Exception as e:
                    error_msg = f"Error writing batch to file: {e}"
                    print(f"    ✗ {error_msg}")
                    log_error(error_msg)
                
                # Clear processed records to free memory
                processed_records = []
    
    # Final summary
    print(f"\n[4/4] Processing complete!")
    print(f"  ✓ Processed {total_records:,} video records")
    print(f"  ✓ Output file: {output_path}")
    if error_count > 0:
        print(f"  ⚠ {error_count:,} errors encountered (see {error_log_path})")
    else:
        print(f"  ✓ No errors encountered")
        # Remove empty error log file if no errors
        if os.path.exists(error_log_path):
            os.remove(error_log_path)
    
    # Load and display summary statistics
    print(f"\n" + "=" * 80)
    print("Summary Statistics")
    print("=" * 80)
    
    try:
        final_df = pd.read_csv(output_path)
        
        print(f"\nDataset shape: {final_df.shape[0]:,} rows × {final_df.shape[1]} columns")
        
        print(f"\nColumns: {list(final_df.columns)}")
        
        print(f"\nBasic Statistics:")
        print(f"  • Videos with comments: {(final_df['no_comments'] == 0).sum():,} ({(final_df['no_comments'] == 0).sum()/len(final_df)*100:.1f}%)")
        print(f"  • Videos without comments: {(final_df['no_comments'] == 1).sum():,} ({(final_df['no_comments'] == 1).sum()/len(final_df)*100:.1f}%)")
        print(f"  • Average comment sample size: {final_df['comment_sample_size'].mean():.1f}")
        
        print(f"\nLike/Dislike Score Distribution (OHE):")
        print(f"  • Category -1 (< 0.33): {(final_df['ld_score_ohe'] == -1).sum():,} ({(final_df['ld_score_ohe'] == -1).sum()/len(final_df)*100:.1f}%)")
        print(f"  • Category  0 (0.33-0.67): {(final_df['ld_score_ohe'] == 0).sum():,} ({(final_df['ld_score_ohe'] == 0).sum()/len(final_df)*100:.1f}%)")
        print(f"  • Category  1 (> 0.67): {(final_df['ld_score_ohe'] == 1).sum():,} ({(final_df['ld_score_ohe'] == 1).sum()/len(final_df)*100:.1f}%)")
        
        print(f"\nSentiment Averages:")
        print(f"  • Positive:  {final_df['avg_pos'].mean():.4f} (±{final_df['avg_pos'].std():.4f})")
        print(f"  • Neutral:   {final_df['avg_neu'].mean():.4f} (±{final_df['avg_neu'].std():.4f})")
        print(f"  • Negative:  {final_df['avg_neg'].mean():.4f} (±{final_df['avg_neg'].std():.4f})")
        print(f"  • Compound:  {final_df['avg_compound'].mean():.4f} (±{final_df['avg_compound'].std():.4f})")
        
        print(f"\nRatio Metrics:")
        print(f"  • Like/Dislike Score:    {final_df['like_dislike_score'].mean():.4f} (median: {final_df['like_dislike_score'].median():.4f})")
        print(f"  • View/Like Ratio:       {final_df['view_like_ratio'].mean():.2f} (median: {final_df['view_like_ratio'].median():.2f})")
        print(f"  • View/Dislike Ratio:    {final_df['view_dislike_ratio'].mean():.2f} (median: {final_df['view_dislike_ratio'].median():.2f})")
        print(f"  • Dislike/Like Ratio:    {final_df['dislike_like_ratio'].mean():.4f} (median: {final_df['dislike_like_ratio'].median():.4f})")
        
    except Exception as e:
        error_msg = f"Error loading summary statistics: {e}"
        print(f"  ✗ {error_msg}")
        log_error(error_msg)
    
    print("\n" + "=" * 80)
    print("✓ Script completed successfully!")
    print("=" * 80)


def main():
    """Main function with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description='Filter and combine YouTube dislike dataset with comment sentiment data.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default file paths
  python filter_and_combine_data.py
  
  # Process all records with custom paths
  python filter_and_combine_data.py --all --input youtube_dislike_dataset.csv --output filtered_data.csv
  
  # Process first 10,000 rows (rows 1 through 10,000 inclusive)
  python filter_and_combine_data.py --start-row 1 --end-row 10000
  
  # Process first 100 rows (rows 1-100)
  python filter_and_combine_data.py --start-row 1 --end-row 100
  
  # Process rows 101-200
  python filter_and_combine_data.py --start-row 101 --end-row 200
  
  # Resume processing from row 10001 (appends to existing file)
  python filter_and_combine_data.py --resume-row 10001 --end-row 20000
  
  # Use parallel processing with 4 threads
  python filter_and_combine_data.py --threads 4
  
  # Combine multiple options
  python filter_and_combine_data.py --start-row 10001 --end-row 20000 --threads 8 --batch-size 5000
        """
    )
    
    # Input/Output file arguments
    parser.add_argument(
        '--input',
        default='..\\youtube_dislike_dataset.csv',
        help='Path to YouTube dislike dataset CSV (default: ..\\youtube_dislike_dataset.csv)'
    )
    
    parser.add_argument(
        '--comments',
        default='..\\combined_comments_all.csv',
        help='Path to comments sentiment CSV (default: ..\\combined_comments_all.csv)'
    )
    
    parser.add_argument(
        '--output',
        default='filtered_combined_dataset.csv',
        help='Path for output CSV file (default: filtered_combined_dataset.csv)'
    )
    
    # Row range arguments
    row_group = parser.add_mutually_exclusive_group()
    
    row_group.add_argument(
        '--all',
        action='store_true',
        help='Process all records (default behavior if no row range specified)'
    )
    
    parser.add_argument(
        '--start-row',
        type=int,
        default=None,
        help='Starting row number (1-based, inclusive). Overwrites existing output file. If not specified, starts from row 1.'
    )
    
    parser.add_argument(
        '--end-row',
        type=int,
        default=None,
        help='Ending row number (1-based, inclusive). If not specified, processes to end.'
    )
    
    row_group.add_argument(
        '--resume-row',
        type=int,
        default=None,
        help='Resume processing from this row number (1-based, inclusive). Appends to existing output file instead of overwriting.'
    )
    
    # Performance arguments
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1000,
        help='Number of records to process before writing to disk (default: 1000)'
    )
    
    parser.add_argument(
        '--threads',
        type=int,
        default=1,
        help='Number of threads for parallel processing (default: 1, single-threaded)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.threads < 1:
        print("Error: --threads must be at least 1")
        return
    
    if args.batch_size < 1:
        print("Error: --batch-size must be at least 1")
        return
    
    # Handle resume-row option
    resume_mode = False
    start_row = args.start_row
    
    if args.resume_row is not None:
        if args.resume_row < 1:
            print("Error: --resume-row must be at least 1 (1-based row numbering)")
            return
        resume_mode = True
        start_row = args.resume_row
    
    if start_row is not None and start_row < 1:
        print("Error: --start-row must be at least 1 (1-based row numbering)")
        return
    
    if args.end_row is not None and args.end_row < 1:
        print("Error: --end-row must be at least 1 (1-based row numbering)")
        return
    
    if start_row is not None and args.end_row is not None and start_row > args.end_row:
        print("Error: --start-row must be <= --end-row")
        return
    
    # Run the filtering and combining process
    filter_and_combine_data(
        dislike_dataset_path=args.input,
        comments_sentiment_path=args.comments,
        output_path=args.output,
        batch_size=args.batch_size,
        start_row=start_row,
        end_row=args.end_row,
        num_threads=args.threads,
        resume_mode=resume_mode
    )


if __name__ == '__main__':
    main()
