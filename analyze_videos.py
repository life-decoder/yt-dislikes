import pandas as pd

def main():
    # Read the comments sentiment CSV
    print("Reading comments_sentiment_all.csv...")
    comments_df = pd.read_csv('comments_sentiment_all.csv')
    
    # Get unique video IDs from comments
    unique_videos = comments_df['video_id'].unique()
    num_unique_videos = len(unique_videos)
    
    print(f"\n{'='*60}")
    print(f"Number of unique videos in comments_sentiment_all.csv: {num_unique_videos}")
    print(f"{'='*60}")
    
    # Get the first and last video_id in comments_sentiment_all.csv
    first_video_id = comments_df['video_id'].iloc[0]
    last_video_id = comments_df['video_id'].iloc[-1]
    print(f"\nFirst video_id in comments_sentiment_all.csv: {first_video_id}")
    print(f"Last video_id in comments_sentiment_all.csv: {last_video_id}")
    
    # Read the dislike dataset to find the corresponding rows
    print("\nSearching youtube_dislike_dataset.csv for the first and last video_ids...")
    
    # Read the dislike dataset in chunks to handle large file
    chunk_size = 10000
    first_row_number = None
    last_row_number = None
    current_row = 0
    
    for chunk in pd.read_csv('youtube_dislike_dataset.csv', chunksize=chunk_size):
        # Check if the first video_id is in this chunk
        if first_row_number is None:
            first_mask = chunk['video_id'] == first_video_id
            if first_mask.any():
                position_in_chunk = first_mask.idxmax() - chunk.index[0]
                first_row_number = current_row + position_in_chunk + 2  # +1 for header, +1 for 1-indexed
                first_row_data = chunk[first_mask].iloc[0]
        
        # Check if the last video_id is in this chunk
        if last_row_number is None:
            last_mask = chunk['video_id'] == last_video_id
            if last_mask.any():
                position_in_chunk = last_mask.idxmax() - chunk.index[0]
                last_row_number = current_row + position_in_chunk + 2  # +1 for header, +1 for 1-indexed
                last_row_data = chunk[last_mask].iloc[0]
        
        # Break if we found both
        if first_row_number is not None and last_row_number is not None:
            break
        
        current_row += len(chunk)
    
    # Display results for first video
    if first_row_number is not None:
        print(f"\n{'='*60}")
        print(f"FIRST VIDEO in comments_sentiment_all.csv:")
        print(f"Row number in youtube_dislike_dataset.csv: {first_row_number}")
        print(f"Video ID: {first_video_id}")
        print(f"{'='*60}")
        print(f"  Title: {first_row_data['title']}")
        print(f"  Channel: {first_row_data['channel_title']}")
        print(f"  Views: {first_row_data['view_count']:,}")
        print(f"  Likes: {first_row_data['likes']:,}")
        print(f"  Dislikes: {first_row_data['dislikes']:,}")
        print(f"  Comments: {first_row_data['comment_count']:,}")
    else:
        print(f"\n{'='*60}")
        print(f"WARNING: First video ID '{first_video_id}' NOT FOUND in youtube_dislike_dataset.csv")
        print(f"{'='*60}")
    
    # Display results for last video
    if last_row_number is not None:
        print(f"\n{'='*60}")
        print(f"LAST VIDEO in comments_sentiment_all.csv:")
        print(f"Row number in youtube_dislike_dataset.csv: {last_row_number}")
        print(f"Video ID: {last_video_id}")
        print(f"{'='*60}")
        print(f"  Title: {last_row_data['title']}")
        print(f"  Channel: {last_row_data['channel_title']}")
        print(f"  Views: {last_row_data['view_count']:,}")
        print(f"  Likes: {last_row_data['likes']:,}")
        print(f"  Dislikes: {last_row_data['dislikes']:,}")
        print(f"  Comments: {last_row_data['comment_count']:,}")
    else:
        print(f"\n{'='*60}")
        print(f"WARNING: Last video ID '{last_video_id}' NOT FOUND in youtube_dislike_dataset.csv")
        print(f"{'='*60}")
    
    # Check for missing videos in comments dataset
    if first_row_number is not None and last_row_number is not None:
        print(f"\n{'='*60}")
        print(f"Checking for videos without comments data...")
        print(f"(From row {first_row_number} to row {last_row_number} in dislike dataset)")
        print(f"{'='*60}")
        
        # Get set of all video_ids in comments dataset
        comments_video_ids = set(comments_df['video_id'].unique())
        
        # Read the dislike dataset again and check each video in the range
        missing_videos = []
        current_row = 0
        
        for chunk in pd.read_csv('youtube_dislike_dataset.csv', chunksize=chunk_size):
            for idx, row in chunk.iterrows():
                actual_row = current_row + (idx - chunk.index[0]) + 2  # +1 for header, +1 for 1-indexed
                
                # Only check videos in the range
                if actual_row >= first_row_number and actual_row <= last_row_number:
                    if row['video_id'] not in comments_video_ids:
                        missing_videos.append({
                            'row_number': actual_row,
                            'video_id': row['video_id'],
                            'title': row['title'],
                            'channel': row['channel_title']
                        })
                
                # Break if we've passed the last row
                if actual_row > last_row_number:
                    break
            
            current_row += len(chunk)
            
            # Break if we've passed the last row
            if current_row + 2 > last_row_number:
                break
        
        if missing_videos:
            print(f"\nFound {len(missing_videos)} videos WITHOUT comments data:")
            print(f"\nRow Number | Video ID        | Title")
            print(f"-" * 80)
            for video in missing_videos:
                title_truncated = video['title'][:50] + "..." if len(video['title']) > 50 else video['title']
                print(f"{video['row_number']:10} | {video['video_id']:15} | {title_truncated}")
        else:
            print(f"\nAll {last_row_number - first_row_number + 1} videos in the range have comments data!")
        
        print(f"\n{'='*60}")
        print(f"Summary:")
        print(f"  Total videos in range: {last_row_number - first_row_number + 1}")
        print(f"  Videos with comments: {len(comments_video_ids)}")
        print(f"  Videos without comments: {len(missing_videos)}")
        print(f"{'='*60}")

if __name__ == "__main__":
    main()
