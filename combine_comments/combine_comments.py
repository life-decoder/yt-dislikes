import pandas as pd
import os
from pathlib import Path
from datetime import datetime

def combine_comments(input_folder=r'../comments_datasets/filtered', output_file='combined_comments_sentiment.csv'):
    """
    Combine all CSV files in the input folder, excluding specified columns.
    Files are processed in alphabetical order by filename.
    
    Args:
        input_folder: Path to folder containing CSV files to combine
        output_file: Name of the output combined CSV file
    """
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Resolve paths relative to script directory if they are relative paths
    if not os.path.isabs(input_folder):
        input_folder = os.path.join(script_dir, input_folder)
    if not os.path.isabs(output_file):
        output_file = os.path.join(script_dir, output_file)
    
    # Create error log file path
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    error_log_path = os.path.join(script_dir, f'combine_errors_{timestamp}.txt')
    error_count = 0
    
    def log_error(message: str):
        """Write error message to log file."""
        nonlocal error_count
        error_count += 1
        with open(error_log_path, 'a', encoding='utf-8') as f:
            timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"[{timestamp_str}] {message}\n")
    
    # Get all CSV files in the folder and sort them
    input_path = Path(input_folder)
    csv_files = sorted(input_path.glob('*.csv'))
    
    if not csv_files:
        error_msg = f"No CSV files found in {input_folder}"
        print(error_msg)
        log_error(error_msg)
        return
    
    print(f"Found {len(csv_files)} CSV files to combine:")
    for file in csv_files:
        print(f"  - {file.name}")
    
    # Columns to exclude
    exclude_columns = ['cid', 'author', 'votes', 'text']
    
    # List to store dataframes
    dfs = []
    
    # Read each file
    for i, file in enumerate(csv_files, 1):
        print(f"\nProcessing file {i}/{len(csv_files)}: {file.name}")
        try:
            # Read the CSV file
            df = pd.read_csv(file)
            
            # Get initial shape
            initial_rows = len(df)
            
            # Drop the excluded columns if they exist
            columns_to_drop = [col for col in exclude_columns if col in df.columns]
            if columns_to_drop:
                df = df.drop(columns=columns_to_drop)
                print(f"  Dropped columns: {columns_to_drop}")
            
            # Add to list
            dfs.append(df)
            print(f"  Rows: {initial_rows:,}")
            
        except Exception as e:
            error_msg = f"Error processing {file.name}: {e}"
            print(f"  {error_msg}")
            log_error(error_msg)
            continue
    
    if not dfs:
        error_msg = "No data to combine!"
        print(f"\n{error_msg}")
        log_error(error_msg)
        return
    
    # Combine all dataframes
    print(f"\nCombining {len(dfs)} dataframes...")
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Save to output file
    print(f"Saving combined data to {output_file}...")
    combined_df.to_csv(output_file, index=False)
    
    print(f"\n✓ Successfully combined {len(csv_files)} files")
    print(f"✓ Total rows: {len(combined_df):,}")
    print(f"✓ Columns: {list(combined_df.columns)}")
    print(f"✓ Output file: {output_file}")
    
    if error_count > 0:
        print(f"\n⚠ {error_count} errors encountered (see {error_log_path})")
    else:
        print(f"✓ No errors encountered")
        # Remove empty error log file if no errors
        if os.path.exists(error_log_path):
            os.remove(error_log_path)

if __name__ == "__main__":
    combine_comments()
