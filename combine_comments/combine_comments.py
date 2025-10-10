import pandas as pd
import os
from pathlib import Path

def combine_comments(input_folder=r'../comments_datasets', output_file='combined_comments_sentiment.csv'):
    """
    Combine all CSV files in the input folder, excluding specified columns.
    Files are processed in alphabetical order by filename.
    
    Args:
        input_folder: Path to folder containing CSV files to combine
        output_file: Name of the output combined CSV file
    """
    # Get all CSV files in the folder and sort them
    input_path = Path(input_folder)
    csv_files = sorted(input_path.glob('*.csv'))
    
    if not csv_files:
        print(f"No CSV files found in {input_folder}")
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
            print(f"  Error processing {file.name}: {e}")
            continue
    
    if not dfs:
        print("\nNo data to combine!")
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

if __name__ == "__main__":
    combine_comments()
