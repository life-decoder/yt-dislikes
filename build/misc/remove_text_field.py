"""
Remove the 'text' field from combined comments CSV file.

This script reads the combined_comments_sentiment.csv file and creates a new version
without the text column, keeping only: video_id, comment_index, time, pos, neu, neg, compound.
"""

import csv
import argparse
from pathlib import Path


def remove_text_field(input_file, output_file):
    """
    Remove the 'text' column from the CSV file.
    
    Args:
        input_file: Path to the input CSV file
        output_file: Path to the output CSV file
    """
    print(f"Reading from: {input_file}")
    print(f"Writing to: {output_file}")
    
    rows_processed = 0
    
    try:
        with open(input_file, 'r', encoding='utf-8') as infile, \
             open(output_file, 'w', encoding='utf-8', newline='') as outfile:
            
            reader = csv.DictReader(infile)
            
            # Define new fieldnames without 'text'
            original_fields = reader.fieldnames
            if original_fields is None:
                print("Error: Could not read CSV headers. The file may be empty or invalid.")
                return
            
            if 'text' not in original_fields:
                print("Warning: 'text' field not found in the CSV. Columns found:", original_fields)
                return
            
            new_fields = [field for field in original_fields if field != 'text']
            
            writer = csv.DictWriter(outfile, fieldnames=new_fields)
            writer.writeheader()
            
            # Process rows in chunks for progress updates
            for row in reader:
                # Remove 'text' field from the row
                row.pop('text', None)
                writer.writerow(row)
                
                rows_processed += 1
                
                # Progress update every 100,000 rows
                if rows_processed % 100000 == 0:
                    print(f"Processed {rows_processed:,} rows...")
            
            print(f"\nCompleted! Total rows processed: {rows_processed:,}")
            print(f"Output saved to: {output_file}")
            
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        return
    except Exception as e:
        print(f"Error processing file: {e}")
        return


def main():
    parser = argparse.ArgumentParser(
        description='Remove the text field from combined comments CSV file.'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='combined_comments_sentiment.csv',
        help='Input CSV file (default: combined_comments_sentiment.csv)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='combined_comments_no_text.csv',
        help='Output CSV file (default: combined_comments_no_text.csv)'
    )
    
    args = parser.parse_args()
    
    # Verify input file exists
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file '{args.input}' does not exist.")
        return
    
    # Get file size for information
    file_size_mb = input_path.stat().st_size / (1024 * 1024)
    print(f"Input file size: {file_size_mb:.2f} MB")
    
    remove_text_field(args.input, args.output)


if __name__ == '__main__':
    main()
