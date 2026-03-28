#!/usr/bin/env python3
"""Merge duration and genre_id from combined_additional_metadata.csv into yt_dataset_v4.csv.

Keeps only records present in both files (inner join on `video_id`).
Writes output to `yt_dataset_v4_merged.csv` in the same folder and prints the total row count.

This script is intentionally defensive: it will try common alternative column names for
duration and genre id and exit with a helpful message if it can't find them.
"""
from pathlib import Path
import sys
import pandas as pd


def find_column(cols, candidates):
    lower_map = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def main():
    base = Path(__file__).parent
    left_path = base / "yt_dataset_v4.csv"
    right_path = base / "combined_additional_metadata.csv"

    if not left_path.exists() or not right_path.exists():
        print(f"Error: missing input files in {base}. Expected:\n - {left_path.name}\n - {right_path.name}")
        sys.exit(2)

    # Read CSVs
    left = pd.read_csv(left_path, dtype=str, low_memory=False)
    right = pd.read_csv(right_path, dtype=str, low_memory=False)

    # Key column
    key = find_column(left.columns.union(right.columns), ["video_id", "video id", "id"])
    if key is None:
        print("Error: could not find a video id column in inputs. Expected 'video_id' or similar.")
        sys.exit(3)

    # Common candidate names for duration and genre id
    duration_candidates = [
        "duration", "video_duration", "length", "duration_seconds", "duration_sec"
    ]
    genre_candidates = [
        "genre_id", "genreid", "genre id", "genre", "category_id", "categoryid"
    ]

    dur_col = find_column(right.columns, duration_candidates)
    genre_col = find_column(right.columns, genre_candidates)

    if dur_col is None and genre_col is None:
        print(f"Error: could not find duration or genre id columns in {right_path.name}.")
        print("Columns available:", ", ".join(right.columns.tolist()))
        sys.exit(4)

    use_cols = [key]
    if dur_col:
        use_cols.append(dur_col)
    if genre_col:
        use_cols.append(genre_col)

    # Reduce right to only needed columns (if duplicates exist, keep unique)
    right_sub = right.loc[:, use_cols]

    # Perform inner join on the key
    merged = pd.merge(left, right_sub, on=key, how="inner")

    out_path = base / "yt_dataset_v4_merged.csv"
    merged.to_csv(out_path, index=False)

    print(f"Wrote merged CSV to: {out_path}\nTotal records in final output: {len(merged)}")


if __name__ == "__main__":
    main()
