#!/usr/bin/env python3
"""Count records missing genre and/or duration in CSV files.

Usage:
  python count_missing_genre_duration.py [folder]

If no folder is provided the script runs against its own directory.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Dict, Optional, Tuple

MISSING_EQUIVALENTS = {"", "na", "n/a", "none", "nan"}

def is_missing(value: Optional[str]) -> bool:
    if value is None:
        return True
    v = str(value).strip()
    return v == "" or v.lower() in MISSING_EQUIVALENTS


def process_file(path: str) -> Tuple[Dict[str, int], Optional[str], Optional[str]]:
    counts = {"total": 0, "missing_genre": 0, "missing_duration": 0, "missing_both": 0}

    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = [fn.strip() for fn in (reader.fieldnames or [])]

        # Find best matching header names (case-insensitive), fall back to substring matches
        genre_keys = [k for k in fieldnames if k.lower() == "genre"]
        duration_keys = [k for k in fieldnames if k.lower() == "duration"]
        if not genre_keys:
            genre_keys = [k for k in fieldnames if "genre" in k.lower()]
        if not duration_keys:
            duration_keys = [k for k in fieldnames if "duration" in k.lower()]

        genre_key = genre_keys[0] if genre_keys else None
        duration_key = duration_keys[0] if duration_keys else None

        for row in reader:
            counts["total"] += 1

            missing_genre = True if genre_key is None else is_missing(row.get(genre_key))
            missing_duration = True if duration_key is None else is_missing(row.get(duration_key))

            if missing_genre:
                counts["missing_genre"] += 1
            if missing_duration:
                counts["missing_duration"] += 1
            if missing_genre and missing_duration:
                counts["missing_both"] += 1

    return counts, genre_key, duration_key


def main(folder: str) -> int:
    if not os.path.isdir(folder):
        print(f"Error: folder does not exist: {folder}")
        return 2

    csv_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".csv")]
    if not csv_files:
        print(f"No CSV files found in: {folder}")
        return 0

    print(f"Scanning {len(csv_files)} CSV file(s) in: {folder}\n")

    for path in sorted(csv_files):
        counts, genre_key, duration_key = process_file(path)
        print(f"File: {os.path.basename(path)}")
        print(f"  detected genre header:   {genre_key or '(not found)'}")
        print(f"  detected duration header: {duration_key or '(not found)'}")
        print(f"  total rows:              {counts['total']}")
        print(f"  missing genre:           {counts['missing_genre']}")
        print(f"  missing duration:        {counts['missing_duration']}")
        print(f"  missing both:            {counts['missing_both']}\n")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count missing genre/duration values in CSV files")
    parser.add_argument("folder", nargs="?", default=os.path.dirname(__file__), help="Folder containing CSV files (default: script folder)")
    args = parser.parse_args()
    sys.exit(main(os.path.abspath(args.folder)))
