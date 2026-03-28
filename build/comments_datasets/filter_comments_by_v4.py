#!/usr/bin/env python3
"""Filter comments datasets keeping only rows whose video id appears in yt_dataset_v4.csv.

Usage examples:
  python comments_datasets/filter_comments_by_v4.py
  python comments_datasets/filter_comments_by_v4.py --videos-file yt_dataset_v4.csv --comments-dir comments_datasets --out-dir comments_datasets/filtered

The script looks for common video id column names if not provided: 'video_id', 'videoId', 'videoId_str', 'videoid'.
It streams CSVs so it can handle large files without loading everything into memory.
"""
import argparse
import csv
import os
from pathlib import Path
from typing import Set, Optional


def load_video_ids(videos_csv: Path, id_column: Optional[str] = None) -> Set[str]:
    """Return a set of video ids read from videos_csv. If id_column is None, try to auto-detect.
    Uses streaming read, expects a header row.
    """
    vids = set()
    with videos_csv.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            raise SystemExit(f"No header found in {videos_csv}")

        # determine id column
        id_col = id_column
        if id_col is None:
            lower = [c.lower() for c in reader.fieldnames]
            # prefer exact matches
            for cand in ("video_id", "videoid", "videoId", "videoId_str"):
                if cand in reader.fieldnames or cand.lower() in lower:
                    id_col = reader.fieldnames[lower.index(cand.lower())]
                    break
            # fallback: any column containing both 'video' and 'id'
            if id_col is None:
                for c in reader.fieldnames:
                    low = c.lower()
                    if "video" in low and "id" in low:
                        id_col = c
                        break

        if id_col is None:
            raise SystemExit("Could not determine video id column in videos CSV. Please provide --id-column.")

        # collect ids
        for row in reader:
            val = row.get(id_col)
            if val is None:
                continue
            val = val.strip()
            if val:
                vids.add(val)

    return vids


def detect_id_column(fieldnames):
    lower = [c.lower() for c in fieldnames]
    for cand in ("video_id", "videoid", "videoid_str", "videoId"):
        if cand in fieldnames or cand.lower() in lower:
            return fieldnames[lower.index(cand.lower())]
    for c in fieldnames:
        low = c.lower()
        if "video" in low and "id" in low:
            return c
    return None


def filter_file(in_path: Path, out_path: Path, video_ids: Set[str], id_column: Optional[str] = None) -> int:
    """Filter in_path writing matching rows to out_path. Returns number of rows written."""
    written = 0
    with in_path.open("r", encoding="utf-8-sig", newline="") as infh:
        reader = csv.DictReader(infh)
        if not reader.fieldnames:
            return 0

        id_col = id_column or detect_id_column(reader.fieldnames)
        if id_col is None:
            raise SystemExit(f"Could not find a video id column in {in_path}. Provide --id-column to override.")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8", newline="") as outfh:
            writer = csv.DictWriter(outfh, fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in reader:
                vid = row.get(id_col, "")
                if vid is None:
                    continue
                if vid.strip() in video_ids:
                    writer.writerow(row)
                    written += 1

    return written


def main():
    p = argparse.ArgumentParser(description="Filter comments datasets keeping only rows where the video id is present in a master video list (yt_dataset_v4.csv)")
    p.add_argument("--videos-file", default=Path("yt_dataset_v4.csv"), type=Path, help="CSV file containing allowed video ids (default: yt_dataset_v4.csv)")
    p.add_argument("--comments-dir", default=Path("comments_datasets"), type=Path, help="Directory containing comment CSV files to filter")
    p.add_argument("--out-dir", default=Path("comments_datasets") / "filtered", type=Path, help="Directory to write filtered CSVs")
    p.add_argument("--id-column", default=None, help="Explicit column name for video id if auto-detection fails")
    p.add_argument("--extensions", default=".csv", help="Comma-separated list of file extensions to process (default: .csv)")
    args = p.parse_args()

    videos_file = Path(args.videos_file)
    if not videos_file.exists():
        raise SystemExit(f"Videos file not found: {videos_file}")

    print(f"Loading video ids from {videos_file}...")
    video_ids = load_video_ids(videos_file, id_column=args.id_column)
    print(f"Loaded {len(video_ids):,} unique video ids")

    comments_dir = Path(args.comments_dir)
    if not comments_dir.exists() or not comments_dir.is_dir():
        raise SystemExit(f"Comments directory not found: {comments_dir}")

    exts = [e.strip().lower() for e in args.extensions.split(",") if e.strip()]

    files = [p for p in comments_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    if not files:
        print("No matching comment files found in", comments_dir)
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_in = 0
    total_out = 0
    for f in sorted(files):
        out_file = out_dir / f.name.replace(".csv", "_filtered_by_v4.csv")
        print(f"Filtering {f.name} -> {out_file.name} ...", end=" ")
        written = filter_file(f, out_file, video_ids, id_column=args.id_column)
        # try to count input rows quickly for reporting (not strictly required)
        print(f"wrote {written:,} rows")
        total_out += written

    print(f"Done. Total rows written: {total_out:,}. Filtered files are in {out_dir}")


if __name__ == "__main__":
    main()
