"""Combine additional metadata CSVs, exclude records missing genre or duration,
convert genre strings to integer ids, and write combined outputs.

Outputs:
- combined_additional_metadata.csv: filtered combined rows with genre_id column
- genre_mapping.json: {genre: id}
- genre_mapping.csv: two-column CSV for easy inspection

This script is intentionally conservative in parsing: it reads CSVs as strings,
coerces duration to numeric, and treats empty strings or non-numeric durations
as missing.
"""
from __future__ import annotations

import glob
import html
import json
import os
from typing import Dict

import pandas as pd


def find_input_files(directory: str) -> list[str]:
    pattern = os.path.join(directory, "*additional_metadata*.csv")
    files = sorted(glob.glob(pattern))
    return files


def read_and_normalize(path: str) -> pd.DataFrame:
    # Read everything as string to avoid surprising dtype inferences
    df = pd.read_csv(path, dtype=str)

    # Normalize expected columns if present
    expected = ["video_id", "duration", "genre", "row_index"]
    present = [c for c in expected if c in df.columns]
    if present:
        df = df[present].copy()

    # Trim whitespace
    if "genre" in df.columns:
        df["genre"] = df["genre"].astype(str).str.strip()
        # Unescape HTML entities (e.g. &amp;)
        df["genre"] = df["genre"].apply(lambda s: html.unescape(s) if pd.notna(s) else s)
        # Treat empty strings and the literal 'nan' as missing
        df.loc[df["genre"].isin(["", "nan", "None"]), "genre"] = pd.NA

    if "duration" in df.columns:
        # Coerce to numeric; invalid or missing durations become NaN
        df["duration"] = pd.to_numeric(df["duration"], errors="coerce")

    return df


def combine_files(files: list[str]) -> pd.DataFrame:
    dfs = []
    for f in files:
        try:
            df = read_and_normalize(f)
            df["_source_file"] = os.path.basename(f)
            dfs.append(df)
        except Exception as e:
            print(f"Warning: failed to read {f}: {e}")

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True, sort=False)
    return combined


def build_genre_mapping(genres: pd.Series) -> Dict[str, int]:
    unique = sorted(g for g in genres.dropna().unique())
    mapping: Dict[str, int] = {g: i for i, g in enumerate(unique)}
    return mapping


def main():
    base_dir = os.path.dirname(__file__)
    files = find_input_files(base_dir)

    if not files:
        print("No input files found in", base_dir)
        return

    print(f"Found {len(files)} files. Combining...")
    combined = combine_files(files)
    total_rows = len(combined)

    if total_rows == 0:
        print("No rows read from input files.")
        return

    # Filter out rows missing genre or duration
    before_drop = len(combined)
    filtered = combined.dropna(subset=["genre", "duration"]).copy()
    after_drop = len(filtered)

    print(f"Rows read: {before_drop}. Rows after dropping missing genre/duration: {after_drop}.")

    # Build mapping and add genre_id
    mapping = build_genre_mapping(filtered["genre"]) if "genre" in filtered.columns else {}
    filtered["genre_id"] = filtered["genre"].map(mapping)

    # Prepare output paths
    out_combined = os.path.join(base_dir, "combined_additional_metadata.csv")
    out_mapping_json = os.path.join(base_dir, "genre_mapping.json")
    out_mapping_csv = os.path.join(base_dir, "genre_mapping.csv")

    # Write combined CSV (choose useful column order)
    cols = []
    for c in ["video_id", "duration", "genre", "genre_id", "row_index", "_source_file"]:
        if c in filtered.columns:
            cols.append(c)

    filtered.to_csv(out_combined, columns=cols, index=False)
    print(f"Wrote combined filtered data to: {out_combined}")

    # Write mapping JSON and CSV
    with open(out_mapping_json, "w", encoding="utf-8") as fh:
        json.dump(mapping, fh, ensure_ascii=False, indent=2)
    print(f"Wrote genre mapping to: {out_mapping_json}")

    # Also CSV for readability
    pd.DataFrame(list(mapping.items()), columns=["genre", "genre_id"]).to_csv(out_mapping_csv, index=False)
    print(f"Wrote genre mapping CSV to: {out_mapping_csv}")


if __name__ == "__main__":
    main()
