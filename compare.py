import os
import re
import json
from typing import Iterable, Optional

import pandas as pd

# Third-party: aiotube for public YouTube metadata
import aiotube


ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


def parse_video_id(value: object) -> Optional[str]:
	"""Try to parse a YouTube video ID from a value (ID or URL)."""
	if value is None:
		return None
	s = str(value).strip()
	if not s or s.lower() == "nan":
		return None
	# If it's a plain ID
	if ID_RE.match(s):
		return s
	# Try to extract from common URL patterns
	# Examples:
	# - https://www.youtube.com/watch?v=VIDEOID
	# - https://youtu.be/VIDEOID
	# - https://www.youtube.com/shorts/VIDEOID
	# - https://www.youtube.com/embed/VIDEOID
	# - additional params after id
	# Find watch?v=
	m = re.search(r"[?&]v=([A-Za-z0-9_-]{11})", s)
	if m:
		return m.group(1)
	# youtu.be/<id>
	m = re.search(r"youtu\.be/([A-Za-z0-9_-]{11})", s)
	if m:
		return m.group(1)
	# shorts/<id>, embed/<id>
	m = re.search(r"/(shorts|embed)/([A-Za-z0-9_-]{11})(?:\b|/|\?|#|$)", s)
	if m:
		return m.group(2)
	return None


def load_first_video_id(csv_path: str) -> str:
	"""
	Scan the CSV to find the first valid YouTube video ID from likely columns.
	Efficiently uses chunks to avoid loading the entire file.
	"""
	# Read header to identify candidate columns
	header = pd.read_csv(csv_path, nrows=0)
	cols = list(header.columns)
	lower_map = {c.lower(): c for c in cols}
	preferred_order = [
		"video_id",
		"videoid",
		"id",
		"video",
		"video_url",
		"url",
		"link",
	]
	present_candidates = [lower_map[c] for c in preferred_order if c in lower_map]
	if not present_candidates:
		present_candidates = cols  # fallback: try every column

	# Pass 1: iterate candidate columns only
	for col in present_candidates:
		for chunk in pd.read_csv(csv_path, usecols=[col], chunksize=5000):
			for val in chunk[col].tolist():
				vid = parse_video_id(val)
				if vid:
					print(f"Detected video_id from column '{col}': {vid}")
					return vid

	# Pass 2: brute-force across all columns
	for chunk in pd.read_csv(csv_path, chunksize=5000):
		for _, row in chunk.iterrows():
			for val in row.values.tolist():
				vid = parse_video_id(val)
				if vid:
					print("Detected video_id by brute-force scan:", vid)
					return vid

	raise RuntimeError("Could not find a valid YouTube video ID in the CSV")


def fetch_aiotube_metadata(video_id: str) -> dict:
	"""Fetch metadata dict for a given video id using aiotube.
	Note: aiotube.Video accepts a full YouTube URL or a plain 11-character ID.
	Some valid YouTube IDs include '-', which plain-ID regex may not match,
	so we pass a youtu.be URL to be safe.
	"""
	video_url = f"https://youtu.be/{video_id}"
	video = aiotube.Video(video_url)
	return dict(video.metadata or {})


def fetch_channel_metadata(channel_identifier: str) -> Optional[dict]:
	"""Fetch channel metadata by channel ID/URL/handle using aiotube."""
	if not channel_identifier:
		return None
	try:
		ch = aiotube.Channel(channel_identifier)
		return dict(ch.metadata or {})
	except Exception:
		return None


def read_csv_row(csv_path: str) -> dict:
	"""
	Read the first row of the CSV as a dict for comparison.
	"""
	df = pd.read_csv(csv_path, nrows=1)
	row = df.iloc[0].to_dict()
	return row


def pick_key_fields(meta: dict) -> dict:
	"""Extract comparable, human-friendly keys from aiotube metadata.
	aiotube.Video.metadata provides keys: title, id, views, streamed, duration,
	author_id, upload_date, url, thumbnails, tags, description, likes?, genre?
	"""
	mapping = {
		"id": "id",
		"title": "title",
		"views": "views",
		"likes": "likes",
		"duration": "duration",
		"upload_date": "upload_date",
		"author_id": "author_id",
		"genre": "genre",
		"url": "url",
	}
	out = {k: meta.get(v) for k, v in mapping.items() if v in meta}
	return out


def pick_channel_fields(meta: Optional[dict]) -> Optional[dict]:
	if not meta:
		return None
	mapping = {
		"id": "id",
		"name": "name",
		"subscribers": "subscribers",
		"views": "views",
		"created_at": "created_at",
		"country": "country",
		"verified": "verified",
		"url": "url",
	}
	return {k: meta.get(v) for k, v in mapping.items() if v in meta}


def combine_metadata(video_id: str, aiotube_video: dict, aiotube_channel: Optional[dict], csv_row: dict) -> dict:
	combined = {
		"video_id": video_id,
	}
	# CSV fields (prefixed)
	for k, v in csv_row.items():
		combined[f"csv_{k}"] = v
	# aiotube video fields (prefixed)
	for k, v in (aiotube_video or {}).items():
		combined[f"aiotube_video_{k}"] = v
	# aiotube channel fields (prefixed)
	for k, v in (aiotube_channel or {}).items():
		combined[f"aiotube_channel_{k}"] = v
	return combined


def main():
	base_dir = os.path.dirname(__file__)
	csv_path = os.path.join(base_dir, "youtube_dislike_dataset.csv")
	if not os.path.exists(csv_path):
		raise FileNotFoundError(f"CSV not found: {csv_path}")

	# 1) Get first video id from CSV
	video_id = load_first_video_id(csv_path)

	# 2) Fetch aiotube metadata for the video
	aiotube_meta = fetch_aiotube_metadata(video_id)

	# 3) Read the CSV row for comparison
	csv_row = read_csv_row(csv_path)

	# 4) Prepare a concise comparison
	aiotube_display = pick_key_fields(aiotube_meta)

	print("First video ID from CSV:", video_id)
	print("\nAIOTube metadata (key fields):")
	print(json.dumps(aiotube_display, indent=2, ensure_ascii=False))

	print("\nCSV first-row fields:")
	# Show a subset if the row is huge
	sample_keys = [
		"video_id",
		"title",
		"channelTitle",
		"channelId",
		"view_count",
		"like_count",
		"dislike_count",
		"comment_count",
		"published_at",
		"duration",
		"category",
	]
	csv_subset = {k: csv_row[k] for k in sample_keys if k in csv_row}
	if not csv_subset:  # fallback: show entire row
		csv_subset = csv_row
	print(json.dumps(csv_subset, indent=2, ensure_ascii=False))

	# 5) Fetch channel metadata (using aiotube video author_id or CSV channelId)
	channel_id = aiotube_meta.get("author_id") or csv_row.get("channelId") or csv_row.get("channel_id")
	channel_meta = fetch_channel_metadata(channel_id) if channel_id else None
	channel_display = pick_channel_fields(channel_meta)
	print("\nAIOTube channel metadata (key fields):")
	print(json.dumps(channel_display or {"note": "channel metadata unavailable"}, indent=2, ensure_ascii=False))

	# 6) Save combined metadata to a new CSV
	combined = combine_metadata(video_id, aiotube_meta, channel_meta, csv_row)
	out_path = os.path.join(base_dir, "combined_metadata_first_video.csv")
	pd.DataFrame([combined]).to_csv(out_path, index=False)
	print(f"\nSaved combined metadata to: {out_path}")


if __name__ == "__main__":
	main()
