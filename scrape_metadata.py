import os
import re
import json
import argparse
import time
from typing import Iterable, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import pandas as pd

# Third-party: aiotube for public YouTube metadata
import aiotube


ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
write_lock = Lock()


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
	try:
		video = aiotube.Video(video_url)
		return dict(video.metadata or {})
	except Exception as e:
		print(f"Error fetching metadata for {video_id}: {e}")
		return {}


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


def process_single_video(row_idx: int, csv_row: dict, delay: float, error_log_path: str) -> Optional[dict]:
	"""Process a single video row and return combined metadata."""
	video_id = None
	try:
		# Extract video ID from the row
		for key in ['video_id', 'videoId', 'id']:
			if key in csv_row:
				video_id = parse_video_id(csv_row[key])
				if video_id:
					break
		
		if not video_id:
			# Try parsing from all values in the row
			for val in csv_row.values():
				video_id = parse_video_id(val)
				if video_id:
					break
		
		if not video_id:
			error_msg = "Could not find valid video ID"
			print(f"Row {row_idx}: {error_msg}")
			log_error(error_log_path, row_idx, "", error_msg)
			return None
		
		print(f"Row {row_idx}: Processing video ID {video_id}")
		
		# Fetch aiotube metadata
		aiotube_meta = fetch_aiotube_metadata(video_id)
		
		# Fetch channel metadata if available
		channel_id = aiotube_meta.get("author_id") or csv_row.get("channelId") or csv_row.get("channel_id")
		channel_meta = fetch_channel_metadata(channel_id) if channel_id else None
		
		# Combine metadata
		combined = combine_metadata(video_id, aiotube_meta, channel_meta, csv_row)
		combined['row_index'] = row_idx
		
		# Add delay to avoid rate limiting
		if delay > 0:
			time.sleep(delay)
		
		return combined
		
	except Exception as e:
		error_msg = f"Error processing - {str(e)}"
		print(f"Row {row_idx}: {error_msg}")
		log_error(error_log_path, row_idx, video_id or "", error_msg)
		return None


def save_to_csv(data: dict, output_path: str, mode: str = 'w'):
	"""Thread-safe CSV writing."""
	with write_lock:
		df = pd.DataFrame([data])
		if mode == 'a' and os.path.exists(output_path):
			df.to_csv(output_path, mode='a', header=False, index=False)
		else:
			df.to_csv(output_path, mode='w', header=True, index=False)


def log_error(error_log_path: str, row_num: int, video_id: str, error_msg: str):
	"""Thread-safe error logging."""
	with write_lock:
		timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
		with open(error_log_path, 'a', encoding='utf-8') as f:
			f.write(f"[{timestamp}] Row {row_num}: {video_id if video_id else 'UNKNOWN'} - {error_msg}\n")


def process_batch(csv_path: str, start_row: int, end_row: int, output_path: str, 
                  delay: float = 1.0, threads: int = 1, resume_row: Optional[int] = None):
	"""Process a batch of videos from the CSV."""
	
	# Determine error log path
	error_log_path = output_path.rsplit('.', 1)[0] + '_errors.txt'
	
	# Load the CSV
	print(f"Loading CSV from row {start_row} to {end_row}...")
	df = pd.read_csv(csv_path)
	total_rows = len(df)
	
	# Validate row ranges (1-based input, convert to 0-based for internal use)
	if start_row < 1 or start_row > total_rows:
		raise ValueError(f"start_row {start_row} is out of range (1-{total_rows})")
	if end_row < start_row or end_row > total_rows:
		raise ValueError(f"end_row {end_row} is invalid (must be >= {start_row} and <= {total_rows})")
	
	# Convert to 0-based indexing for internal use
	start_idx = start_row - 1
	end_idx = end_row - 1
	
	# Handle resume functionality
	actual_start_idx = start_idx
	actual_start_row = start_row
	mode = 'w'
	if resume_row is not None:
		if resume_row < start_row or resume_row > end_row:
			raise ValueError(f"resume_row {resume_row} must be between {start_row} and {end_row}")
		actual_start_idx = resume_row - 1
		actual_start_row = resume_row
		mode = 'a'
		print(f"Resuming from row {resume_row}...")
	
	# Extract the rows to process
	rows_to_process = df.iloc[actual_start_idx:end_idx + 1]
	
	print(f"Processing {len(rows_to_process)} videos with {threads} thread(s), delay: {delay}s")
	print(f"Output file: {output_path}")
	print(f"Error log: {error_log_path}")
	
	# Initialize error log
	if mode == 'w':
		with open(error_log_path, 'w', encoding='utf-8') as f:
			f.write(f"Error log started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
			f.write(f"Processing rows {start_row} to {end_row}\n")
			f.write("-" * 80 + "\n")
	
	# Process with threading
	processed_count = 0
	error_count = 0
	last_successful_row = actual_start_row - 1
	
	try:
		if threads == 1:
			# Single-threaded processing
			for row_num, (idx, row) in enumerate(rows_to_process.iterrows(), start=actual_start_row):
				result = process_single_video(row_num, row.to_dict(), delay, error_log_path)
				if result:
					save_to_csv(result, output_path, mode if row_num == actual_start_row else 'a')
					processed_count += 1
					last_successful_row = row_num
				else:
					error_count += 1
		else:
			# Multi-threaded processing
			with ThreadPoolExecutor(max_workers=threads) as executor:
				# Submit all tasks
				futures = {}
				for row_num, (idx, row) in enumerate(rows_to_process.iterrows(), start=actual_start_row):
					future = executor.submit(process_single_video, row_num, row.to_dict(), delay, error_log_path)
					futures[future] = row_num
				
				# Process completed tasks
				first_write = True
				for future in as_completed(futures):
					row_num = futures[future]
					try:
						result = future.result()
						if result:
							write_mode = mode if (first_write and mode == 'a') else ('w' if first_write else 'a')
							save_to_csv(result, output_path, write_mode)
							first_write = False
							processed_count += 1
							last_successful_row = max(last_successful_row, row_num)
						else:
							error_count += 1
					except Exception as e:
						error_msg = f"Exception in thread - {str(e)}"
						print(f"Row {row_num}: {error_msg}")
						log_error(error_log_path, row_num, "", error_msg)
						error_count += 1
		
		print(f"\nCompleted! Processed: {processed_count}, Errors: {error_count}")
		print(f"Results saved to: {output_path}")
		if error_count > 0:
			print(f"Errors logged to: {error_log_path}")
			
	except KeyboardInterrupt:
		print(f"\n\n{'='*80}")
		print("Process interrupted by user!")
		print(f"{'='*80}")
		print(f"\nProcessed: {processed_count}, Errors: {error_count}")
		print(f"Last successful row: {last_successful_row}")
		print(f"\nTo resume from the next row, use:")
		print(f"  python scrape_metadata.py --resume-row {last_successful_row + 1} --end-row {end_row} --output {output_path}")
		print(f"\nError log: {error_log_path}")
		print(f"{'='*80}\n")
		raise
	except Exception as e:
		print(f"\n\n{'='*80}")
		print(f"Fatal error occurred: {str(e)}")
		print(f"{'='*80}")
		print(f"\nProcessed: {processed_count}, Errors: {error_count}")
		print(f"Last successful row: {last_successful_row}")
		print(f"\nTo resume from the next row, use:")
		print(f"  python scrape_metadata.py --resume-row {last_successful_row + 1} --end-row {end_row} --output {output_path}")
		print(f"\nError log: {error_log_path}")
		print(f"{'='*80}\n")
		raise


def main():
	parser = argparse.ArgumentParser(
		description='Fetch YouTube metadata for videos in the dislike dataset',
		formatter_class=argparse.RawDescriptionHelpFormatter,
		epilog="""
Examples:
  # Process rows 1-100 with default settings
  python scrape_metadata.py --start-row 1 --end-row 100

  # Process with 5 threads and 2 second delay
  python scrape_metadata.py --start-row 1 --end-row 1000 --threads 5 --delay 2

  # Resume from row 500 to end of previous job (end-row auto-detected)
  python scrape_metadata.py --resume-row 500 --end-row 1000 --output combined_metadata.csv
		"""
	)
	
	parser.add_argument('--start-row', type=int, default=None,
	                    help='Starting row index (1-based, inclusive). Not required if using --resume-row')
	parser.add_argument('--end-row', type=int, required=True,
	                    help='Ending row index (1-based, inclusive)')
	parser.add_argument('--delay', type=float, default=1.0,
	                    help='Delay in seconds between requests (default: 1.0)')
	parser.add_argument('--threads', type=int, default=1,
	                    help='Number of threads for parallel processing (default: 1)')
	parser.add_argument('--resume-row', type=int, default=None,
	                    help='Resume from this row, appending to existing output file. If specified, --start-row is not required.')
	parser.add_argument('--input', type=str, default='youtube_dislike_dataset.csv',
	                    help='Input CSV file (default: youtube_dislike_dataset.csv)')
	parser.add_argument('--output', type=str, default='combined_metadata.csv',
	                    help='Output CSV file (default: combined_metadata.csv)')
	
	args = parser.parse_args()
	
	# Determine start_row: if resume_row is specified, start_row defaults to resume_row
	if args.resume_row is not None:
		start_row = args.resume_row if args.start_row is None else args.start_row
		resume_row = args.resume_row
	elif args.start_row is not None:
		start_row = args.start_row
		resume_row = None
	else:
		parser.error("Either --start-row or --resume-row must be specified")
	
	# Resolve paths
	base_dir = os.path.dirname(__file__)
	csv_path = os.path.join(base_dir, args.input)
	output_path = os.path.join(base_dir, args.output)
	
	if not os.path.exists(csv_path):
		raise FileNotFoundError(f"CSV not found: {csv_path}")
	
	# Process the batch
	process_batch(
		csv_path=csv_path,
		start_row=start_row,
		end_row=args.end_row,
		output_path=output_path,
		delay=args.delay,
		threads=args.threads,
		resume_row=resume_row
	)


if __name__ == "__main__":
	main()
