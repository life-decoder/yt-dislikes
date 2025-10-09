import csv
import os
import re
import sys
import argparse
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import islice
from urllib.parse import urlparse, parse_qs
from typing import cast

# Sentiment analysis
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

try:
	# Package name is youtube-comment-downloader; import module with underscore
	from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_RECENT
except Exception:
	print("youtube-comment-downloader is not installed or failed to import.")
	print("Install with: pip install youtube-comment-downloader")
	raise


def extract_video_id(value: str) -> str | None:
	"""Try to extract an 11-char YouTube video ID from a raw ID or URL.

	Supports:
	- Plain 11-char ID (A-Za-z0-9_-)  
	- https://www.youtube.com/watch?v=ID  
	- https://youtu.be/ID
	"""
	if not value:
		return None

	value = value.strip().strip('"\'')

	# If it's already an 11-char ID
	if re.fullmatch(r"[A-Za-z0-9_-]{11}", value):
		return value

	# If it's a URL, try to parse
	try:
		parsed = urlparse(value)
		if parsed.netloc:
			# Long URL format: youtube.com/watch?v=ID
			qs = parse_qs(parsed.query)
			if 'v' in qs and qs['v']:
				vid = qs['v'][0]
				if re.fullmatch(r"[A-Za-z0-9_-]{11}", vid):
					return vid

			# Short URL format: youtu.be/ID
			path_parts = parsed.path.strip('/').split('/')
			if path_parts:
				cand = path_parts[-1]
				if re.fullmatch(r"[A-Za-z0-9_-]{11}", cand):
					return cand
	except Exception:
		pass

	return None


def _infer_video_id_from_row(row: dict, fieldnames: list[str]) -> str | None:
	"""Infer a YouTube video ID from a CSV row using common column names or any value that parses as an ID/URL."""
	preferred = [
		'video_id', 'videoId', 'Video ID', 'video.id', 'yt_id', 'youtube_id', 'id', 'Id', 'ID', 'videoIdParsed'
	]
	lower_map = {fn.lower(): fn for fn in fieldnames}

	# 1) Preferred columns (case-insensitive lookup)
	for key in preferred:
		actual = lower_map.get(key.lower())
		if actual and actual in row:
			vid = extract_video_id(row.get(actual, '') or '')
			if vid:
				return vid

	# 2) Any column whose value looks like a video ID or URL
	for name in fieldnames:
		vid = extract_video_id(row.get(name, '') or '')
		if vid:
			return vid
	return None


def iterate_video_ids_from_csv(csv_path: str, dedup: bool = True, start_row: int = 1, with_row_index: bool = False):
	"""Yield video IDs found in the CSV by scanning rows. Optionally de-duplicates sequentially.

	Args:
	- csv_path: Path to CSV with a header row.
	- dedup: If True, de-duplicate video IDs.
	- start_row: 1-based data row index to start from (rows after the header). 1 means the first data row.
	- with_row_index: If True, yield tuples of (video_id, row_index) where row_index is 1-based data row.

	Tries utf-8-sig then utf-8 for BOM safety. Skips rows where no valid ID is found.
	"""
	encodings = ['utf-8-sig', 'utf-8']
	last_err = None
	seen = set()
	for enc in encodings:
		try:
			with open(csv_path, 'r', encoding=enc, newline='') as f:
				reader = csv.DictReader(f)
				if not reader.fieldnames:
					raise ValueError("CSV has no header/fieldnames.")
				fieldnames = [fn.strip() for fn in reader.fieldnames]
				for idx, row in enumerate(reader, start=1):
					if idx < start_row:
						continue
					vid = _infer_video_id_from_row(row, fieldnames)
					if not vid:
						continue
					if dedup:
						if vid in seen:
							continue
						seen.add(vid)
					yield (vid, idx) if with_row_index else vid
			return
		except Exception as e:
			last_err = e
			continue
	# If both encodings failed entirely
	raise last_err or RuntimeError("Failed to read CSV.")


def fetch_comments(video_id: str, max_comments: int = 100):
	"""Yield up to max_comments comment dicts for the given video ID using youtube-comment-downloader."""
	url = f"https://www.youtube.com/watch?v={video_id}"
	downloader = YoutubeCommentDownloader()
	# SORT_BY_RECENT to avoid requiring popularity ordering which may be slower
	gen = downloader.get_comments_from_url(url, sort_by=SORT_BY_RECENT)
	yield from islice(gen, max_comments)


def main():
	parser = argparse.ArgumentParser(description="Fetch YouTube comments and compute VADER sentiment per video.")
	parser.add_argument('--all', dest='process_all', action='store_true', help='Process all videos in the CSV')
	parser.add_argument('--limit', type=int, default=1, help='Number of videos to process (ignored if --all)')
	parser.add_argument('--max-comments', type=int, default=50, help='Max comments to fetch per video')
	parser.add_argument('--csv', type=str, default='youtube_dislike_dataset.csv', help='Path to the videos CSV')
	parser.add_argument('--out', type=str, default='comments_sentiment_all.csv', help='Output CSV for all processed videos')
	parser.add_argument('--append', action='store_true', help='Append to output file if it exists (default overwrites)')
	parser.add_argument('--workers', type=int, default=1, help='Number of worker threads to use')
	parser.add_argument('--delay', type=float, default=1.0, help='Delay in seconds between requests (default: 1.0)')
	parser.add_argument('--start-row', type=int, default=1, help='1-based CSV data row to start from (after header)')
	parser.add_argument('--end-row', type=int, help='1-based CSV data row to end at (inclusive). Must be >= start-row.')
	parser.add_argument('--resume-row', type=int, help='Shorthand for resuming: equivalent to --start-row N --append')
	args = parser.parse_args()

	# Normalize resume-row behavior: set start_row and force append
	if getattr(args, 'resume_row', None):
		if args.resume_row < 1:
			print("--resume-row must be >= 1", file=sys.stderr)
			sys.exit(2)
		args.start_row = args.resume_row
		args.append = True

	# Validate end-row
	if getattr(args, 'end_row', None):
		if args.end_row < 1:
			print("--end-row must be >= 1", file=sys.stderr)
			sys.exit(2)
		if args.end_row < args.start_row:
			print(f"--end-row ({args.end_row}) must be >= --start-row ({args.start_row})", file=sys.stderr)
			sys.exit(2)

	csv_path = os.path.join(os.path.dirname(__file__), args.csv)
	if not os.path.exists(csv_path):
		print(f"CSV not found at {csv_path}", file=sys.stderr)
		sys.exit(1)

	# Initialize VADER (download lexicon if missing)
	try:
		nltk.data.find('sentiment/vader_lexicon.zip')
	except LookupError:
		nltk.download('vader_lexicon', quiet=True)
	# Ensure VADER resources are available; per-thread analyzers are created later.

	# Collect the list of (video_id, row_index) to process
	video_entries: list[tuple[str, int]] = []
	for vid, row_idx in iterate_video_ids_from_csv(csv_path, start_row=max(1, args.start_row), with_row_index=True):
		# Check if we've passed the end row
		if getattr(args, 'end_row', None) and row_idx > args.end_row:
			break
		video_entries.append((vid, cast(int, row_idx)))
		if not args.process_all and len(video_entries) >= args.limit:
			break

	# Build description of what we're processing
	if getattr(args, 'end_row', None):
		range_desc = f"rows {args.start_row}-{args.end_row}"
	else:
		range_desc = f"starting at row {max(1, args.start_row)}"
	
	total_target = len(video_entries) if not args.process_all else f"ALL ({len(video_entries)})"
	print(f"Starting sentiment extraction for {total_target} videos (max {args.max_comments} comments each) with {args.workers} worker(s) {range_desc}…")

	fieldnames = ['video_id', 'comment_index', 'cid', 'author', 'time', 'votes', 'text', 'pos', 'neu', 'neg', 'compound']
	mode = 'a' if args.append else 'w'
	out_path = os.path.join(os.path.dirname(__file__), args.out)
	wrote_header = False
	if args.append and os.path.exists(out_path):
		wrote_header = True  # assume existing file already has a header

	write_lock = threading.Lock()

	stop_event = threading.Event()
	
	last_request_time = [0.0]  # Use list for mutable shared state

	def process_video(video_id: str, row_index: int) -> tuple[str, int, int, str | None]:
		"""Fetch comments for a video, compute sentiment, and write rows under lock.
		Returns (video_id, row_index, n_rows, error)."""
		if stop_event.is_set():
			return video_id, row_index, 0, "skipped_due_to_prior_error"
		
		# Apply delay between requests
		with write_lock:
			elapsed = time.time() - last_request_time[0]
			if elapsed < args.delay:
				time.sleep(args.delay - elapsed)
			last_request_time[0] = time.time()
		
		try:
			comments = list(fetch_comments(video_id, max_comments=args.max_comments))
		except Exception as e:
			stop_event.set()
			return video_id, row_index, 0, f"fetch_error: {e}"

		local_sia = SentimentIntensityAnalyzer()
		rows_written = 0
		try:
			with write_lock:
				for idx, c in enumerate(comments, start=1):
					text = (c.get('text') or '').strip()
					scores = local_sia.polarity_scores(text or '')
					writer.writerow({
						'video_id': video_id,
						'comment_index': idx,
						'cid': c.get('cid'),
						'author': c.get('author'),
						'time': c.get('time'),
						'votes': c.get('votes'),
						'text': text.replace('\n', ' '),
						'pos': scores.get('pos', 0.0),
						'neu': scores.get('neu', 0.0),
						'neg': scores.get('neg', 0.0),
						'compound': scores.get('compound', 0.0),
					})
					rows_written += 1
				f.flush()
		except Exception as e:
			return video_id, row_index, rows_written, f"write_error: {e}"
		return video_id, row_index, rows_written, None

	processed = 0
	last_processed_row = None
	try:
		with open(out_path, mode, encoding='utf-8', newline='') as f:
			writer = csv.DictWriter(f, fieldnames=fieldnames)
			if not wrote_header:
				writer.writeheader()
				f.flush()

			with ThreadPoolExecutor(max_workers=args.workers) as executor:
				futures = {executor.submit(process_video, vid, row_idx): (vid, row_idx) for vid, row_idx in video_entries}
				fail_fast_triggered = False
				try:
					for fut in as_completed(futures):
						vid, row_idx = futures[fut]
						try:
							v_id, r_idx, n_rows, err = fut.result()
						except Exception as e:
							processed += 1
							last_processed_row = row_idx
							print(f"[{processed}] Unexpected error for {vid} (row {row_idx}): {e}", file=sys.stderr)
							continue
						if err:
							processed += 1
							last_processed_row = r_idx
							if err.startswith('fetch_error') and not fail_fast_triggered:
								fail_fast_triggered = True
								stop_event.set()
								print(f"[{processed}] {v_id} (row {r_idx}): {err}", file=sys.stderr)
								print(f"Stopping due to fetch error. Re-run with --start-row {r_idx} to resume from this row.", file=sys.stderr)
								# Attempt to cancel any pending futures
								for f2, (v2, r2) in futures.items():
									if f2 is not fut and not f2.done():
										f2.cancel()
								# Best-effort shutdown (Py>=3.9 supports cancel_futures)
								try:
									executor.shutdown(wait=False, cancel_futures=True)
								except TypeError:
									# Python < 3.9 fallback (no cancel_futures)
									pass
								break
							else:
								print(f"[{processed}] {v_id} (row {r_idx}): {err}", file=sys.stderr)
						else:
							processed += 1
							last_processed_row = r_idx
							print(f"[{processed}] {v_id} (row {r_idx}): saved {n_rows} comments to {out_path}")
				except KeyboardInterrupt:
					print("\n\nInterrupted by user (Ctrl+C). Stopping gracefully...", file=sys.stderr)
					stop_event.set()
					# Cancel pending futures
					for f2, (v2, r2) in futures.items():
						if not f2.done():
							f2.cancel()
					# Best-effort shutdown
					try:
						executor.shutdown(wait=False, cancel_futures=True)
					except TypeError:
						pass
					if last_processed_row:
						print(f"Processed {processed} video(s) before interruption.", file=sys.stderr)
						print(f"To resume, use: --start-row {last_processed_row + 1} --append", file=sys.stderr)
					sys.exit(130)  # Standard exit code for SIGINT
	except KeyboardInterrupt:
		print("\n\nInterrupted by user (Ctrl+C).", file=sys.stderr)
		if last_processed_row:
			print(f"Processed {processed} video(s) before interruption.", file=sys.stderr)
			print(f"To resume, use: --start-row {last_processed_row + 1} --append", file=sys.stderr)
		sys.exit(130)
	except Exception as e:
		print(f"Failed to write output CSV {out_path}: {e}", file=sys.stderr)
		sys.exit(1)

	print(f"Done. Successfully processed {processed} video(s). Output: {out_path}")


if __name__ == '__main__':
	main()
