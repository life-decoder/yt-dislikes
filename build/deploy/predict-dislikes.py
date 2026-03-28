import joblib
import pandas as pd
import os
import numpy as np
import sys
import traceback
import re
import aiotube
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from datetime import datetime, timezone
from dateutil import parser

def fetch_metadata(video_id, api_key):
    try:
        # Build the YouTube API client
        youtube = build('youtube', 'v3', developerKey=api_key)

        # Request video statistics
        request = youtube.videos().list(
            part='statistics',
            id=video_id
        )
        response = request.execute()

        # Extract data
        if not response['items']:
            print("Video not found.")
            return None

        stats = response['items'][0]['statistics']

        view_count = int(stats.get('viewCount', 0))
        like_count = int(stats.get('likeCount', 0))
        comment_count = int(stats.get('commentCount', 0)) # Already provided by API

        #fetch more metadata
        video = aiotube.Video(video_id)
        #print(video.metadata.keys(), end='\n\n')   # video metadata in dict format
        
        video_data = {
            "likes": like_count,
            "view_count": view_count,
            "duration": video.metadata["duration"],
            "upload_date": video.metadata["upload_date"],
            "comment_count": comment_count,
            "genre": video.metadata["genre"]
        }

        return  video_data

    except HttpError as e:
        print(f"An HTTP error occurred: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage

def preprocess_data(df):
    df_processed = df.copy() # Handle missing values
    df_processed['log_comment_count'] = df_processed['log_comment_count'].fillna(df_processed['log_comment_count'].median())
    df_processed['log_likes'] = df_processed['log_likes'].fillna(df_processed['log_likes'].median())
    return df_processed


def predict_dislikes(sample_data):
    # Define the path to the saved pipeline
    output_dir = os.path.dirname(os.path.abspath(__file__))
    pipeline_path = os.path.join(output_dir, 'full_xgboost_pipeline.pkl')

    # Load the pipeline
    try:
        loaded_pipeline = joblib.load(pipeline_path)
        print("Pipeline loaded successfully.")
    except Exception as e:
        print("ERROR: failed to load pipeline:", e)
        traceback.print_exc()
        print("\nIf the error mentions missing functions like 'preprocess_data',")
        print("ensure the original preprocessing function is available (either")
        print("implement it here or import it from the module where it was defined).")
        sys.exit(1)


    """ sample_data = {
        #"dislikes": [955],
        "age": [287],
        "avg_compound": [0.042724],
        "avg_neg": [0.03814],
        "avg_neu": [0.89316],
        "avg_pos": [0.0687],
        "comment_count": [2086],
        "comment_sample_size": [50],
        "likes": [23342],
        "view_count": [1355784],
        "no_comments": [0],
        "duration": [397],
        "genre_id": [13],
        "log_comment_count": [3.319314304],
        "log_likes": [10.05805243],
        #"log_dislikes": [6.862757913],
        "log_view_count": [14.11989118],
        "view_like_ratio": [58.08096646],
        "log_view_like_ratio": [4.078908816]
    }
    """
    sample_df = pd.DataFrame(sample_data)

    # Ensure columns are numeric where expected
    for col in sample_df.columns:
        # try to coerce object columns that represent numbers
        if sample_df[col].dtype == object:
            try:
                sample_df[col] = pd.to_numeric(sample_df[col])
            except Exception:
                pass

    # Try to get the model's expected feature order so XGBoost doesn't complain
    expected_features = None
    try:
        final_est = loaded_pipeline.steps[-1][1]
        # XGBoost booster stores feature names
        if hasattr(final_est, 'get_booster'):
            booster = final_est.get_booster()
            expected_features = getattr(booster, 'feature_names', None)
        # scikit-learn may expose feature_names_in_
        if expected_features is None and hasattr(final_est, 'feature_names_in_'):
            expected_features = list(final_est.feature_names_in_)
    except Exception:
        expected_features = None

    if expected_features:
        # Add any missing expected features with safe defaults and reorder columns
        for f in expected_features:
            if f not in sample_df.columns:
                # default numeric value 0 or nan for averages
                sample_df[f] = 0

        # Reorder to expected order
        sample_df = sample_df[expected_features]

    # (pre-predict debug prints removed)

    # Predict dislikes using the loaded pipeline
    predicted_log_dislikes = loaded_pipeline.predict(sample_df)

    # Inverse transform the log predictions to get the actual predicted dislikes
    predicted_dislikes = np.expm1(predicted_log_dislikes)

    print("\nPredicted Log Dislikes:")
    print(predicted_log_dislikes)

    print("\nPredicted Dislikes:")
    print(predicted_dislikes)


def add_sentiment_scores(video_data, video_id: str | None, max_comments: int = 50):
    """Fetch up to `max_comments` comments for the given video_id and compute
    VADER averages: avg_compound, avg_pos, avg_neu, avg_neg, and
    comment_sample_size. Adds 'no_comments' flag (1 if zero fetched).
    Returns modified video_data.
    """

    # Prefer explicit video_id parameter, fall back to video_data if not provided
    vid = video_id or video_data.get('video_id') or video_data.get('id') or video_data.get('VIDEO_ID')
    if not vid:
        print("add_sentiment_scores: video_id not provided; skipping sentiment.")
        video_data.setdefault('avg_compound', None)
        video_data.setdefault('avg_pos', None)
        video_data.setdefault('avg_neu', None)
        video_data.setdefault('avg_neg', None)
        video_data.setdefault('comment_sample_size', 0)
        video_data.setdefault('no_comments', 1)
        return video_data

    # Import heavy deps locally so missing packages are handled gracefully
    try:
        from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_RECENT
    except Exception as e:
        print("youtube-comment-downloader is not available; cannot fetch comments:", e)
        video_data.setdefault('avg_compound', None)
        video_data.setdefault('avg_pos', None)
        video_data.setdefault('avg_neu', None)
        video_data.setdefault('avg_neg', None)
        video_data.setdefault('comment_sample_size', 0)
        video_data.setdefault('no_comments', 1)
        return video_data

    try:
        import nltk
        from nltk.sentiment import SentimentIntensityAnalyzer
        # ensure lexicon is present
        try:
            nltk.data.find('sentiment/vader_lexicon.zip')
        except LookupError:
            nltk.download('vader_lexicon', quiet=True)
    except Exception as e:
        print("NLTK or VADER not available; cannot compute sentiment:", e)
        video_data.setdefault('avg_compound', None)
        video_data.setdefault('avg_pos', None)
        video_data.setdefault('avg_neu', None)
        video_data.setdefault('avg_neg', None)
        video_data.setdefault('comment_sample_size', 0)
        video_data.setdefault('no_comments', 1)
        return video_data

    downloader = YoutubeCommentDownloader()
    url = f"https://www.youtube.com/watch?v={vid}"
    gen = downloader.get_comments_from_url(url, sort_by=SORT_BY_RECENT)

    sia = SentimentIntensityAnalyzer()
    count = 0
    sum_compound = 0.0
    sum_pos = 0.0
    sum_neu = 0.0
    sum_neg = 0.0
    most_neg = (1.0, '')  # (compound score, text)
    most_pos = (-1.0, '')  # (compound score, text)

    try:
        for c in gen:
            if count >= max_comments:
                break
            text = (c.get('text') or '').strip()
            # compute scores
            scores = sia.polarity_scores(text or '')
            compound_score = scores.get('compound', 0.0)
            sum_compound += compound_score
            if compound_score < most_neg[0]:
                most_neg = (compound_score, text)
            if compound_score > most_pos[0]:
                most_pos = (compound_score, text)
            sum_pos += scores.get('pos', 0.0)
            sum_neu += scores.get('neu', 0.0)
            sum_neg += scores.get('neg', 0.0)
            count += 1
    except Exception as e:
        # If fetching comments fails, log and continue with what we have
        print(f"add_sentiment_scores: error fetching comments for {vid}: {e}")

    # Compute averages or set None when no comments
    if count > 0:
        video_data['avg_compound'] = float(sum_compound / count)
        video_data['avg_pos'] = float(sum_pos / count)
        video_data['avg_neu'] = float(sum_neu / count)
        video_data['avg_neg'] = float(sum_neg / count)
    else:
        video_data['avg_compound'] = None
        video_data['avg_pos'] = None
        video_data['avg_neu'] = None
        video_data['avg_neg'] = None

    video_data['comment_sample_size'] = count
    video_data['no_comments'] = 1 if count == 0 else 0
    
    return most_neg, most_pos


def add_features(video_data):
    # Add log-transformed features
    upload = video_data.get('upload_date')
    age_days = np.nan
    if upload:
        try:
            # parse ISO 8601 with offset like "2021-02-27T19:57:04-08:00"
            upload_dt = datetime.fromisoformat(upload)
            if upload_dt.tzinfo is None:
                upload_dt = upload_dt.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            age_days = (now - upload_dt.astimezone(timezone.utc)).total_seconds() / 86400.0
        except Exception:
            # fallback to dateutil if available
            try:
                upload_dt = parser.isoparse(upload)
                now = datetime.now(timezone.utc)
                age_days = (now - upload_dt.astimezone(timezone.utc)).total_seconds() / 86400.0
            except Exception:
                age_days = np.nan

        genres = {
            "Autos & Vehicles": 0,
            "Comedy": 1,
            "Education": 2,
            "Entertainment": 3,
            "Film & Animation": 4,
            "Gaming": 5,
            "Howto & Style": 6,
            "Music": 7,
            "News & Politics": 8,
            "Nonprofits & Activism": 9,
            "People & Blogs": 10,
            "Pets & Animals": 11,
            "Science & Technology": 12,
            "Sports": 13,
            "Travel & Events": 14
            }
    
    # Coerce numeric fields saved by external libraries (they are often strings)
    def to_int_safe(val, default=0):
        try:
            if val is None:
                return default
            return int(float(val))
        except Exception:
            return default

    def to_float_safe(val, default=0.0):
        try:
            if val is None:
                return default
            return float(val)
        except Exception:
            return default

    # Ensure basic counts are numeric
    video_data['likes'] = to_int_safe(video_data.get('likes', 0), 0)
    video_data['view_count'] = to_int_safe(video_data.get('view_count', 0), 0)
    video_data['comment_count'] = to_int_safe(video_data.get('comment_count', 0), 0)
    video_data['duration'] = to_int_safe(video_data.get('duration', 0), 0)

    # Map genre to id if possible
    try:
        video_data['genre_id'] = genres.get(video_data.get('genre'), -1)
    except Exception:
        video_data['genre_id'] = -1
    video_data.pop('genre', None)

    # Age in days (int)
    try:
        video_data['age'] = int(age_days) if not np.isnan(age_days) else 0
    except Exception:
        video_data['age'] = 0

    video_data.pop('upload_date', None)

    # Log transforms (use safe numeric conversion)
    video_data['log_likes'] = float(np.log1p(to_float_safe(video_data.get('likes', 0))))
    video_data['log_view_count'] = float(np.log1p(to_float_safe(video_data.get('view_count', 0))))

    # Avoid division by zero
    video_data['view_like_ratio'] = (
        video_data['view_count'] / (video_data['likes'] + 1) if (video_data['likes'] is not None) else 0.0
    )
    video_data['log_view_like_ratio'] = float(np.log1p(to_float_safe(video_data.get('view_like_ratio', 0.0))))
    video_data['log_comment_count'] = float(np.log1p(to_float_safe(video_data.get('comment_count', 0))))

    return video_data


def extract_video_id(url: str) -> str | None:
    """
    Extracts the YouTube video ID from a given URL.
    Returns the video ID as a string, or None if not found.
    """
    # Regular expression to match YouTube video IDs
    pattern = r'(?:v=|\/)([0-9A-Za-z_-]{11}).*'
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    return None

if __name__ == "__main__":
    # Replace with your own API key
    API_KEY = "AIzaSyAKc2tJ1cVtd5wluOxqCkpYsTRVsjyG44w"
    url =  input("Enter YouTube video URL: ").strip() #"https://www.youtube.com/watch?v=eByxkesA6lY"
    VIDEO_ID = extract_video_id(url)
    print(VIDEO_ID)
    video_data = fetch_metadata(VIDEO_ID, API_KEY)
    if video_data is None:
        print("Failed to retrieve video metadata.")
        sys.exit(1)

    #video_data = {'likes': 25422, 'view_count': 1614553, 'duration': '397', 'upload_date': '2021-02-27T19:57:04-08:00', 'comment_count': 2085}
    add_features(video_data)
    most_neg, most_pos = add_sentiment_scores(video_data, VIDEO_ID)

    
    predict_dislikes({k: [v] for k, v in video_data.items()})
    
    for key, val in video_data.items():
        print(f"{key}: {val}")

    input("Enter feedback ('o' - overpredict, 'u' - underpredict, 'a' - accurate): ")
    print("Feedback recorded. Thank you!")
