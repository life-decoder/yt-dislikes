"""Load trained model and run evaluation on a holdout CSV (or the same dataset) to produce metrics.

Usage:
    python test_xgboost.py --model ./artifacts/model.joblib --data ../yt_dataset_v4.csv --output ./artifacts
"""
from pathlib import Path
import argparse
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def detect_likes_percentage(df: pd.DataFrame) -> pd.Series:
    for col in ["likes_percentage", "likes_pct", "like_percentage"]:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce")
    like_cols = [c for c in df.columns if c.lower() in ("likes", "likecount", "like_count", "like")]
    dislike_cols = [c for c in df.columns if c.lower() in ("dislikes", "dislikecount", "dislike_count", "dislike")]
    if like_cols and dislike_cols:
        like = pd.to_numeric(df[like_cols[0]], errors="coerce")
        dislike = pd.to_numeric(df[dislike_cols[0]], errors="coerce")
        denom = like + dislike
        with np.errstate(divide='ignore', invalid='ignore'):
            pct = 100 * (like / denom)
        return pct
    raise ValueError("Could not detect likes/dislikes columns to compute likes percentage.")


def make_target_series(likes_pct: pd.Series) -> pd.Series:
    bins = [0, 20, 40, 60, 80, 100.0001]
    labels = [0, 1, 2, 3, 4]
    pct = likes_pct.clip(lower=0, upper=100)
    return pd.cut(pct, bins=bins, labels=labels, include_lowest=True).astype(float).astype('Int64')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--output", type=str, default="./artifacts")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data)
    likes_pct = detect_likes_percentage(df)
    df = df.copy()
    df["likes_pct"] = likes_pct
    df = df[df["likes_pct"].notna()]
    df["target"] = make_target_series(df["likes_pct"]).astype(int)

    model = joblib.load(args.model)

    # Build feature matrix from model's expected features
    preproc = model.named_steps["preprocessor"]
    # Collect input column names used by preprocessor
    input_cols = []
    for name, trans, cols in preproc.transformers_:
        input_cols.extend(cols)

    X = df[input_cols]
    y = df["target"]

    y_pred = model.predict(X)
    report_text = classification_report(y, y_pred, target_names=["Very negative","Negative","Mixed","Positive","Very positive"], output_dict=False)
    # classification_report returns a string when output_dict=False
    print(report_text)
    with open(out_dir / "test_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(str(report_text))

    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["VN","N","M","P","VP"], yticklabels=["VN","N","M","P","VP"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion matrix")
    plt.tight_layout()
    plt.savefig(out_dir / "test_confusion_matrix.png")


if __name__ == "__main__":
    main()
