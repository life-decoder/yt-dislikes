"""
Sentiment analysis for a comments dataset using CardiffNLP/twitter-roberta-base-sentiment-latest.

Reads a CSV (optionally in chunks), finds the comment/text column, runs predictions in batches,
and writes out a new CSV that preserves all original columns except the original comment text.

Usage (example):
  python roberta/sentiment_comments_roberta.py \
    --input roberta/combined_comments_sentiment.csv \
    --output roberta/combined_comments_sentiment_with_roberta.csv \
    --batch-size 64 --chunk-size 1000

Requirements:
  pip install transformers torch pandas scipy tqdm

This file is written but not executed by the assistant.
"""
from __future__ import annotations

import argparse
import math
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
from scipy.special import softmax
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig


def preprocess(text: str) -> str:
    """Preprocess typical twitter-style text: replace @user and http links with placeholders.

    Keep this minimal so the model still sees tokens relevant to sentiment.
    """
    if not isinstance(text, str):
        return ""
    new_text = []
    for t in text.split():
        if t.startswith('@') and len(t) > 1:
            t = '@user'
        elif t.startswith('http'):
            t = 'http'
        new_text.append(t)
    return " ".join(new_text)


def find_comment_column(columns: List[str]) -> Optional[str]:
    candidates = ['comment', 'text', 'comment_text', 'content', 'message', 'body']
    cols_lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in cols_lower:
            return cols_lower[cand]
    # fallback: return first column that contains 'comment' or 'text'
    for k, v in cols_lower.items():
        if 'comment' in k or 'text' in k or 'body' in k:
            return v
    return None


def batch_predict(texts: List[str], tokenizer, model, device: torch.device, batch_size: int = 32) -> np.ndarray:
    """Return NxC probabilities for a list of texts using the tokenizer and model on device.

    This function processes inputs in internal batches (batch_size) to avoid OOM.
    """
    all_probs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            # tokenizer will pad and truncate as needed
            encoded = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt')
            encoded = {k: v.to(device) for k, v in encoded.items()}
            outputs = model(**encoded)
            logits = outputs.logits.cpu().numpy()
            probs = softmax(logits, axis=1)
            all_probs.append(probs)
    if all_probs:
        return np.vstack(all_probs)
    return np.zeros((0, model.config.num_labels))


def process_file(
    input_path: str,
    output_path: str,
    model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
    chunk_size: int = 2000,
    predict_batch_size: int = 64,
    device: Optional[str] = None,
):
    device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))

    print(f"Loading model {model_name} on device {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)

    # labels mapping: model.config.id2label maps ints to label names
    id2label = {int(k): v for k, v in config.id2label.items()} if hasattr(config, 'id2label') else None
    if id2label is None:
        # Fallback: use 0..n-1
        id2label = {i: str(i) for i in range(model.config.num_labels)}
    labels = [id2label[i] for i in sorted(id2label.keys())]

    first_chunk = True
    total_rows = 0
    reader = pd.read_csv(input_path, chunksize=chunk_size, dtype=str, keep_default_na=False)
    for chunk in tqdm(reader, desc="chunks"):
        total_rows += len(chunk)
        # detect comment column on first chunk
        if first_chunk:
            comment_col = find_comment_column(list(chunk.columns))
            if comment_col is None:
                raise ValueError("Could not find a comment/text column in the input CSV. Please pass a file with a column named 'comment' or 'text' or use --comment-col to set it explicitly.")
            print(f"Using comment column: {comment_col}")

        texts_raw = chunk[comment_col].fillna("").astype(str).tolist()
        # preprocess texts and mark empty rows
        texts = [preprocess(t) for t in texts_raw]

        # Build a mask for empty/blank texts
        empty_mask = [not bool(t.strip()) for t in texts]

        # Predict probabilities for non-empty texts in a single call
        if any(not m for m in empty_mask):
            probs = batch_predict([t for t, m in zip(texts, empty_mask) if not m], tokenizer, model, device, batch_size=predict_batch_size)
        else:
            probs = np.zeros((0, model.config.num_labels))

        # Reassemble per-row probabilities aligned with chunk rows
        probs_per_row = []
        it = iter(probs)
        for is_empty in empty_mask:
            if is_empty:
                probs_per_row.append([math.nan] * model.config.num_labels)
            else:
                probs_per_row.append(next(it).tolist())

        probs_arr = np.array(probs_per_row)

        # Add probability columns for each label
        for idx, label in enumerate(labels):
            col_name = f"sentiment_{label.lower()}"
            chunk[col_name] = probs_arr[:, idx]

        # predicted label and score
        pred_idxs = np.nanargmax(probs_arr, axis=1)
        pred_labels = [labels[i] if not np.all(np.isnan(row)) else "empty" for i, row in zip(pred_idxs, probs_arr)]
        pred_scores = [None if np.all(np.isnan(row)) else float(np.nanmax(row)) for row in probs_arr]
        chunk['sentiment_pred'] = pred_labels
        chunk['sentiment_pred_score'] = pred_scores

        # Drop the original comment text before saving
        chunk_to_save = chunk.drop(columns=[comment_col])

        # Write header on first chunk, append afterwards
        if first_chunk:
            chunk_to_save.to_csv(output_path, index=False, mode='w')
            first_chunk = False
        else:
            chunk_to_save.to_csv(output_path, index=False, mode='a', header=False)

    print(f"Finished processing. Total rows processed: {total_rows}. Output written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Run RoBERTa sentiment analysis on a comments CSV and save results without original text')
    parser.add_argument('--input', '-i', required=True, help='Path to input CSV with comments')
    parser.add_argument('--output', '-o', required=True, help='Path to output CSV (will be overwritten)')
    parser.add_argument('--model', '-m', default='cardiffnlp/twitter-roberta-base-sentiment-latest', help='HuggingFace model id')
    parser.add_argument('--chunk-size', type=int, default=2000, help='Pandas CSV chunk size for streaming')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for model prediction')
    parser.add_argument('--device', type=str, default=None, help='Torch device (e.g., cpu or cuda); default auto-detect')
    args = parser.parse_args()

    process_file(
        input_path=args.input,
        output_path=args.output,
        model_name=args.model,
        chunk_size=args.chunk_size,
        predict_batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == '__main__':
    main()
