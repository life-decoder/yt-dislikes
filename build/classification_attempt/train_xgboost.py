"""Train an XGBoost multiclass classifier to predict likes-percentage sentiment bins.

Usage:
    python train_xgboost.py --data "../yt_dataset_v4.csv" --output-dir "./artifacts"

This script is intentionally defensive: it detects common column names for like/dislike counts
and selects numeric features automatically. It saves a scikit-learn Pipeline containing
preprocessing and the trained XGBoost model to `output_dir/model.joblib` and stores
metrics/plots there as well.
"""
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


def detect_likes_percentage(df: pd.DataFrame) -> pd.Series:
    # Try direct column names first
    for col in ["likes_percentage", "likes_pct", "like_percentage"]:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce")

    # Try raw counts
    like_cols = [c for c in df.columns if c.lower() in ("likes", "likecount", "like_count", "like")]
    dislike_cols = [c for c in df.columns if c.lower() in ("dislikes", "dislikecount", "dislike_count", "dislike")]

    if like_cols and dislike_cols:
        like = pd.to_numeric(df[like_cols[0]], errors="coerce")
        dislike = pd.to_numeric(df[dislike_cols[0]], errors="coerce")
        denom = like + dislike
        with np.errstate(divide='ignore', invalid='ignore'):
            pct = 100 * (like / denom)
        return pct

    # As a fallback, try columns named likeCount / dislikeCount (YouTube API style)
    for l in ("likeCount", "likesCount", "likes"):
        for d in ("dislikeCount", "dislikesCount", "dislikes"):
            if l in df.columns and d in df.columns:
                like = pd.to_numeric(df[l], errors="coerce")
                dislike = pd.to_numeric(df[d], errors="coerce")
                denom = like + dislike
                with np.errstate(divide='ignore', invalid='ignore'):
                    pct = 100 * (like / denom)
                return pct

    raise ValueError("Could not detect likes/dislikes columns to compute likes percentage.")


def make_target_series(likes_pct: pd.Series) -> pd.Series:
    # Bins: 0-20,20-40,40-60,60-80,80-100 mapped to 0..4
    bins = [0, 20, 40, 60, 80, 100.0001]
    labels = [0, 1, 2, 3, 4]
    # Clip to [0,100]
    pct = likes_pct.clip(lower=0, upper=100)
    return pd.cut(pct, bins=bins, labels=labels, include_lowest=True).astype(float).astype('Int64')


def select_features(df: pd.DataFrame, target_col: str):
    # Choose numeric columns (except the target) and a small set of safe categorical columns
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    # Avoid leaking the derived likes percentage into the features
    for t in ("likes_pct", "likes_percentage", "likesPct", "like_percentage"):
        if t in numeric_cols:
            numeric_cols.remove(t)

    # Allow category-like columns if present
    cat_candidates = [c for c in ["category_id", "channelId", "channel_id"] if c in df.columns]
    return numeric_cols, cat_candidates


def build_pipeline(numeric_cols, categorical_cols):
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # OneHotEncoder changed the `sparse` arg name to `sparse_output` in newer scikit-learn.
    # Use a try/except to remain compatible across versions.
    categorical_transformer = None
    if categorical_cols:
        try:
            categorical_transformer = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse=False))
            ])
        except TypeError:
            # Older/newer scikit-learn versions may use sparse_output instead
            categorical_transformer = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ])

    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_transformer, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_transformer, categorical_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    xgb = XGBClassifier(use_label_encoder=False, objective="multi:softprob", eval_metric="mlogloss", verbosity=1)

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("xgb", xgb)])
    return pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="../yt_dataset_v4.csv", help="Path to CSV dataset")
    parser.add_argument("--output-dir", type=str, default="./artifacts", help="Directory to save model and outputs")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    data_path = Path(args.data)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from {data_path}")
    df = pd.read_csv(data_path)

    # Detect or compute likes percentage
    likes_pct = detect_likes_percentage(df)
    df = df.copy()
    df["likes_pct"] = likes_pct

    # Drop rows with missing target
    df = df[df["likes_pct"].notna()]
    if df.empty:
        raise RuntimeError("No rows with computable likes percentage found.")

    df["target"] = make_target_series(df["likes_pct"]).astype(int)

    # Feature selection
    numeric_cols, categorical_cols = select_features(df, target_col="target")
    print("Numeric features detected:", numeric_cols)
    print("Categorical features detected:", categorical_cols)

    X = df[numeric_cols + categorical_cols]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, stratify=y, random_state=args.random_state)

    pipeline = build_pipeline(numeric_cols, categorical_cols)

    # Quick grid for a few sensible params to keep run-time reasonable
    param_grid = {
        "xgb__n_estimators": [100, 300],
        "xgb__max_depth": [3, 6],
        "xgb__learning_rate": [0.1, 0.01]
    }

    grid = GridSearchCV(pipeline, param_grid, cv=3, scoring="accuracy", n_jobs=1, verbose=2)
    print("Starting GridSearchCV...")
    grid.fit(X_train, y_train)

    print("Best params:", grid.best_params_)
    best = grid.best_estimator_

    # Evaluate on test set
    y_pred = best.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=["Very negative","Negative","Mixed","Positive","Very positive"], output_dict=True)

    # Save artifacts
    joblib.dump(best, out_dir / "model.joblib")
    pd.DataFrame(report).transpose().to_csv(out_dir / "classification_report.csv")

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["VN","N","M","P","VP"], yticklabels=["VN","N","M","P","VP"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion matrix")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png")

    # Feature importance (for XGBoost part). Try to extract if possible
    try:
        # Access feature names after preprocessing
        preproc = best.named_steps["preprocessor"]
        # Build feature name list
        feature_names = []
        if "num" in dict(preproc.transformers_):
            feature_names.extend(preproc.named_transformers_["num"].named_steps.get("imputer", SimpleImputer()).feature_names_in_ if hasattr(preproc.named_transformers_["num"].named_steps.get("imputer", None), 'feature_names_in_') else numeric_cols)
        if "cat" in dict(preproc.transformers_):
            # OneHotEncoder get_feature_names_out
            ohe = preproc.named_transformers_["cat"].named_steps["ohe"]
            cat_names = ohe.get_feature_names_out(categorical_cols).tolist()
            feature_names.extend(cat_names)

        booster = best.named_steps["xgb"].get_booster()
        fmap = list(zip(feature_names, booster.get_score(importance_type='gain').values() if booster.get_score() else []))
    except Exception:
        fmap = None

    # Save simple notes
    with open(out_dir / "notes.txt", "w", encoding="utf-8") as f:
        f.write(f"Best params: {grid.best_params_}\n")
        f.write(f"Train size: {len(X_train)}, Test size: {len(X_test)}\n")

    print("Training complete. Artifacts written to", out_dir)


if __name__ == "__main__":
    main()
