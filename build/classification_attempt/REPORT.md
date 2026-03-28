## XGBoost v2 — Predicting Likes Percentage Sentiment Bins

This report documents the scripts and process used to train and evaluate an XGBoost multiclass classifier that predicts sentiment bins based on a video's likes percentage. The five bins are:

- 0-20%: Very negative (0)
- 20-40%: Negative (1)
- 40-60%: Mixed (2)
- 60-80%: Positive (3)
- 80-100%: Very positive (4)

Files added in `xgboost_v2/`:

- `train_xgboost.py` — training script. Detects or computes likes percentage, creates features, runs GridSearchCV on an XGBoost classifier, and saves model/artifacts to `--output-dir` (default `./artifacts`).
- `test_xgboost.py` — loads the trained pipeline and evaluates it on a dataset, producing a text classification report and confusion matrix image.
- `REPORT.md` — this document.

How the scripts work (high level):

1. Data loading: The scripts read a CSV (default `../yt_dataset_v4.csv`) and attempt to find either a likes-percentage column or like/dislike counts to compute the percentage.
2. Target creation: likes percentage is clipped to [0,100] and binned into the five classes listed above.
3. Feature selection: Numeric columns are automatically selected as features; a few categorical columns (`category_id`, `channelId`, `channel_id`) are optionally included if present.
4. Preprocessing: Numeric features are median-imputed and standardized; categorical features are imputed and one-hot encoded.
5. Model: An XGBoost classifier (`multi:softprob`) is wrapped in a scikit-learn `Pipeline` and tuned with a small grid (`n_estimators`, `max_depth`, `learning_rate`) for practicality.
6. Outputs: Trained pipeline saved as `model.joblib`, classification report CSV, confusion matrix PNG, and simple notes.

Usage examples

Train:

```
python xgboost_v2/train_xgboost.py --data "../yt_dataset_v4.csv" --output-dir "xgboost_v2/artifacts"
```

Test / evaluate:

```
python xgboost_v2/test_xgboost.py --model xgboost_v2/artifacts/model.joblib --data ../yt_dataset_v4.csv --output xgboost_v2/artifacts
```

Important notes & assumptions

- The dataset `yt_dataset_v4.csv` provided in the workspace is used by default; the scripts attempt to detect columns for likes and dislikes. If your dataset uses different column names, pass a precomputed `likes_pct` column named `likes_percentage` or `likes_pct` or update the detection logic in the scripts.
- The feature selection is intentionally simple: it picks numeric columns and a few known categorical fields. For best performance you should curate features in `feature_engineering/` (there are scripts and recommended feature lists in the repo).
- Grid search is small to keep runtime reasonable on a developer machine. Expand `param_grid` in `train_xgboost.py` for a more thorough search.
- The model pipeline is saved with preprocessing, so `test_xgboost.py` expects the same input columns that the pipeline was trained with. Use the same dataset columns when evaluating.

Next steps and improvements

- Expand feature engineering: text features from titles/descriptions, comment sentiment aggregates, upload time features, and channel-level features.
- Add class balancing (SMOTE/weights) if classes are imbalanced.
- Use cross-validation with more folds and a larger grid for robust hyperparameter tuning.
- Add unit tests and a small example dataset for CI-based smoke tests.

Requirements

See `requirements.txt` at repository root. The file has been updated to include `scikit-learn`, `xgboost`, `joblib`, `matplotlib`, and `seaborn`.

Contact

If you want changes (different bins, regression instead of classification, or other metrics), tell me which direction to take and I'll update the scripts and report.
