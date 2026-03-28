# EDA and Feature Engineering Report

Input file: `D:\Coding\Machine learning\YT dislikes\feature_engineering_v2\yt_dataset_v4_merged.csv`

Rows: 29156, Columns: 21

Detected target column: `dislikes`

## Missing values (columns with >0% missing)

| column | percent_missing |
|---|---:|
| `log_comment_count` | 0.003 |
| `log_likes` | 0.000 |

## Top features by absolute Pearson correlation with target

| feature | abs_corr_with_target |
|---|---:|
| `view_count` | 0.6878 |
| `likes` | 0.6641 |
| `comment_count` | 0.4237 |
| `log_dislikes` | 0.3559 |
| `log_view_count` | 0.3064 |
| `log_likes` | 0.2630 |
| `log_comment_count` | 0.2130 |
| `avg_compound` | 0.0494 |
| `genre_id` | 0.0337 |
| `avg_pos` | 0.0318 |
| `comment_sample_size` | 0.0229 |
| `no_comments` | 0.0208 |
| `avg_neu` | 0.0116 |
| `age` | 0.0101 |
| `duration` | 0.0100 |
| `avg_neg` | 0.0093 |
| `view_like_ratio` | 0.0018 |

Correlation heatmap: `analysis/plots/correlation_heatmap.png`

PCA explained variance plot: `analysis/plots/pca_explained_variance.png`

PCA scatter (PC1 vs PC2): `analysis/plots/pca_scatter_pc1_pc2.png`

## PCA summary

- 75% variance reached with 6 components
- 90% variance reached with 9 components
- 95% variance reached with 10 components

## Feature distributions

Individual histograms and log-transformed histograms saved to `analysis/plots/` (files prefixed with `hist_`).

## Engineered dataset

Wrote engineered dataset to `D:\Coding\Machine learning\YT dislikes\analysis\yt_dataset_feature_engineered.csv` containing added column `log_dislikes` and conditional ratio features when input columns were present.

## Recommendations

- Use `log_dislikes` as the regression target for models that expect gaussian-like residuals.
- Start modeling with the top correlated features listed above.
- Consider dimensionality reduction (PCA) or feature selection if you have many numeric features or multicollinearity.

