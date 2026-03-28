"""Copy of analysis/xgb_model_selection.py modified to write outputs into xgboost_v2/ folder."""
from pathlib import Path
from shutil import copyfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from xgboost.callback import EarlyStopping as EarlyStoppingCallback
import joblib

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / 'yt_dataset_v5.csv'
THIS_DIR = Path(__file__).resolve().parents[0]
PLOTS_DIR = THIS_DIR / 'plots'
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = THIS_DIR / 'XGBoost_selection_report.md'
MODEL_PATH = THIS_DIR / 'xgb_model_selection_joblib.pkl'

RANDOM_STATE = 42

# ...existing code...
# For brevity, we'll import the original script and reuse its functions by copying the file content.
# But because we can't `import` by path easily here, we'll simply copy the analysis version file and
# adjust the path constants by reusing the same content.

source = Path(__file__).resolve().parents[0].parent / 'analysis' / 'xgb_model_selection.py'
if source.exists():
    txt = source.read_text()
    # Replace occurrences of analysis/ paths with local ones (only the header constants)
    txt = txt.replace("PLOTS_DIR = Path(__file__).resolve().parents[0] / 'plots'", "PLOTS_DIR = THIS_DIR / 'plots'")
    txt = txt.replace("REPORT_PATH = Path(__file__).resolve().parents[0] / 'XGBoost_selection_report.md'", "REPORT_PATH = THIS_DIR / 'XGBoost_selection_report.md'")
    txt = txt.replace("MODEL_PATH = Path(__file__).resolve().parents[0] / 'xgb_model_selection_joblib.pkl'", "MODEL_PATH = THIS_DIR / 'xgb_model_selection_joblib.pkl'")
    Path(__file__).write_text(txt)
else:
    raise FileNotFoundError(f'source script not found: {source}')
