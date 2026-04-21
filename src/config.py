from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "PS_20174392719_1491204439457_log.csv"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
MLRUNS_DIR = ROOT_DIR / "mlruns"
MODEL_PATH = MODELS_DIR / "fraud_model.joblib"
FEATURE_COLUMNS_PATH = MODELS_DIR / "feature_columns.joblib"
REFERENCE_STATS_PATH = MODELS_DIR / "reference_stats.joblib"
