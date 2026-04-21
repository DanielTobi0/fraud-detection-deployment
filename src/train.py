from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from src.config import (
    FEATURE_COLUMNS_PATH,
    MLRUNS_DIR,
    MODEL_PATH,
    PROCESSED_DIR,
    RAW_DATA_PATH,
    REFERENCE_STATS_PATH,
)
from src.features import build_training_features


def _prepare_dirs() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)


def _temporal_split(
    df: pd.DataFrame, validation_ratio: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = int(df["step"].quantile(1.0 - validation_ratio))
    train = df[df["step"] <= cutoff].copy()
    valid = df[df["step"] > cutoff].copy()
    return train, valid


def _train_model(x_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
    positives = max(int(y_train.sum()), 1)
    negatives = max(int((1 - y_train).sum()), 1)
    scale_pos_weight = negatives / positives

    model = XGBClassifier(
        n_estimators=180,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(x_train, y_train)
    return model


def run_training(data_path: Path = RAW_DATA_PATH, max_rows: int | None = None) -> dict[str, float]:
    _prepare_dirs()

    raw_df = pd.read_csv(data_path, nrows=max_rows)
    featured = build_training_features(raw_df)

    train_df, valid_df = _temporal_split(featured)

    y_train = train_df["isFraud"]
    y_valid = valid_df["isFraud"]

    x_train = train_df.drop(columns=["isFraud", "step"])
    x_valid = valid_df.drop(columns=["isFraud", "step"])

    model = _train_model(x_train, y_train)

    valid_probs = model.predict_proba(x_valid)[:, 1]
    valid_preds = (valid_probs >= 0.5).astype(int)

    roc_auc = 0.5
    if y_valid.nunique() > 1:
        roc_auc = float(roc_auc_score(y_valid, valid_probs))

    pr_auc = 0.0
    if int(np.sum(y_valid)) > 0:
        pr_auc = float(average_precision_score(y_valid, valid_probs))

    metrics = {
        "precision": float(precision_score(y_valid, valid_preds, zero_division=0)),
        "recall": float(recall_score(y_valid, valid_preds, zero_division=0)),
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
    }

    mlflow.set_tracking_uri(f"file://{MLRUNS_DIR}")
    mlflow.set_experiment("fraud-detection-mvp")

    with mlflow.start_run(run_name="xgboost-sequence-v1"):
        mlflow.log_params(
            {
                "rows": len(featured),
                "features": x_train.shape[1],
                "validation_rows": len(valid_df),
            }
        )
        mlflow.log_metrics(metrics)

        train_dump_path = PROCESSED_DIR / "train_features.parquet"
        valid_dump_path = PROCESSED_DIR / "valid_features.parquet"
        train_df.to_parquet(train_dump_path, index=False)
        valid_df.to_parquet(valid_dump_path, index=False)
        mlflow.log_artifact(str(train_dump_path))
        mlflow.log_artifact(str(valid_dump_path))

    joblib.dump(model, MODEL_PATH)
    joblib.dump(x_train.columns.tolist(), FEATURE_COLUMNS_PATH)
    reference_stats = {
        "pred_mean": float(valid_probs.mean()),
        "pred_std": float(valid_probs.std() + 1e-6),
    }
    joblib.dump(reference_stats, REFERENCE_STATS_PATH)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train fraud detection model")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=RAW_DATA_PATH,
        help="Path to PaySim CSV",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional row limit for quick iteration",
    )
    args = parser.parse_args()

    metrics = run_training(args.data_path, args.max_rows)
    print("Training complete")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
