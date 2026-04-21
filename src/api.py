from __future__ import annotations

from time import time

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.config import FEATURE_COLUMNS_PATH, MODEL_PATH, REFERENCE_STATS_PATH
from src.drift import DriftMonitor
from src.features import align_feature_columns

app = FastAPI(title="Fraud Detection API", version="0.1.0")


class PredictRequest(BaseModel):
    step: int
    type: str
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float
    orig_txn_count_before: float = 0.0
    orig_amount_rolling3: float = 0.0
    orig_step_gap: float = 0.0
    dest_txn_count_before: float = 0.0


class PredictResponse(BaseModel):
    prediction: int
    fraud_probability: float


model = None
feature_columns: list[str] = []
monitor: DriftMonitor | None = None
start_time = time()


def _build_single_row_features(payload: PredictRequest) -> pd.DataFrame:
    row = {
        "amount": payload.amount,
        "oldbalanceOrg": payload.oldbalanceOrg,
        "newbalanceOrig": payload.newbalanceOrig,
        "oldbalanceDest": payload.oldbalanceDest,
        "newbalanceDest": payload.newbalanceDest,
        "balance_delta_orig": payload.oldbalanceOrg - payload.newbalanceOrig,
        "balance_delta_dest": payload.newbalanceDest - payload.oldbalanceDest,
        "amount_to_oldbalance_ratio": payload.amount / (payload.oldbalanceOrg + 1.0),
        "orig_txn_count_before": payload.orig_txn_count_before,
        "orig_amount_rolling3": payload.orig_amount_rolling3,
        "orig_step_gap": payload.orig_step_gap,
        "dest_txn_count_before": payload.dest_txn_count_before,
    }
    row[f"type_{payload.type}"] = 1.0
    return pd.DataFrame([row])


@app.on_event("startup")
def _startup() -> None:
    global model, feature_columns, monitor

    if (
        not MODEL_PATH.exists()
        or not FEATURE_COLUMNS_PATH.exists()
        or not REFERENCE_STATS_PATH.exists()
    ):
        raise RuntimeError("Model artifacts not found. Run training first.")

    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
    reference_stats = joblib.load(REFERENCE_STATS_PATH)
    monitor = DriftMonitor(
        reference_mean=reference_stats["pred_mean"],
        reference_std=reference_stats["pred_std"],
        window_size=200,
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    if model is None or monitor is None:
        raise HTTPException(status_code=500, detail="Model is not initialized")

    input_frame = _build_single_row_features(payload)
    model_frame = align_feature_columns(input_frame, feature_columns)

    probability = float(model.predict_proba(model_frame)[:, 1][0])
    prediction = int(probability >= 0.5)

    monitor.add_score(probability)

    return PredictResponse(prediction=prediction, fraud_probability=probability)


@app.get("/metrics")
def metrics() -> dict[str, float | int | str | None]:
    if monitor is None:
        raise HTTPException(status_code=500, detail="Drift monitor not initialized")

    drift = monitor.evaluate()
    return {
        "predictions_tracked": monitor.total_observations,
        "drift_status": drift.status,
        "z_score": drift.z_score,
        "recent_mean": drift.recent_mean,
        "uptime_seconds": int(time() - start_time),
    }
