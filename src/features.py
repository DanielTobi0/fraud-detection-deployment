from __future__ import annotations

from typing import Iterable

import pandas as pd

BASE_NUMERIC_COLUMNS = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]


REQUIRED_COLUMNS = [
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
]


def _validate_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def build_training_features(df: pd.DataFrame) -> pd.DataFrame:
    _validate_columns(df, REQUIRED_COLUMNS)

    work = df.copy()
    work = work.sort_values(["nameOrig", "step"]).reset_index(drop=True)

    work["balance_delta_orig"] = work["oldbalanceOrg"] - work["newbalanceOrig"]
    work["balance_delta_dest"] = work["newbalanceDest"] - work["oldbalanceDest"]
    work["amount_to_oldbalance_ratio"] = work["amount"] / (work["oldbalanceOrg"] + 1.0)

    work["orig_txn_count_before"] = work.groupby("nameOrig").cumcount()
    work["orig_amount_rolling3"] = (
        work.groupby("nameOrig")["amount"]
        .transform(lambda x: x.shift(1).rolling(window=3, min_periods=1).mean())
        .fillna(0.0)
    )
    work["orig_step_gap"] = (
        work.groupby("nameOrig")["step"].diff().fillna(0.0).astype(float)
    )

    work["dest_txn_count_before"] = work.groupby("nameDest").cumcount()

    type_dummies = pd.get_dummies(work["type"], prefix="type", dtype=int)
    work = pd.concat([work, type_dummies], axis=1)

    feature_columns = BASE_NUMERIC_COLUMNS + [
        "balance_delta_orig",
        "balance_delta_dest",
        "amount_to_oldbalance_ratio",
        "orig_txn_count_before",
        "orig_amount_rolling3",
        "orig_step_gap",
        "dest_txn_count_before",
    ] + sorted(type_dummies.columns.tolist())

    features = work[feature_columns].fillna(0.0)
    features["isFraud"] = work["isFraud"].astype(int)
    features["step"] = work["step"].astype(int)
    return features


def align_feature_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    missing = [col for col in columns if col not in df.columns]
    for col in missing:
        df[col] = 0.0
    return df[columns]
