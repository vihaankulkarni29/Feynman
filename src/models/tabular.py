from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

import xgboost as xgb

from config.pipeline import load_config


def prepare_tabular_dataset(
    df: pd.DataFrame,
    target: str = "total_points",
    exclude_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    exclude_cols = exclude_cols or ["element_id", "name", "team", "position", target]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    X = df[feature_cols].select_dtypes(include="number").copy()
    y = df[target].astype(float)
    return X, y


def train_tabular_model(
    df: pd.DataFrame,
    target: str = "total_points",
) -> Tuple[xgb.XGBRegressor, Dict[str, float]]:
    config = load_config()["models"]["tabular"]
    X, y = prepare_tabular_dataset(df, target=target)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.get("test_fraction", 0.15), random_state=config.get("random_state", 42)
    )
    model = xgb.XGBRegressor(
        objective=config.get("objective", "reg:squarederror"),
        n_estimators=config.get("n_estimators", 500),
        learning_rate=config.get("learning_rate", 0.05),
        max_depth=config.get("max_depth", 6),
        subsample=config.get("subsample", 0.8),
        colsample_bytree=config.get("colsample_bytree", 0.8),
        random_state=config.get("random_state", 42),
        n_jobs=4,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        early_stopping_rounds=config.get("early_stopping_rounds", 50),
        verbose=False,
    )
    preds = model.predict(X_test)
    metrics = {
        "mae": float(mean_absolute_error(y_test, preds)),
        "r2": float(r2_score(y_test, preds)),
    }
    logger.info("Tabular model trained. Metrics: {}", metrics)
    return model, metrics


def predict_tabular(model: xgb.XGBRegressor, df: pd.DataFrame) -> pd.Series:
    X, _ = prepare_tabular_dataset(df, exclude_cols=["element_id", "name", "team", "position", "total_points"])
    preds = model.predict(X)
    return pd.Series(preds, index=df.index, name="xPts_tabular")
