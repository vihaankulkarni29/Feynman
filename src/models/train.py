from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from loguru import logger
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score

from config.pipeline import load_config
from src.models.sequence import FPLLSTM, train_sequence_model, predict_sequence
from src.models.tabular import train_tabular_model, predict_tabular, export_feature_importance


def _chronological_split(
    df: pd.DataFrame,
    time_col: str = "gameweek",
    season_col: Optional[str] = None,
    validate_first_n_gws: int = 19,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    has_multiple_seasons = False
    if season_col and season_col in df.columns:
        seasons = sorted(df[season_col].dropna().unique())
        has_multiple_seasons = len(seasons) >= 2

    if has_multiple_seasons:
        train_seasons = seasons[:-1]
        val_test_seasons = seasons[-1:]
        train_df = df[df[season_col].isin(train_seasons)].copy()
        val_test_df = df[df[season_col].isin(val_test_seasons)].copy()
    else:
        train_df = df.copy()
        val_test_df = df.copy()

    if time_col in val_test_df.columns:
        val_test_df = val_test_df.sort_values(time_col).reset_index(drop=True)
        max_gw = val_test_df[time_col].max()

        if has_multiple_seasons:
            if validate_first_n_gws and validate_first_n_gws > 0 and max_gw > validate_first_n_gws + 5:
                val_df = val_test_df[val_test_df[time_col] <= validate_first_n_gws].copy()
                test_df = val_test_df[val_test_df[time_col] > validate_first_n_gws].copy()
            else:
                total = len(val_test_df)
                split_val = int(total * 0.6)
                split_test = int(total * 0.8)
                val_df = val_test_df.iloc[:split_val].copy()
                test_df = val_test_df.iloc[split_val:split_test].copy()
        else:
            if validate_first_n_gws and validate_first_n_gws > 0 and max_gw > validate_first_n_gws + 5:
                train_df = val_test_df[val_test_df[time_col] <= validate_first_n_gws].copy()
                val_df = val_test_df[(val_test_df[time_col] > validate_first_n_gws) & (val_test_df[time_col] <= validate_first_n_gws * 2)].copy()
                test_df = val_test_df[val_test_df[time_col] > validate_first_n_gws * 2].copy()
            else:
                total = len(val_test_df)
                split_train = int(total * 0.6)
                split_val = int(total * 0.8)
                train_df = val_test_df.iloc[:split_train].copy()
                val_df = val_test_df.iloc[split_train:split_val].copy()
                test_df = val_test_df.iloc[split_val:].copy()
    else:
        val_df = val_test_df.copy()
        test_df = val_test_df.copy()

    logger.info(
        "Chronological split: train={}, val={}, test={}",
        len(train_df),
        len(val_df),
        len(test_df),
    )
    return train_df, val_df, test_df


def _prepare_sequences(
    df: pd.DataFrame,
    target_col: str = "target_xPts",
    group_by: str = "element_id",
    time_col: str = "gameweek",
    seq_len: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    import torch
    from src.models.sequence import _build_sequences

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in {target_col, group_by, time_col}]
    X, y = _build_sequences(df, feature_cols, target_col, seq_len, group_by, time_col)
    return X, y


def train_ensemble(
    df: pd.DataFrame,
    models_dir: Path,
    target_col: str = "target_xPts",
    time_col: str = "gameweek",
    season_col: Optional[str] = None,
    group_by: str = "element_id",
) -> Dict[str, any]:
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    config = load_config()
    seq_config = config["models"]["sequence"]
    tab_config = config["models"]["tabular"]
    ensemble_config = config["models"]["ensemble"]
    chrono_config = config.get("models", {}).get("ensemble", {}).get("chronological", {})
    validate_first_n_gws = chrono_config.get("validate_first_n_gws", 19)

    train_df, val_df, test_df = _chronological_split(
        df, time_col=time_col, season_col=season_col, validate_first_n_gws=validate_first_n_gws
    )

    logger.info("Training tabular model...")
    tabular_model, tab_metrics = train_tabular_model(train_df, target=target_col)
    tab_val_preds = predict_tabular(tabular_model, val_df)
    tab_test_preds = predict_tabular(tabular_model, test_df)

    feature_names = [c for c in train_df.select_dtypes(include="number").columns if c != target_col]
    export_feature_importance(tabular_model, feature_names, models_dir / "feature_importances.json")

    logger.info("Training sequence model...")
    seq_model, seq_metrics = train_sequence_model(
        train_df,
        target_col=target_col,
        time_col=time_col,
        group_by=group_by,
    )
    seq_val_preds = predict_sequence(seq_model, val_df, target_col=target_col, time_col=time_col, group_by=group_by)
    seq_test_preds = predict_sequence(seq_model, test_df, target_col=target_col, time_col=time_col, group_by=group_by)

    val_stacked = np.column_stack([
        tab_val_preds.reindex(val_df.index).fillna(0).values,
        seq_val_preds.reindex(val_df.index).fillna(0).values,
    ])
    test_stacked = np.column_stack([
        tab_test_preds.reindex(test_df.index).fillna(0).values,
        seq_test_preds.reindex(test_df.index).fillna(0).values,
    ])
    y_val = val_df[target_col].values
    y_test = test_df[target_col].values

    meta = Ridge(alpha=1.0)
    meta.fit(val_stacked, y_val)
    final_test_preds = meta.predict(test_stacked)

    test_mae = float(mean_absolute_error(y_test, final_test_preds))
    test_r2 = float(r2_score(y_test, final_test_preds))
    logger.info("Ensemble test MAE: {:.4f}, R2: {:.4f}", test_mae, test_r2)

    artifacts = {
        "tabular_model": tabular_model,
        "sequence_model": seq_model,
        "meta_model": meta,
        "tabular_metrics": tab_metrics,
        "sequence_metrics": seq_metrics,
        "test_mae": test_mae,
        "test_r2": test_r2,
        "split_sizes": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
        },
    }
    return artifacts


def save_artifacts(artifacts: Dict[str, any], models_dir: Path) -> None:
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    if "sequence_model" in artifacts:
        torch.save(artifacts["sequence_model"].state_dict(), models_dir / "lstm_weights.pt")

    if "tabular_model" in artifacts:
        model = artifacts["tabular_model"]
        if hasattr(model, "save_model"):
            model.save_model(str(models_dir / "xgboost_model.json"))
        elif hasattr(model, "booster_"):
            model.booster_.save_model(str(models_dir / "xgboost_model.json"))
        else:
            joblib.dump(model, models_dir / "xgboost_model.joblib")

    if "meta_model" in artifacts:
        joblib.dump(artifacts["meta_model"], models_dir / "ensemble_meta.joblib")

    metrics = {
        "tabular_metrics": artifacts.get("tabular_metrics", {}),
        "sequence_metrics": artifacts.get("sequence_metrics", {}),
        "test_mae": artifacts.get("test_mae"),
        "test_r2": artifacts.get("test_r2"),
        "split_sizes": artifacts.get("split_sizes", {}),
    }
    with (models_dir / "training_metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    logger.info("Artifacts saved to {}", models_dir)


def run_training(
    features_dir: Path,
    models_dir: Path,
    input_file: str = "model_input.parquet",
) -> Dict[str, any]:
    features_dir = Path(features_dir)
    input_path = features_dir / input_file
    if not input_path.exists():
        raise FileNotFoundError(f"Feature file not found at {input_path}")

    df = pd.read_parquet(input_path)
    logger.info("Loaded feature matrix with shape {}", df.shape)

    artifacts = train_ensemble(df, models_dir=models_dir)
    save_artifacts(artifacts, models_dir)
    logger.info("Training pipeline completed successfully.")
    return artifacts
