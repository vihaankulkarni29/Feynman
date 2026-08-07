from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Dict, Optional

from loguru import logger

import joblib
import pandas as pd
import torch

from config.pipeline import load_config
from src.ingestion import run_ingestion
from src.preprocessing.cleaning import clean_dataframe
from src.preprocessing.features import build_feature_matrix
from src.models.tabular import train_tabular_model, predict_tabular
from src.models.sequence import LSTMPredictor, train_sequence_model, predict_sequence
from src.optimization.solver import solve_squad_selection, solve_transfer_plan, solve_problem
from src.optimization.simulator import monte_carlo_squad


def _raw_data_exists(raw_dir: Path) -> bool:
    files = list(raw_dir.glob("bootstrap_static.json")) + list(raw_dir.glob("player_*_summary.json"))
    return len(files) > 0


def step_ingest(raw_dir: Path) -> None:
    if not _raw_data_exists(raw_dir):
        logger.info("Running ingestion...")
        asyncio.run(run_ingestion(raw_dir))
    else:
        logger.info("Raw data already present. Skipping ingestion.")


def step_preprocess(raw_dir: Path, processed_dir: Path) -> pd.DataFrame:
    logger.info("Preprocessing data...")
    records = []
    for path in raw_dir.glob("player_*_summary.json"):
        import json
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        history = data.get("history", [])
        for row in history:
            row["element_id"] = path.stem.replace("player_", "").replace("_summary", "")
            records.append(row)
    if not records:
        raise ValueError("No element summary records found.")
    df = pd.DataFrame(records)
    df = clean_dataframe(df)
    processed_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(processed_dir / "processed.parquet", index=False)
    logger.info("Processed data saved with shape {}", df.shape)
    return df


def step_features(processed_dir: Path, features_dir: Path) -> pd.DataFrame:
    logger.info("Building features...")
    df = pd.read_parquet(processed_dir / "processed.parquet")
    df = build_feature_matrix(df)
    features_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(features_dir / "features.parquet", index=False)
    logger.info("Feature matrix saved with shape {}", df.shape)
    return df


def step_train(features_dir: Path, models_dir: Path) -> None:
    logger.info("Training models...")
    df = pd.read_parquet(features_dir / "features.parquet")
    tabular_model, tabular_metrics = train_tabular_model(df, target="total_points")
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(tabular_model, models_dir / "tabular_model.joblib")
    logger.info("Tabular model saved. Metrics: {}", tabular_metrics)

    seq_model, seq_metrics = train_sequence_model(df)
    torch.save(seq_model.state_dict(), models_dir / "sequence_model.pt")
    logger.info("Sequence model saved. Metrics: {}", seq_metrics)


def step_predict(features_dir: Path, models_dir: Path) -> pd.DataFrame:
    logger.info("Generating predictions...")
    df = pd.read_parquet(features_dir / "features.parquet")
    tabular_model = joblib.load(models_dir / "tabular_model.joblib")
    tabular_preds = predict_tabular(tabular_model, df)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in {"element_id", "gameweek", "total_points"}]
    seq_model = LSTMPredictor(input_dim=len(feature_cols), hidden_dim=64, num_layers=2, dropout=0.3)
    seq_model.load_state_dict(torch.load(models_dir / "sequence_model.pt", map_location="cpu"))
    seq_preds = predict_sequence(seq_model, df, feature_cols=feature_cols)

    df = df.merge(tabular_preds, left_index=True, right_index=True, how="left")
    df = df.merge(seq_preds, left_on="element_id", right_index=True, how="left")
    config = load_config()["models"]["ensemble"]
    tw = config.get("tabular_weight", 0.7)
    sw = config.get("sequence_weight", 0.3)
    df["xPts"] = tw * df["xPts_tabular"].fillna(0) + sw * df["xPts_sequence"].fillna(0)
    return df


def step_optimize(pred_df: pd.DataFrame, models_dir: Path) -> Dict[str, float]:
    logger.info("Optimizing squad...")
    config = load_config()["optimization"]
    problem = solve_squad_selection(pred_df, horizon_gws=config.get("horizon_gws", 5))
    solution = solve_problem(problem)
    if solution is None:
        logger.warning("No optimal solution found")
        return {}
    return solution


def step_simulate(pred_df: pd.DataFrame) -> Dict[str, float]:
    logger.info("Running Monte Carlo simulation...")
    return monte_carlo_squad(pred_df, horizon_gws=5)


def run_pipeline(mode: str = "run") -> None:
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    features_dir = Path("data/features")
    models_dir = Path("models")

    if mode == "run":
        step_ingest(raw_dir)
        df = step_preprocess(raw_dir, processed_dir)
        df = step_features(processed_dir, features_dir)
        step_train(features_dir, models_dir)
        pred_df = step_predict(features_dir, models_dir)
        step_optimize(pred_df, models_dir)
        step_simulate(pred_df)
    elif mode == "ingest-only":
        step_ingest(raw_dir)
    elif mode == "optimize-only":
        df = step_preprocess(raw_dir, processed_dir)
        df = step_features(processed_dir, features_dir)
        pred_df = step_predict(features_dir, models_dir)
        step_optimize(pred_df, models_dir)
    elif mode == "simulate-only":
        df = step_preprocess(raw_dir, processed_dir)
        df = step_features(processed_dir, features_dir)
        pred_df = step_predict(features_dir, models_dir)
        step_simulate(pred_df)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    logger.info("Pipeline completed successfully.")


def main() -> int:
    parser = argparse.ArgumentParser(description="FPL Moneyball pipeline")
    parser.add_argument("mode", nargs="?", default="run", choices=["run", "ingest-only", "optimize-only", "simulate-only"])
    args = parser.parse_args()
    logger.add(
        "logs/pipeline_{time}.log",
        rotation="10 MB",
        encoding="utf-8",
        level="INFO",
    )
    try:
        run_pipeline(args.mode)
    except Exception as exc:
        logger.exception("Pipeline failed: {}", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
