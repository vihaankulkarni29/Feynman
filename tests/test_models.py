from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.models.sequence import FPLLSTM, FPLSequenceDataset, train_sequence_model, predict_sequence
from src.models.tabular import train_tabular_model, predict_tabular, export_feature_importance
from src.models.train import _chronological_split, train_ensemble, save_artifacts


def _make_sample_df(n_players: int = 5, n_gws: int = 8) -> pd.DataFrame:
    rows = []
    for pid in range(n_players):
        for gw in range(1, n_gws + 1):
            rows.append({
                "element_id": pid,
                "gameweek": gw,
                "target_xPts": float(gw % 5),
                "expected_goals": 0.1 * gw,
                "expected_assists": 0.05 * gw,
                "minutes": 90.0,
                "starts": 1,
                "appearances": 1,
                "clearances_blocks_interceptions": 5.0 + gw,
                "clean_sheets": 1 if gw % 2 == 0 else 0,
                "total_points": float(gw % 6),
            })
    return pd.DataFrame(rows)


def test_fpl_sequence_dataset_shape():
    df = _make_sample_df(n_players=3, n_gws=8)
    feature_cols = ["expected_goals", "expected_assists", "minutes", "clearances_blocks_interceptions"]
    ds = FPLSequenceDataset(df, feature_cols=feature_cols, target_col="target_xPts", seq_len=5)
    assert len(ds) > 0
    seq, target, mask, length = ds[0]
    assert seq.shape == (5, len(feature_cols))
    assert target.shape == ()
    assert mask.shape == (5,)
    assert length.item() == 5


def test_fpl_sequence_dataset_short_player():
    df = pd.DataFrame([
        {"element_id": 0, "gameweek": 1, "target_xPts": 1.0, "expected_goals": 0.1, "expected_assists": 0.0, "minutes": 90.0, "clearances_blocks_interceptions": 3.0},
        {"element_id": 0, "gameweek": 2, "target_xPts": 2.0, "expected_goals": 0.2, "expected_assists": 0.1, "minutes": 90.0, "clearances_blocks_interceptions": 4.0},
    ])
    feature_cols = ["expected_goals", "expected_assists", "minutes", "clearances_blocks_interceptions"]
    ds = FPLSequenceDataset(df, feature_cols=feature_cols, target_col="target_xPts", seq_len=5)
    assert len(ds) == 1
    seq, target, mask, length = ds[0]
    assert seq.shape == (5, len(feature_cols))
    assert int(mask.sum().item()) == 2
    assert length.item() == 2


def test_fpl_lstm_forward():
    model = FPLLSTM(input_dim=4, hidden_dim=64, num_layers=2, dropout=0.2)
    x = torch.randn(8, 5, 4)
    out = model(x)
    assert out.shape == (8,)


def test_train_sequence_model_runs():
    df = _make_sample_df(n_players=5, n_gws=8)
    model, metrics = train_sequence_model(df, target_col="target_xPts")
    assert isinstance(model, FPLLSTM)
    assert "val_huber" in metrics


def test_predict_sequence_runs():
    df = _make_sample_df(n_players=5, n_gws=8)
    model, _ = train_sequence_model(df, target_col="target_xPts")
    preds = predict_sequence(model, df, target_col="target_xPts")
    assert isinstance(preds, pd.Series)
    assert preds.name == "xPts_sequence"


def test_train_tabular_model_runs():
    df = _make_sample_df(n_players=10, n_gws=8)
    model, metrics = train_tabular_model(df, target="target_xPts")
    assert "mae" in metrics
    assert "r2" in metrics


def test_predict_tabular_runs():
    df = _make_sample_df(n_players=10, n_gws=8)
    model, _ = train_tabular_model(df, target="target_xPts")
    preds = predict_tabular(model, df)
    assert isinstance(preds, pd.Series)
    assert preds.name == "xPts_tabular"


def test_export_feature_importance(tmp_path: Path):
    df = _make_sample_df(n_players=10, n_gws=8)
    model, _ = train_tabular_model(df, target="target_xPts")
    feature_names = [c for c in df.columns if c not in ["element_id", "gameweek", "target_xPts"]]
    out = export_feature_importance(model, feature_names, tmp_path / "feature_importances.json")
    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert "feature_importances" in data


def test_chronological_split_by_gameweek():
    rows = []
    for pid in range(10):
        for gw in range(1, 26):
            rows.append({
                "element_id": pid,
                "gameweek": gw,
                "target_xPts": float(gw % 5),
            })
    df = pd.DataFrame(rows)
    train_df, val_df, test_df = _chronological_split(df, time_col="gameweek", validate_first_n_gws=10)
    assert len(train_df) == 100
    assert len(val_df) == 100
    assert len(test_df) == 50
    assert set(train_df["gameweek"].unique()) == set(range(1, 11))
    assert set(val_df["gameweek"].unique()) == set(range(11, 21))
    assert set(test_df["gameweek"].unique()) == set(range(21, 26))


def test_train_ensemble_runs(tmp_path: Path):
    df = _make_sample_df(n_players=10, n_gws=8)
    artifacts = train_ensemble(df, models_dir=tmp_path)
    assert "tabular_model" in artifacts
    assert "sequence_model" in artifacts
    assert "meta_model" in artifacts
    assert "test_mae" in artifacts


def test_save_artifacts_creates_files(tmp_path: Path):
    df = _make_sample_df(n_players=10, n_gws=8)
    artifacts = train_ensemble(df, models_dir=tmp_path)
    save_artifacts(artifacts, tmp_path)
    assert (tmp_path / "lstm_weights.pt").exists()
    assert (tmp_path / "xgboost_model.json").exists() or (tmp_path / "xgboost_model.joblib").exists()
    assert (tmp_path / "ensemble_meta.joblib").exists()
    assert (tmp_path / "training_metrics.json").exists()
