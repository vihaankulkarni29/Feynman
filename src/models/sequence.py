from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger

from config.pipeline import load_config


class LSTMPredictor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out.squeeze(-1)


def _build_sequences(df: pd.DataFrame, feature_cols: List[str], window: int) -> Tuple[np.ndarray, np.ndarray]:
    sequences = []
    targets = []
    for _, group in df.groupby("element_id"):
        group = group.sort_values("gameweek").reset_index(drop=True)
        if len(group) <= window:
            continue
        values = group[feature_cols].values
        target_vals = group["total_points"].values
        for i in range(window, len(group)):
            sequences.append(values[i - window : i])
            targets.append(target_vals[i])
    if not sequences:
        raise ValueError("No sequences built; check window and data size.")
    return np.array(sequences, dtype=np.float32), np.array(targets, dtype=np.float32)


def train_sequence_model(df: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> Tuple[LSTMPredictor, Dict[str, float]]:
    config = load_config()["models"]["sequence"]
    window = config.get("input_window", 5)
    hidden_dim = config.get("hidden_dim", 64)
    num_layers = config.get("num_layers", 2)
    dropout = config.get("dropout", 0.3)
    learning_rate = config.get("learning_rate", 0.001)
    batch_size = config.get("batch_size", 32)
    epochs = config.get("epochs", 50)
    patience = config.get("patience", 10)
    device = torch.device(config.get("device", "cpu"))

    if feature_cols is None:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        feature_cols = [c for c in numeric_cols if c not in {"element_id", "gameweek", "total_points"}]

    X, y = _build_sequences(df, feature_cols=feature_cols, window=window)
    X = torch.tensor(X, device=device)
    y = torch.tensor(y, device=device)

    model = LSTMPredictor(input_dim=X.shape[2], hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(X.size(0))
        train_loss = 0.0
        for i in range(0, X.size(0), batch_size):
            idx = permutation[i : i + batch_size]
            batch_x, batch_y = X[idx], y[idx]
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)
        train_loss /= X.size(0)

        model.eval()
        with torch.no_grad():
            val_preds = model(X)
            val_loss = float(criterion(val_preds, y).item())

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                logger.info("Early stopping at epoch {}", epoch)
                break

        if epoch % 5 == 0:
            logger.debug("Epoch {} train_loss={:.4f} val_loss={:.4f}", epoch, train_loss, val_loss)

    if best_state is not None:
        model.load_state_dict(best_state)

    metrics = {"val_mse": best_loss}
    logger.info("Sequence model trained. Best val MSE: {:.4f}", best_loss)
    return model, metrics


def predict_sequence(model: LSTMPredictor, df: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> pd.Series:
    config = load_config()["models"]["sequence"]
    window = config.get("input_window", 5)
    device = torch.device(config.get("device", "cpu"))

    if feature_cols is None:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        feature_cols = [c for c in numeric_cols if c not in {"element_id", "gameweek", "total_points"}]

    preds = []
    indices = []
    model.eval()
    with torch.no_grad():
        for element_id, group in df.groupby("element_id"):
            group = group.sort_values("gameweek").reset_index(drop=True)
            if len(group) <= window:
                continue
            seq = group[feature_cols].values[-window:]
            x = torch.tensor(seq, dtype=torch.float32, device=device).unsqueeze(0)
            pred = model(x).item()
            preds.append(pred)
            indices.append(element_id)

    return pd.Series(preds, index=indices, name="xPts_sequence")
