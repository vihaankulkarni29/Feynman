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


class FPLSequenceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = "target_xPts",
        seq_len: int = 5,
        group_by: str = "element_id",
        time_col: str = "gameweek",
    ) -> None:
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.seq_len = seq_len
        self.group_by = group_by
        self.time_col = time_col

        sequences: List[np.ndarray] = []
        targets: List[float] = []
        masks: List[np.ndarray] = []
        lengths: List[int] = []

        for _, group in df.groupby(group_by):
            group = group.sort_values(time_col).reset_index(drop=True)
            values = group[feature_cols].values.astype(np.float32)
            target_vals = group[target_col].values.astype(np.float32)

            if len(group) <= seq_len:
                padded = np.zeros((seq_len, len(feature_cols)), dtype=np.float32)
                mask = np.zeros(seq_len, dtype=np.float32)
                if len(group) > 0:
                    padded[-len(group):] = values
                    mask[-len(group):] = 1.0
                sequences.append(padded)
                targets.append(float(target_vals[-1]) if len(target_vals) > 0 else 0.0)
                masks.append(mask)
                lengths.append(min(len(group), seq_len))
                continue

            for i in range(seq_len, len(group)):
                seq = values[i - seq_len : i]
                target = target_vals[i]
                sequences.append(seq)
                targets.append(float(target))
                masks.append(np.ones(seq_len, dtype=np.float32))
                lengths.append(seq_len)

        if not sequences:
            raise ValueError("No sequences built; check data size and seq_len.")

        self.sequences = np.array(sequences, dtype=np.float32)
        self.targets = np.array(targets, dtype=np.float32)
        self.masks = np.array(masks, dtype=np.float32)
        self.lengths = np.array(lengths, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.sequences[idx], dtype=torch.float32),
            torch.tensor(self.targets[idx], dtype=torch.float32),
            torch.tensor(self.masks[idx], dtype=torch.float32),
            torch.tensor(self.lengths[idx], dtype=torch.long),
        )


class FPLLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(last).squeeze(-1)


def _build_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    seq_len: int,
    group_by: str,
    time_col: str,
) -> Tuple[np.ndarray, np.ndarray]:
    sequences = []
    targets = []
    for _, group in df.groupby(group_by):
        group = group.sort_values(time_col).reset_index(drop=True)
        values = group[feature_cols].values.astype(np.float32)
        target_vals = group[target_col].values.astype(np.float32)
        if len(group) <= seq_len:
            padded = np.zeros((seq_len, len(feature_cols)), dtype=np.float32)
            if len(group) > 0:
                padded[-len(group):] = values
            sequences.append(padded)
            targets.append(float(target_vals[-1]) if len(target_vals) > 0 else 0.0)
            continue
        for i in range(seq_len, len(group)):
            sequences.append(values[i - seq_len : i])
            targets.append(float(target_vals[i]))
    if not sequences:
        raise ValueError("No sequences built; check data size and seq_len.")
    return np.array(sequences, dtype=np.float32), np.array(targets, dtype=np.float32)


def train_sequence_model(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    target_col: str = "target_xPts",
    time_col: str = "gameweek",
    group_by: str = "element_id",
) -> Tuple[FPLLSTM, Dict[str, float]]:
    config = load_config()["models"]["sequence"]
    seq_len = config.get("input_window", 5)
    hidden_dim = config.get("hidden_dim", 64)
    num_layers = config.get("num_layers", 2)
    dropout = config.get("dropout", 0.2)
    learning_rate = config.get("learning_rate", 0.001)
    batch_size = config.get("batch_size", 32)
    epochs = config.get("epochs", 50)
    patience = config.get("patience", 5)
    device = torch.device(config.get("device", "cpu"))

    if feature_cols is None:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        feature_cols = [c for c in numeric_cols if c not in {target_col, group_by, time_col}]

    X, y = _build_sequences(df, feature_cols, target_col, seq_len, group_by, time_col)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    dataset = torch.utils.data.TensorDataset(X, y)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = FPLLSTM(input_dim=X.shape[2], hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout).to(device)
    criterion = nn.HuberLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    best_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                preds = model(batch_x)
                val_loss += criterion(preds, batch_y).item() * batch_x.size(0)
        val_loss /= len(val_loader.dataset)

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

    metrics = {"val_huber": best_loss}
    logger.info("Sequence model trained. Best val Huber: {:.4f}", best_loss)
    return model, metrics


def predict_sequence(
    model: FPLLSTM,
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    target_col: str = "target_xPts",
    time_col: str = "gameweek",
    group_by: str = "element_id",
) -> pd.Series:
    config = load_config()["models"]["sequence"]
    seq_len = config.get("input_window", 5)
    device = torch.device(config.get("device", "cpu"))

    if feature_cols is None:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        feature_cols = [c for c in numeric_cols if c not in {target_col, group_by, time_col}]

    model.eval()
    preds = []
    indices = []
    with torch.no_grad():
        for element_id, group in df.groupby(group_by):
            group = group.sort_values(time_col).reset_index(drop=True)
            if len(group) < seq_len:
                continue
            seq = group[feature_cols].values[-seq_len:]
            x = torch.tensor(seq, dtype=torch.float32, device=device).unsqueeze(0)
            pred = model(x).item()
            preds.append(pred)
            indices.append(element_id)

    return pd.Series(preds, index=indices, name="xPts_sequence")
