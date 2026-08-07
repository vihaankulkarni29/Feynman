from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from config.pipeline import load_config


def simulate_player_points(
    mean_points: float,
    std_points: float,
    n_sims: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng(42)
    if std_points is None or std_points <= 0:
        std_points = max(mean_points * 0.5, 1.0)
    return rng.normal(loc=mean_points, scale=std_points, size=n_sims)


def monte_carlo_squad(
    squad_df: pd.DataFrame,
    n_sims: int = 1000,
    horizon_gws: int = 5,
) -> Dict[str, float]:
    config = load_config().get("optimization", {})
    rng = np.random.default_rng(42)
    total_points = np.zeros(n_sims)
    for _, row in squad_df.iterrows():
        mean = float(row.get("xPts", 0.0)) * horizon_gws
        std = float(row.get("xPts_std", mean * 0.4))
        sims = simulate_player_points(mean, std, n_sims=n_sims, rng=rng)
        total_points += sims

    results = {
        "mean_total": float(np.mean(total_points)),
        "p10": float(np.percentile(total_points, 10)),
        "p90": float(np.percentile(total_points, 90)),
        "std_total": float(np.std(total_points)),
    }
    logger.info("Monte Carlo simulation complete. Mean total xPts: {:.2f}", results["mean_total"])
    return results


def probability_above_threshold(
    squad_df: pd.DataFrame,
    threshold: float,
    n_sims: int = 1000,
    horizon_gws: int = 5,
) -> float:
    rng = np.random.default_rng(42)
    total_points = np.zeros(n_sims)
    for _, row in squad_df.iterrows():
        mean = float(row.get("xPts", 0.0)) * horizon_gws
        std = float(row.get("xPts_std", mean * 0.4))
        sims = simulate_player_points(mean, std, n_sims=n_sims, rng=rng)
        total_points += sims
    prob = float(np.mean(total_points >= threshold))
    logger.info("Probability of exceeding {:.2f} xPts: {:.2%}", threshold, prob)
    return prob
