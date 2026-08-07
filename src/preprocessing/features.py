from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd
from loguru import logger

from config.pipeline import load_config


def _ewma(series: pd.Series, halflife: float, min_periods: int = 2) -> pd.Series:
    return series.ewm(halflife=halflife, min_periods=min_periods, adjust=False).mean()


def compute_ewma_features(df: pd.DataFrame, stats: List[str]) -> pd.DataFrame:
    config = load_config().get("features", {})
    halflife = config.get("ewma_halflife", 3.0)
    min_periods = config.get("ewma_min_periods", 2)
    df = df.copy()
    for stat in stats:
        col = f"ewma_{stat}"
        if stat not in df.columns:
            logger.warning("Stat {} not found for EWMA", stat)
            continue
        df[col] = (
            df.groupby("element_id")[stat]
            .transform(lambda s: _ewma(s, halflife=halflife, min_periods=min_periods))
        )
    logger.info("EWMA features computed for stats: {}", stats)
    return df


def compute_starts_ratio(df: pd.DataFrame, starts_col: str = "starts", appearances_col: str = "appearances") -> pd.DataFrame:
    config = load_config().get("features", {})
    min_games = config.get("start_ratio_min_games", 3)
    df = df.copy()
    df["starts_ratio"] = 0.0
    mask = df[appearances_col].fillna(0) >= min_games
    df.loc[mask, "starts_ratio"] = df.loc[mask, starts_col].fillna(0) / df.loc[mask, appearances_col].replace(0, pd.NA)
    logger.info("Starts ratio computed")
    return df


def compute_venue_multipliers(df: pd.DataFrame, home_strength_col: str = "home_strength", away_strength_col: str = "away_strength") -> pd.DataFrame:
    config = load_config().get("features", {})
    window = config.get("venue_adjustment_window", 5)
    df = df.copy()
    if home_strength_col not in df.columns or away_strength_col not in df.columns:
        df["venue_multiplier"] = 1.0
        return df
    df["venue_multiplier"] = 1.0
    df["venue_strength_diff"] = (
        df[home_strength_col].rolling(window=window, min_periods=1).mean()
        - df[away_strength_col].rolling(window=window, min_periods=1).mean()
    )
    df["venue_multiplier"] = 1.0 + df["venue_strength_diff"].clip(-0.3, 0.3) * 0.5
    logger.info("Venue multipliers computed")
    return df


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    stats = [col for col in ["xG", "xA", "clean_sheets", "minutes"] if col in df.columns]
    df = compute_ewma_features(df, stats=stats)
    df = compute_starts_ratio(df)
    df = compute_venue_multipliers(df)
    logger.info("Feature matrix built with shape {}", df.shape)
    return df
