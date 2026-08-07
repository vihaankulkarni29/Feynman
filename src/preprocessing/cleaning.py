from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
from loguru import logger
from thefuzz import fuzz, process

from config.pipeline import load_config


_MANUAL_LOOKUP: Dict[str, str] = {
    "Man City": "Manchester City",
    "Man United": "Manchester United",
    "Spurs": "Tottenham",
    "Nott'm Forest": "Nottingham Forest",
}


def _normalize_team(name: str) -> str:
    name = name.strip()
    return _MANUAL_LOOKUP.get(name, name)


def fuzzy_match_players(
    source_names: Iterable[str],
    target_names: Iterable[str],
    threshold: int = 85,
) -> List[Tuple[str, str, int]]:
    target_list = list(target_names)
    matches: List[Tuple[str, str, int]] = []
    for name in source_names:
        best = process.extractOne(name, target_list, scorer=fuzz.token_sort_ratio)
        if best is None:
            continue
        target_name = best[0]
        score = best[1]
        if score >= threshold:
            matches.append((name, target_name, score))
        else:
            logger.debug("No fuzzy match above threshold for '{}' (best: {})", name, score)
    return matches


def resolve_entity_names(df: pd.DataFrame, column: str) -> pd.DataFrame:
    config = load_config().get("preprocessing", {})
    threshold = config.get("fuzzy_match_threshold", 85)
    df = df.copy()
    df[column] = df[column].astype(str).str.strip()
    df["team_normalized"] = df.get("team", "").map(_normalize_team)
    logger.info("Entity resolution complete for {}", column)
    return df


def impute_nan_by_position(df: pd.DataFrame, position_col: str) -> pd.DataFrame:
    config = load_config().get("preprocessing", {})
    if not config.get("position_median_fill", True):
        return df
    df = df.copy()
    numeric_cols = df.select_dtypes(include="number").columns
    for position, group in df.groupby(position_col):
        medians = group[numeric_cols].median()
        mask = df[position_col] == position
        df.loc[mask, numeric_cols] = df.loc[mask, numeric_cols].fillna(medians)
    logger.info("Position-specific NaN imputation complete")
    return df


def forward_fill_time_series(df: pd.DataFrame, group_by: str, time_col: str) -> pd.DataFrame:
    config = load_config().get("preprocessing", {})
    max_gap = config.get("forward_fill_gap_days", 7)
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values([group_by, time_col])
    if max_gap is not None and max_gap > 0:
        df["_gap"] = df.groupby(group_by)[time_col].diff().dt.days
        df.loc[df["_gap"] > max_gap, df.columns] = pd.NA
    df = df.groupby(group_by).ffill()
    if "_gap" in df.columns:
        df.drop(columns=["_gap"], inplace=True)
    logger.info("Forward-fill complete for {} by {}", time_col, group_by)
    return df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = resolve_entity_names(df, column="name")
    df = impute_nan_by_position(df, position_col="position")
    return df
