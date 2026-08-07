from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

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


def build_entity_mapping(bootstrap_path: Path, processed_dir: Path) -> Path:
    """Build master entity_mapping.json from bootstrap_static.json."""
    if not bootstrap_path.exists():
        raise FileNotFoundError(f"bootstrap_static.json not found at {bootstrap_path}")

    with bootstrap_path.open("r", encoding="utf-8") as fh:
        bootstrap = json.load(fh)

    elements = bootstrap.get("elements", [])
    teams = {t["id"]: t["name"] for t in bootstrap.get("teams", [])}
    positions = {t["id"]: t["singular_name"] for t in bootstrap.get("element_types", [])}

    mapping: Dict[str, Any] = {}
    for p in elements:
        element_id = str(p.get("id", ""))
        mapping[element_id] = {
            "element_id": p.get("id"),
            "first_name": p.get("first_name", ""),
            "second_name": p.get("second_name", ""),
            "web_name": p.get("web_name", ""),
            "team_id": p.get("team"),
            "team_name": teams.get(p.get("team"), ""),
            "position_id": p.get("element_type"),
            "position": positions.get(p.get("element_type"), ""),
            "now_cost": p.get("now_cost"),
            "cost_normalized": _normalize_cost(p.get("now_cost")),
            "status": p.get("status", ""),
            "chance_of_playing_this_round": p.get("chance_of_playing_this_round"),
            "chance_of_playing_next_round": p.get("chance_of_playing_next_round"),
        }

    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / "entity_mapping.json"
    tmp = out_path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(mapping, fh, ensure_ascii=False, indent=2)
    tmp.replace(out_path)
    logger.info("Entity mapping saved with {} players", len(mapping))
    return out_path


def load_entity_mapping(processed_dir: Path) -> Dict[str, Any]:
    path = processed_dir / "entity_mapping.json"
    if not path.exists():
        raise FileNotFoundError(f"entity_mapping.json not found at {path}. Run build_entity_mapping() first.")
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _normalize_cost(cost_tenths: Any) -> Optional[float]:
    if cost_tenths is None:
        return None
    try:
        return float(cost_tenths) / 10.0
    except (TypeError, ValueError):
        return None


def _normalize_kickoff_to_utc(kickoff_time: Any) -> Optional[pd.Timestamp]:
    if kickoff_time is None or pd.isna(kickoff_time):
        return None
    try:
        ts = pd.to_datetime(str(kickoff_time), utc=True)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return ts
    except Exception:
        return None


def _infer_player_status(row: pd.Series) -> str:
    minutes = row.get("minutes", pd.NA)
    chance_this = row.get("chance_of_playing_this_round", pd.NA)
    chance_next = row.get("chance_of_playing_next_round", pd.NA)

    if pd.notna(chance_this) and chance_this == 0:
        return "injured"
    if pd.notna(chance_next) and chance_next == 0:
        return "injured"

    if pd.isna(minutes):
        return "dnp"

    if pd.notna(chance_this) and chance_this is not None:
        if minutes == 0:
            return "benched"
        return "active"

    if pd.notna(chance_next) and chance_next is not None:
        if minutes == 0:
            return "benched"
        return "active"

    if minutes == 0:
        return "dnp"

    return "active"


def apply_status_flags(df: pd.DataFrame, entity_mapping: Dict[str, Any]) -> pd.DataFrame:
    df = df.copy()

    if "chance_of_playing_this_round" not in df.columns or "chance_of_playing_next_round" not in df.columns:
        if "element_id" in df.columns:
            df["chance_of_playing_this_round"] = df["element_id"].astype(str).map(
                lambda k: entity_mapping.get(str(k), {}).get("chance_of_playing_this_round")
            )
            df["chance_of_playing_next_round"] = df["element_id"].astype(str).map(
                lambda k: entity_mapping.get(str(k), {}).get("chance_of_playing_next_round")
            )

    df["player_status"] = df.apply(_infer_player_status, axis=1)
    df["is_benched"] = df["player_status"] == "benched"
    df["is_injured_dnp"] = df["player_status"].isin(["injured", "dnp"])
    df["masked_for_ewma"] = df["is_injured_dnp"]
    logger.info("Status flags applied: {} benched, {} injured/dnp",
                df["is_benched"].sum(), df["is_injured_dnp"].sum())
    return df


def normalize_volume_metrics(df: pd.DataFrame, volume_cols: Optional[List[str]] = None) -> pd.DataFrame:
    if volume_cols is None:
        volume_cols = ["expected_goals", "expected_assists", "shots", "goals_scored", "assists", "clean_sheets"]

    df = df.copy()
    existing_cols = [c for c in volume_cols if c in df.columns]
    if not existing_cols:
        return df

    for col in existing_cols:
        df[col] = df[col].astype(float)

    benched_mask = df.get("is_benched", False)
    for col in existing_cols:
        df.loc[benched_mask, col] = 0.0

    logger.info("Volume metrics normalized: benched rows zeroed for {}", existing_cols)
    return df


def normalize_costs(df: pd.DataFrame, cost_col: str = "now_cost") -> pd.DataFrame:
    df = df.copy()
    if cost_col in df.columns:
        df["cost_normalized"] = df[cost_col].apply(_normalize_cost)
    logger.info("Cost normalization complete")
    return df


def normalize_dates(df: pd.DataFrame, date_col: str = "kickoff_time") -> pd.DataFrame:
    df = df.copy()
    if date_col in df.columns:
        df[f"{date_col}_utc"] = df[date_col].apply(_normalize_kickoff_to_utc)
    logger.info("Date normalization complete")
    return df


def resolve_entity_names(df: pd.DataFrame, column: str = "name") -> pd.DataFrame:
    config = load_config().get("preprocessing", {})
    threshold = config.get("fuzzy_match_threshold", 85)
    df = df.copy()
    df[column] = df[column].astype(str).str.strip()
    df["team_normalized"] = df.get("team", "").map(_normalize_team)
    logger.info("Entity resolution complete for {}", column)
    return df


def impute_nan_by_position(df: pd.DataFrame, position_col: str = "position") -> pd.DataFrame:
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


def clean_dataframe(df: pd.DataFrame, entity_mapping: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    df = resolve_entity_names(df, column="name")
    df = normalize_costs(df, cost_col="now_cost")
    df = normalize_dates(df, date_col="kickoff_time")
    if entity_mapping is not None:
        df = apply_status_flags(df, entity_mapping)
        df = normalize_volume_metrics(df)
    df = impute_nan_by_position(df, position_col="position")
    return df
