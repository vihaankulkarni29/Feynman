from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger

from config.pipeline import load_config


def _sort_for_time_series(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    sort_cols = ["element_id"]
    if "gameweek" in df.columns:
        sort_cols.append("gameweek")
    elif "round" in df.columns:
        sort_cols.append("round")
    elif "kickoff_time_utc" in df.columns:
        sort_cols.append("kickoff_time_utc")
    return df.sort_values(sort_cols).reset_index(drop=True)


def _ewma_span(series: pd.Series, span: float, min_periods: int = 2) -> pd.Series:
    return series.ewm(span=span, min_periods=min_periods, adjust=False).mean()


def compute_ewma_features_masked(
    df: pd.DataFrame,
    metrics: Optional[List[str]] = None,
) -> pd.DataFrame:
    config = load_config().get("features", {}).get("ewma", {})
    spans = config.get("spans", [3, 5, 10])
    min_periods = config.get("min_periods", 2)

    if metrics is None:
        metrics = config.get("metrics", ["expected_goals", "expected_assists", "clearances_blocks_interceptions", "total_points", "minutes"])

    df = _sort_for_time_series(df)
    df = df.copy()

    for metric in metrics:
        if metric not in df.columns:
            logger.warning("Metric {} not found for EWMA; skipping", metric)
            continue
        for span in spans:
            out_col = f"ewma_{metric}_span{span}"

            def _ewma_for_group(group: pd.DataFrame, m: str = metric, s: float = span, mp: int = min_periods) -> pd.Series:
                values = group[m].astype(float)
                if "masked_for_ewma" in group.columns:
                    values = values.where(~group["masked_for_ewma"])
                return _ewma_span(values, span=s, min_periods=mp)

            df[out_col] = df.groupby("element_id", group_keys=False).apply(
                lambda g: _ewma_for_group(g), include_groups=False
            ).reset_index(drop=True)

    logger.info("Masked EWMA features computed for metrics={} spans={}", metrics, spans)
    return df


def export_ewma_isolated(df: pd.DataFrame, features_dir: Path) -> Optional[Path]:
    config = load_config().get("features", {}).get("ewma", {})
    if not config.get("export_isolated_csv", False):
        return None
    ewma_cols = [c for c in df.columns if c.startswith("ewma_")]
    if not ewma_cols:
        return None
    iso = df[["element_id"] + ewma_cols].copy()
    features_dir = Path(features_dir)
    features_dir.mkdir(parents=True, exist_ok=True)
    out_path = features_dir / "ewma_isolated.csv"
    tmp = out_path.with_suffix(".tmp")
    iso.to_csv(tmp, index=False)
    tmp.replace(out_path)
    logger.info("EWMA isolated CSV exported with {} columns", len(ewma_cols))
    return out_path


def compute_rotation_risk(df: pd.DataFrame) -> pd.DataFrame:
    config = load_config().get("features", {}).get("rotation_risk", {})
    window = config.get("window", 6)
    starts_col = config.get("starts_col", "starts")
    appearances_col = config.get("appearances_col", "appearances")
    min_games = config.get("min_games", 3)

    df = _sort_for_time_series(df)
    df = df.copy()

    if starts_col not in df.columns or appearances_col not in df.columns:
        logger.warning("Starts/appearances columns missing; setting rotation_risk=0")
        df["rotation_risk"] = 0.0
        return df

    df["rolling_starts"] = df.groupby("element_id")[starts_col].transform(lambda s: s.rolling(window=window, min_periods=1).sum())
    df["rolling_appearances"] = df.groupby("element_id")[appearances_col].transform(lambda s: s.rolling(window=window, min_periods=1).sum())

    df["rotation_risk"] = 0.0
    mask = df["rolling_appearances"].fillna(0) >= min_games
    df.loc[mask, "rotation_risk"] = df.loc[mask, "rolling_starts"].fillna(0) / df.loc[mask, "rolling_appearances"].replace(0, pd.NA)

    df.drop(columns=["rolling_starts", "rolling_appearances"], inplace=True)
    logger.info("Rotation risk computed with window={}", window)
    return df


def load_tactical_cache(cache_path: Optional[str] = None) -> pd.DataFrame:
    config = load_config().get("features", {}).get("tactical", {})
    if cache_path is None:
        cache_path = config.get("historical_cache", "data/raw/tactical/tactical_metrics_historical.csv")
    path = Path(cache_path)
    if not path.exists():
        logger.warning("Tactical cache not found at {}", path)
        return pd.DataFrame()
    df = pd.read_csv(path)
    logger.info("Loaded {} rows from tactical cache {}", len(df), path)
    return df


def compute_tactical_multipliers(df: pd.DataFrame, tactical_df: pd.DataFrame) -> pd.DataFrame:
    config = load_config().get("features", {}).get("tactical", {})
    defensive_weight = config.get("defensive_solidity_weight", 0.5)
    attacking_weight = config.get("attacking_intensity_weight", 0.5)

    df = df.copy()
    if tactical_df.empty:
        logger.warning("Empty tactical DataFrame; setting tactical_multiplier=1.0")
        df["tactical_multiplier"] = 1.0
        return df

    team_meta = tactical_df.copy()
    team_meta["defensive_solidity_index"] = (
        team_meta["defensive_third_field_tilt"].fillna(50.0) * 0.6
        + team_meta["PPDA_allowed"].fillna(12.0).rdiv(1).mul(10) * 0.4
    ).clip(0.5, 1.5)

    team_meta["attacking_intensity_index"] = (
        team_meta["final_third_tackles"].fillna(15.0).rdiv(1).mul(2) * 0.5
        + team_meta["shots"].fillna(450).rdiv(1).mul(0.1) * 0.5
    ).clip(0.5, 1.5)

    team_meta["tactical_multiplier"] = (
        defensive_weight * team_meta["defensive_solidity_index"]
        + attacking_weight * team_meta["attacking_intensity_index"]
    )

    if "team_normalized" in df.columns:
        df = df.merge(
            team_meta[["team", "season", "tactical_multiplier", "defensive_solidity_index", "attacking_intensity_index"]],
            left_on=["team_normalized"],
            right_on=["team"],
            how="left",
        )
    elif "team" in df.columns:
        df = df.merge(
            team_meta[["team", "season", "tactical_multiplier", "defensive_solidity_index", "attacking_intensity_index"]],
            on=["team", "season"],
            how="left",
        )
    else:
        df["tactical_multiplier"] = 1.0

    df["tactical_multiplier"] = df["tactical_multiplier"].fillna(1.0)
    logger.info("Tactical multipliers merged; mean={:.3f}", df["tactical_multiplier"].mean())
    return df


def compute_positional_xpts(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    position_col = None
    for candidate in ["position", "position_normalized", "element_type"]:
        if candidate in df.columns:
            position_col = candidate
            break

    if position_col is None:
        logger.warning("No position column found; setting target_xPts=0")
        df["target_xPts"] = 0.0
        return df

    def _safe(val: Any) -> float:
        if val is None or pd.isna(val):
            return 0.0
        try:
            return float(val)
        except (TypeError, ValueError):
            return 0.0

    def _row_xpts(row: pd.Series) -> float:
        pos = str(row.get(position_col, "")).upper()
        e_xg = _safe(row.get("ewma_expected_goals_span5"))
        e_xa = _safe(row.get("ewma_expected_assists_span5"))
        cbi = _safe(row.get("ewma_clearances_blocks_interceptions_span5"))
        cs_prob = _safe(row.get("ewma_clean_sheets_span5"))
        base = _safe(row.get("ewma_total_points_span5")) * 0.5

        if pos in ["GK", "DEFENDER", "DEF"]:
            return base + (cs_prob * 4) + (e_xg * 6) + (e_xa * 3) + (1.0 if cbi >= 10 else 0.0) * 2
        if pos in ["MIDFIELDER", "MID"]:
            return base + (cs_prob * 1) + (e_xg * 5) + (e_xa * 3) + (1.0 if cbi >= 12 else 0.0) * 2
        if pos in ["FORWARD", "FWD"]:
            return base + (e_xg * 4) + (e_xa * 3) + (1.0 if cbi >= 12 else 0.0) * 2
        return base + e_xg + e_xa

    df["target_xPts"] = df.apply(_row_xpts, axis=1)
    logger.info("Positional target_xPts formulated")
    return df


def drop_non_predictive(df: pd.DataFrame, keep_ids: bool = True) -> pd.DataFrame:
    df = df.copy()
    string_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    if keep_ids:
        id_like = [c for c in string_cols if "id" in c.lower() or c in ["web_name", "team_normalized", "position"]]
        string_cols = [c for c in string_cols if c not in id_like]
    df.drop(columns=string_cols, inplace=True, errors="ignore")
    return df


def immutable_write_parquet(df: pd.DataFrame, features_dir: Path, basename: str) -> Path:
    from datetime import datetime, timezone
    from config.pipeline import load_config

    features_dir = Path(features_dir)
    features_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    versioned = features_dir / f"{basename}.{timestamp}.parquet"
    current = features_dir / f"{basename}.parquet"

    tmp = versioned.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(versioned)

    tmp2 = current.with_suffix(".tmp")
    df.to_parquet(tmp2, index=False)
    tmp2.replace(current)

    logger.info("Immutable parquet write complete: {} -> {}", versioned.name, current.name)
    return current


def generate_features(
    df: pd.DataFrame,
    features_dir: Path,
    entity_mapping: Optional[Dict[str, Any]] = None,
    tactical_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    df = df.copy()
    features_dir = Path(features_dir)

    df = compute_ewma_features_masked(df)
    export_ewma_isolated(df, features_dir)

    df = compute_rotation_risk(df)

    if tactical_df is None:
        tactical_df = load_tactical_cache()
    df = compute_tactical_multipliers(df, tactical_df)

    df = compute_positional_xpts(df)

    df = drop_non_predictive(df, keep_ids=True)

    out_path = immutable_write_parquet(df, features_dir, "model_input")
    logger.info("Feature generation complete. Output shape: {} -> {}", df.shape, out_path)
    return df
