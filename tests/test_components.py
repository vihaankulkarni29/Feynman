from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.preprocessing.cleaning import (
    fuzzy_match_players,
    impute_nan_by_position,
    resolve_entity_names,
)
from src.preprocessing.features import (
    compute_ewma_features,
    compute_starts_ratio,
    compute_venue_multipliers,
)
from src.optimization.solver import solve_squad_selection, solve_transfer_plan, solve_problem


def test_resolve_entity_names_adds_normalized_team():
    df = pd.DataFrame({"name": ["A. Player"], "team": ["Man City"], "position": ["MID"]})
    result = resolve_entity_names(df, column="name")
    assert "team_normalized" in result.columns
    assert result.loc[0, "team_normalized"] == "Manchester City"


def test_fuzzy_match_players_threshold():
    source = ["Mohamed Salah"]
    target = ["Mohamed Salah", "Erling Haaland"]
    matches = fuzzy_match_players(source, target, threshold=90)
    assert len(matches) == 1
    assert matches[0][1] == "Mohamed Salah"


def test_impute_nan_by_position_fills_numeric():
    df = pd.DataFrame({
        "name": ["A", "B", "C", "D"],
        "team": ["T", "T", "T", "T"],
        "position": ["MID", "MID", "DEF", "DEF"],
        "xG": [0.5, None, None, 0.3],
        "minutes": [90.0, None, 180.0, 60.0],
    })
    result = impute_nan_by_position(df, position_col="position")
    assert result["xG"].isna().sum() == 0
    assert result["minutes"].isna().sum() == 0


def test_compute_starts_ratio():
    df = pd.DataFrame({
        "starts": [10, 2, 5],
        "appearances": [10, 5, 5],
    })
    result = compute_starts_ratio(df)
    assert result.loc[0, "starts_ratio"] == pytest.approx(1.0)
    assert result.loc[1, "starts_ratio"] == pytest.approx(0.4)
    assert result.loc[2, "starts_ratio"] == pytest.approx(1.0)


def test_compute_venue_multipliers():
    df = pd.DataFrame({
        "home_strength": [80.0, 70.0, 60.0],
        "away_strength": [60.0, 80.0, 70.0],
    })
    result = compute_venue_multipliers(df)
    assert "venue_multiplier" in result.columns
    assert result["venue_multiplier"].notna().all()


def test_compute_ewma_features():
    df = pd.DataFrame({
        "element_id": [1, 1, 1, 2, 2, 2],
        "xG": [0.5, 0.8, 0.6, 0.2, 0.4, 0.3],
    })
    result = compute_ewma_features(df, stats=["xG"])
    assert "ewma_xG" in result.columns
    assert result.loc[1, "ewma_xG"] > 0
    assert result.loc[2, "ewma_xG"] > 0
    assert result.loc[4, "ewma_xG"] > 0
    assert result.loc[5, "ewma_xG"] > 0


def test_solve_squad_selection_basic():
    df = pd.DataFrame({
        "element_id": list(range(1, 16)),
        "position": ["GK"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3,
        "team": ["T1", "T2", "T3", "T4", "T5"] * 3,
        "now_cost": [50.0] * 15,
        "xPts": [5.0] * 15,
    })
    problem = solve_squad_selection(df, horizon_gws=1, budget=100.0)
    solution = solve_problem(problem)
    assert solution is not None
    selected = [int(v.replace("p_", "")) for v in solution]
    assert len(selected) == 15
