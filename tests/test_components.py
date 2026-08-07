from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.preprocessing.cleaning import (
    apply_status_flags,
    build_entity_mapping,
    clean_dataframe,
    fuzzy_match_players,
    impute_nan_by_position,
    load_entity_mapping,
    normalize_costs,
    normalize_dates,
    normalize_volume_metrics,
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


def test_build_entity_mapping(tmp_path: Path):
    bootstrap = {
        "elements": [
            {"id": 1, "first_name": "Mohamed", "second_name": "Salah", "web_name": "Salah", "team": 1, "element_type": 3, "now_cost": 130, "status": "a", "chance_of_playing_this_round": None, "chance_of_playing_next_round": None},
            {"id": 2, "first_name": "Erling", "second_name": "Haaland", "web_name": "Haaland", "team": 2, "element_type": 4, "now_cost": 150, "status": "a", "chance_of_playing_this_round": 100, "chance_of_playing_next_round": 100},
        ],
        "teams": [{"id": 1, "name": "Liverpool"}, {"id": 2, "name": "Man City"}],
        "element_types": [{"id": 1, "singular_name": "Goalkeeper"}, {"id": 2, "singular_name": "Defender"}, {"id": 3, "singular_name": "Midfielder"}, {"id": 4, "singular_name": "Forward"}],
    }
    bs_path = tmp_path / "bootstrap_static.json"
    with bs_path.open("w", encoding="utf-8") as fh:
        json.dump(bootstrap, fh)

    out = build_entity_mapping(bs_path, tmp_path)
    assert out.exists()
    mapping = load_entity_mapping(tmp_path)
    assert "1" in mapping
    assert mapping["1"]["web_name"] == "Salah"
    assert mapping["1"]["cost_normalized"] == pytest.approx(13.0)
    assert mapping["2"]["cost_normalized"] == pytest.approx(15.0)


def test_normalize_costs():
    df = pd.DataFrame({"now_cost": [130, 150, None, "abc"]})
    result = normalize_costs(df)
    assert result.loc[0, "cost_normalized"] == pytest.approx(13.0)
    assert result.loc[1, "cost_normalized"] == pytest.approx(15.0)
    assert pd.isna(result.loc[2, "cost_normalized"])
    assert pd.isna(result.loc[3, "cost_normalized"])


def test_normalize_dates():
    df = pd.DataFrame({"kickoff_time": ["2025-08-15T12:30:00Z", None, "invalid"]})
    result = normalize_dates(df)
    assert isinstance(result.loc[0, "kickoff_time_utc"], pd.Timestamp)
    assert str(result.loc[0, "kickoff_time_utc"].tz) in ("UTC", "datetime.timezone.utc")
    assert pd.isna(result.loc[1, "kickoff_time_utc"])
    assert pd.isna(result.loc[2, "kickoff_time_utc"])


def test_apply_status_flags():
    entity_mapping = {
        "1": {"chance_of_playing_this_round": 0, "chance_of_playing_next_round": None},
        "2": {"chance_of_playing_this_round": 100, "chance_of_playing_next_round": 100},
        "3": {"chance_of_playing_this_round": None, "chance_of_playing_next_round": None},
    }
    df = pd.DataFrame({
        "element_id": [1, 2, 3, 3],
        "minutes": [0, 0, 0, 90],
        "chance_of_playing_this_round": [0, 100, None, None],
        "chance_of_playing_next_round": [None, 100, None, None],
    })
    result = apply_status_flags(df, entity_mapping)
    assert result.loc[0, "player_status"] == "injured"
    assert result.loc[1, "player_status"] == "benched"
    assert result.loc[2, "player_status"] == "dnp"
    assert result.loc[3, "player_status"] == "active"
    assert bool(result.loc[0, "is_injured_dnp"]) is True
    assert bool(result.loc[1, "is_benched"]) is True
    assert bool(result.loc[3, "masked_for_ewma"]) is False


def test_normalize_volume_metrics():
    df = pd.DataFrame({
        "expected_goals": [0.5, 0.3, 0.2],
        "expected_assists": [0.2, 0.1, 0.0],
        "shots": [3, 2, 1],
        "is_benched": [False, True, False],
    })
    result = normalize_volume_metrics(df)
    assert result.loc[0, "expected_goals"] == pytest.approx(0.5)
    assert result.loc[1, "expected_goals"] == 0.0
    assert result.loc[1, "shots"] == 0.0
    assert result.loc[2, "expected_goals"] == pytest.approx(0.2)


def test_clean_dataframe_integration():
    df = pd.DataFrame({
        "name": ["Player A"],
        "team": ["Man City"],
        "position": ["MID"],
        "now_cost": [130],
        "kickoff_time": ["2025-08-15T12:30:00Z"],
        "minutes": [0],
        "expected_goals": [0.5],
        "expected_assists": [0.2],
        "element_id": [1],
    })
    entity_mapping = {
        "1": {
            "web_name": "Player A",
            "team_name": "Manchester City",
            "position": "Midfielder",
            "cost_normalized": 13.0,
            "chance_of_playing_this_round": 100,
            "chance_of_playing_next_round": 100,
        }
    }
    result = clean_dataframe(df, entity_mapping=entity_mapping)
    assert "team_normalized" in result.columns
    assert result.loc[0, "team_normalized"] == "Manchester City"
    assert result.loc[0, "cost_normalized"] == pytest.approx(13.0)
    assert str(result.loc[0, "kickoff_time_utc"].tz) in ("UTC", "datetime.timezone.utc")
    assert bool(result.loc[0, "is_benched"]) is True
    assert result.loc[0, "expected_goals"] == 0.0
