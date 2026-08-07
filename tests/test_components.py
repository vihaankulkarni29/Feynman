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
    compute_ewma_features_masked,
    compute_rotation_risk,
    compute_tactical_multipliers,
    compute_positional_xpts,
    drop_non_predictive,
    generate_features,
    load_tactical_cache,
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


def test_compute_ewma_features_masked():
    df = pd.DataFrame({
        "element_id": [1, 1, 1, 1, 2, 2, 2],
        "gameweek": [1, 2, 3, 4, 1, 2, 3],
        "expected_goals": [0.5, 0.8, 0.6, 0.4, 0.2, 0.4, 0.3],
        "masked_for_ewma": [False, False, True, False, False, True, False],
    })
    result = compute_ewma_features_masked(df, metrics=["expected_goals"])
    assert "ewma_expected_goals_span3" in result.columns
    assert result.loc[3, "ewma_expected_goals_span3"] > 0
    assert pd.notna(result.loc[3, "ewma_expected_goals_span3"])
    assert pd.notna(result.loc[2, "ewma_expected_goals_span3"])


def test_compute_rotation_risk():
    df = pd.DataFrame({
        "element_id": [1, 1, 1, 1, 1, 1],
        "gameweek": [1, 2, 3, 4, 5, 6],
        "starts": [1, 1, 0, 1, 1, 0],
        "appearances": [1, 1, 1, 1, 1, 1],
    })
    result = compute_rotation_risk(df)
    assert "rotation_risk" in result.columns
    assert result.loc[5, "rotation_risk"] == pytest.approx(4.0 / 6.0)


def test_compute_rotation_risk_insufficient_appearances():
    df = pd.DataFrame({
        "element_id": [1, 1],
        "gameweek": [1, 2],
        "starts": [0, 0],
        "appearances": [1, 1],
    })
    result = compute_rotation_risk(df)
    assert result.loc[1, "rotation_risk"] == 0.0


def test_compute_tactical_multipliers():
    tactical = pd.DataFrame({
        "season": ["2022-2023", "2022-2023"],
        "team": ["Arsenal", "Liverpool"],
        "PPDA": [8.5, 7.2],
        "PPDA_allowed": [12.3, 10.9],
        "defensive_third_field_tilt": [55.2, 60.5],
        "final_third_tackles": [18.4, 22.1],
        "xG": [68.5, 72.8],
        "xGA": [42.1, 35.6],
        "shots": [520, 550],
        "shots_allowed": [380, 320],
    })
    df = pd.DataFrame({
        "element_id": [1, 2],
        "team_normalized": ["Arsenal", "Liverpool"],
        "position": ["DEF", "MID"],
    })
    result = compute_tactical_multipliers(df, tactical)
    assert "tactical_multiplier" in result.columns
    assert result.loc[0, "tactical_multiplier"] > 0
    assert result.loc[1, "tactical_multiplier"] > 0


def test_compute_positional_xpts():
    df = pd.DataFrame({
        "element_id": [1, 2, 3],
        "position": ["GK", "MID", "FWD"],
        "ewma_expected_goals_span5": [0.1, 0.3, 1.0],
        "ewma_expected_assists_span5": [0.05, 0.2, 0.3],
        "ewma_clearances_blocks_interceptions_span5": [12.0, 8.0, 5.0],
        "ewma_clean_sheets_span5": [0.6, 0.4, 0.1],
        "ewma_total_points_span5": [4.0, 5.0, 6.0],
    })
    result = compute_positional_xpts(df)
    assert "target_xPts" in result.columns
    assert result.loc[0, "target_xPts"] > 0
    assert result.loc[2, "target_xPts"] > result.loc[0, "target_xPts"]


def test_drop_non_predictive():
    df = pd.DataFrame({
        "element_id": [1, 2],
        "name": ["A", "B"],
        "value": [10, 20],
    })
    result = drop_non_predictive(df)
    assert "element_id" in result.columns
    assert "name" not in result.columns
    assert "value" in result.columns


def test_generate_features_outputs_parquet(tmp_path: Path):
    df = pd.DataFrame({
        "element_id": [1, 1, 2, 2],
        "gameweek": [1, 2, 1, 2],
        "position": ["MID", "MID", "FWD", "FWD"],
        "team_normalized": ["Arsenal", "Arsenal", "Liverpool", "Liverpool"],
        "expected_goals": [0.5, 0.8, 0.3, 0.6],
        "expected_assists": [0.2, 0.3, 0.1, 0.2],
        "clearances_blocks_interceptions": [5.0, 8.0, 2.0, 3.0],
        "clean_sheets": [1, 0, 0, 0],
        "total_points": [4.0, 6.0, 5.0, 7.0],
        "minutes": [90, 90, 60, 90],
        "starts": [1, 1, 1, 1],
        "appearances": [1, 1, 1, 1],
        "masked_for_ewma": [False, False, False, False],
    })
    tactical = pd.DataFrame({
        "season": ["2022-2023", "2022-2023"],
        "team": ["Arsenal", "Liverpool"],
        "PPDA": [8.5, 7.2],
        "PPDA_allowed": [12.3, 10.9],
        "defensive_third_field_tilt": [55.2, 60.5],
        "final_third_tackles": [18.4, 22.1],
        "xG": [68.5, 72.8],
        "xGA": [42.1, 35.6],
        "shots": [520, 550],
        "shots_allowed": [380, 320],
    })
    result = generate_features(df, tmp_path, tactical_df=tactical)
    assert "target_xPts" in result.columns
    assert "tactical_multiplier" in result.columns
    assert "rotation_risk" in result.columns
    assert "ewma_expected_goals_span3" in result.columns
    out = tmp_path / "model_input.parquet"
    assert out.exists()


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
