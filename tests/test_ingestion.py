from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.ingestion._common import immutable_write_csv, immutable_write_json
from src.ingestion.fpl_api import BootstrapStatic, ElementSummary, Fixture


def test_pydantic_bootstrap_static_validation():
    payload = {
        "elements": [{"id": 1, "first_name": "Mohamed", "second_name": "Salah"}],
        "teams": [{"id": 1, "name": "Liverpool"}],
        "gameweeks": [{"id": 1, "name": "GW1"}],
    }
    model = BootstrapStatic(**payload)
    assert len(model.elements) == 1
    assert model.teams[0]["name"] == "Liverpool"


def test_pydantic_element_summary_defaults():
    model = ElementSummary()
    assert model.fixtures == []
    assert model.history == []
    assert model.history_past == []


def test_pydantic_fixture_optional_fields():
    f = Fixture(id=1, event=1, team_a=2, team_h=1, team_a_difficulty=3, team_h_difficulty=2)
    assert f.team_a_difficulty == 3


def test_immutable_write_json_creates_files(tmp_path: Path):
    payload = {"hello": "world"}
    result = immutable_write_json(payload, tmp_path, "test")
    assert result.exists()
    assert result.name == "test.json"
    versioned = list(tmp_path.glob("test.*.json"))
    assert len(versioned) == 1
    assert versioned[0].name != "test.json"


def test_immutable_write_csv_creates_files(tmp_path: Path):
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    result = immutable_write_csv(df, tmp_path, "test_csv")
    assert result.exists()
    assert result.name == "test_csv.csv"
    versioned = list(tmp_path.glob("test_csv.*.csv"))
    assert len(versioned) == 1
