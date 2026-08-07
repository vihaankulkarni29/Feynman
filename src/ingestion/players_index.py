from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd

from src.ingestion._common import immutable_write_csv


def extract_players_index(raw_dir: Path) -> Path:
    bootstrap_path = raw_dir / "bootstrap_static.json"
    if not bootstrap_path.exists():
        raise FileNotFoundError(f"bootstrap_static.json not found at {bootstrap_path}")

    import json
    with bootstrap_path.open("r", encoding="utf-8") as fh:
        bootstrap = json.load(fh)

    elements = bootstrap.get("elements", [])
    teams = {t["id"]: t["name"] for t in bootstrap.get("teams", [])}
    positions = {t["id"]: t["singular_name"] for t in bootstrap.get("element_types", [])}

    rows = []
    for p in elements:
        rows.append({
            "element_id": p.get("id"),
            "first_name": p.get("first_name", ""),
            "second_name": p.get("second_name", ""),
            "web_name": p.get("web_name", ""),
            "team_id": p.get("team"),
            "team_name": teams.get(p.get("team"), ""),
            "position_id": p.get("element_type"),
            "position": positions.get(p.get("element_type"), ""),
            "now_cost": p.get("now_cost"),
            "status": p.get("status", ""),
        })

    df = pd.DataFrame(rows)
    return immutable_write_csv(df, raw_dir, "players_index")
