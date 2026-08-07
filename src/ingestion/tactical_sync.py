from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp
import pandas as pd
from bs4 import BeautifulSoup
from loguru import logger

from config.pipeline import load_config
from src.ingestion._common import immutable_write_csv


UNDERSTAT_EPL = "https://understat.com/league/EPL/"
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0.0.0 Safari/537.36"
)


def _extract_understat_json(html: str, var_name: str) -> Any:
    match = re.search(rf"var\s+{var_name}\s*=\s*JSON\.parse\(['\"](.*?)['\"]\)", html)
    if not match:
        return None
    import json
    import html as ihtml
    raw = ihtml.unescape(match.group(1))
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


async def fetch_understat_players(season: str, session: aiohttp.ClientSession) -> List[Dict[str, Any]]:
    url = f"{UNDERSTAT_EPL}{season}"
    headers = {"User-Agent": UA}
    async with session.get(url, headers=headers) as resp:
        if resp.status != 200:
            logger.warning("Understat fetch failed for season {}: status {}", season, resp.status)
            return []
        html = await resp.text()

    soup = BeautifulSoup(html, "html.parser")
    script = soup.find("script", text=re.compile("playersData"))
    if not script:
        logger.warning("Understat playersData script not found for season {}", season)
        return []

    text = script.string
    match = re.search(r"var\s+playersData\s*=\s*JSON\.parse\(['\"](.*?)['\"]\)", text)
    if not match:
        logger.warning("Understat playersData regex failed for season {}", season)
        return []

    import json
    import html as ihtml
    raw = ihtml.unescape(match.group(1))
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("Understat playersData JSON parse failed: {}", exc)
        return []


async def fetch_fotmob_league(league_id: int = 47, season: str = "2024-2025", session: aiohttp.ClientSession = None) -> List[Dict[str, Any]]:
    url = f"https://www.fotmob.com/api/leagues?id={league_id}&season={season}"
    headers = {
        "User-Agent": UA,
        "Accept": "application/json, text/plain, */*",
    }
    async with session.get(url, headers=headers) as resp:
        if resp.status != 200:
            logger.warning("FotMob fetch failed: status {}", resp.status)
            return []
        try:
            payload = await resp.json()
            return payload.get("players", [])
        except Exception as exc:
            logger.warning("FotMob JSON parse failed: {}", exc)
            return []


def normalize_understat_to_tactical(players: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for p in players:
        rows.append({
            "player": p.get("player_name", ""),
            "team": p.get("team_title", ""),
            "xG": float(p.get("xG", 0.0) or 0.0),
            "xA": float(p.get("xA", 0.0) or 0.0),
            "shot_creation_volume": float(p.get("shots", 0.0) or 0.0),
            "PPDA": None,
            "defensive_third_field_tilt": None,
            "final_third_tackles": None,
            "source": "understat",
        })
    return pd.DataFrame(rows)


def normalize_fotmob_to_tactical(players: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for p in players:
        rows.append({
            "player": p.get("name", ""),
            "team": p.get("teamName", ""),
            "xG": float(p.get("xG", 0.0) or 0.0),
            "xA": float(p.get("xA", 0.0) or 0.0),
            "shot_creation_volume": float(p.get("shots", 0.0) or 0.0),
            "PPDA": None,
            "defensive_third_field_tilt": None,
            "final_third_tackles": None,
            "source": "fotmob",
        })
    return pd.DataFrame(rows)


async def _run_async(raw_dir: Path) -> None:
    config = load_config().get("tactical", {})
    seasons = [s for s in config.get("seasons", []) if "-" in s]
    season = seasons[0] if seasons else "2024-2025"
    raw_dir = Path(raw_dir)
    delay = config.get("request_delay_seconds", 1.0)

    all_frames: List[pd.DataFrame] = []

    async with aiohttp.ClientSession() as session:
        understat_players = await fetch_understat_players(season, session)
        if understat_players:
            all_frames.append(normalize_understat_to_tactical(understat_players))
        await asyncio.sleep(delay)

        fotmob_players = await fetch_fotmob_league(season=season.split("-")[0], session=session)
        if fotmob_players:
            all_frames.append(normalize_fotmob_to_tactical(fotmob_players))

    if not all_frames:
        logger.warning("No tactical data fetched. Writing empty DataFrame.")
        df = pd.DataFrame(columns=[
            "player", "team", "xG", "xA", "shot_creation_volume",
            "PPDA", "defensive_third_field_tilt", "final_third_tackles", "source"
        ])
    else:
        df = pd.concat(all_frames, ignore_index=True)

    immutable_write_csv(df, raw_dir, config.get("raw_filename", "tactical_metrics").replace(".csv", ""))


def run_tactical_ingestion(raw_dir: Path) -> None:
    asyncio.run(_run_async(raw_dir))


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    run_tactical_ingestion(Path("data/raw"))
