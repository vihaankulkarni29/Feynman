from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Dict, List

import aiohttp
import pandas as pd
from loguru import logger

from config.pipeline import load_config
from src.ingestion._common import immutable_write_csv


RAW_SUBDIR = "historical"
SEASON_FILES = ["merged_gw.csv", "players_raw.csv"]


async def _download_file(session: aiohttp.ClientSession, url: str, destination: Path) -> None:
    async with session.get(url) as resp:
        if resp.status == 404:
            logger.warning("File not found: {}", url)
            return
        resp.raise_for_status()
        content = await resp.read()
        destination.parent.mkdir(parents=True, exist_ok=True)
        tmp = destination.with_suffix(".tmp")
        tmp.write_bytes(content)
        tmp.replace(destination)
        logger.info("Downloaded {}", destination)


async def _download_season(session: aiohttp.ClientSession, season: str, raw_dir: Path) -> None:
    base = f"https://raw.githubusercontent.com/vaastav/FPL/{season}/data"
    season_dir = raw_dir / RAW_SUBDIR / season
    tasks = []
    for filename in SEASON_FILES:
        url = f"{base}/{filename}"
        dest = season_dir / filename
        tasks.append(_download_file(session, url, dest))
    await asyncio.gather(*tasks, return_exceptions=True)


async def _run_async(raw_dir: Path) -> None:
    config = load_config().get("vaastav", {})
    seasons = config.get("seasons", [])
    if not seasons:
        logger.info("No Vaastav seasons configured. Skipping.")
        return
    raw_dir = Path(raw_dir)
    async with aiohttp.ClientSession() as session:
        for season in seasons:
            logger.info("Downloading Vaastav data for season {}", season)
            await _download_season(session, season, raw_dir)


def run_vaastav_ingestion(raw_dir: Path) -> None:
    asyncio.run(_run_async(raw_dir))


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    run_vaastav_ingestion(Path("data/raw"))
