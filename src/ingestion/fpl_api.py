from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp
import pandas as pd
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from config.pipeline import load_config
from src.ingestion._common import immutable_write_json, immutable_write_csv


BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
FIXTURES_URL = "https://fantasy.premierleague.com/api/fixtures/"
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0.0.0 Safari/537.36"
)


class ElementSummary(BaseModel):
    fixtures: List[Dict[str, Any]] = Field(default_factory=list)
    history: List[Dict[str, Any]] = Field(default_factory=list)
    history_past: List[Dict[str, Any]] = Field(default_factory=list)


class Fixture(BaseModel):
    id: int
    event: Optional[int] = None
    team_a: Optional[int] = None
    team_h: Optional[int] = None
    team_a_difficulty: Optional[int] = None
    team_h_difficulty: Optional[int] = None


class BootstrapStatic(BaseModel):
    chips: List[Dict[str, Any]] = Field(default_factory=list)
    events: List[Dict[str, Any]] = Field(default_factory=list)
    game_settings: Dict[str, Any] = Field(default_factory=dict)
    game_config: Dict[str, Any] = Field(default_factory=dict)
    phases: List[Dict[str, Any]] = Field(default_factory=list)
    teams: List[Dict[str, Any]]
    total_players: int = 0
    element_stats: List[Dict[str, Any]] = Field(default_factory=list)
    element_types: List[Dict[str, Any]] = Field(default_factory=list)
    elements: List[Dict[str, Any]]


class AsyncFPLClient:
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        if config is None:
            config = load_config()["fpl_api"]
        self.base_url = config["base_url"]
        self.timeout = aiohttp.ClientTimeout(total=config.get("timeout_seconds", 30))
        self.semaphore = asyncio.Semaphore(config.get("max_concurrency", 30))
        self.headers = {"User-Agent": UA}

    def _url(self, endpoint: str) -> str:
        return f"{self.base_url}{endpoint}"

    async def _fetch_json(self, session: aiohttp.ClientSession, url: str) -> Any:
        async with self.semaphore:
            async with session.get(url, headers=self.headers, timeout=self.timeout) as resp:
                resp.raise_for_status()
                return await resp.json()

    async def fetch_bootstrap_static(self, session: aiohttp.ClientSession, raw_dir: Path) -> BootstrapStatic:
        logger.info("Fetching bootstrap-static")
        payload = await self._fetch_json(session, self._url("bootstrap-static/"))
        try:
            model = BootstrapStatic(**payload)
        except ValidationError as exc:
            logger.error("Bootstrap-static validation failed: {}", exc)
            raise
        immutable_write_json(payload, raw_dir, "bootstrap_static")
        return model

    async def fetch_element_summary(
        self, session: aiohttp.ClientSession, element_id: int, raw_dir: Path
    ) -> ElementSummary:
        url = self._url(f"element-summary/{element_id}/")
        payload = await self._fetch_json(session, url)
        try:
            model = ElementSummary(**payload)
        except ValidationError as exc:
            logger.warning("Element-summary validation failed for {}: {}", element_id, exc)
            return ElementSummary()
        immutable_write_json(payload, raw_dir, f"player_{element_id}_summary")
        return model

    async def fetch_all_element_summaries(self, session: aiohttp.ClientSession, raw_dir: Path, bootstrap: BootstrapStatic) -> None:
        elements = bootstrap.elements
        logger.info("Fanning out element-summary for {} players", len(elements))
        tasks = [self.fetch_element_summary(session, int(e["id"]), raw_dir) for e in elements]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        failures = sum(1 for r in results if isinstance(r, Exception))
        logger.info("Element-summary complete. Failures: {}", failures)

    async def fetch_fixtures(self, session: aiohttp.ClientSession, raw_dir: Path) -> List[Fixture]:
        logger.info("Fetching fixtures")
        payload = await self._fetch_json(session, self._url("fixtures/"))
        try:
            fixtures = [Fixture(**f) for f in payload]
        except ValidationError as exc:
            logger.error("Fixtures validation failed: {}", exc)
            raise
        immutable_write_json(payload, raw_dir, "fixtures")
        return fixtures


async def _run_async(raw_dir: Path) -> None:
    config = load_config()["fpl_api"]
    client = AsyncFPLClient(config)
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    async with aiohttp.ClientSession() as session:
        bootstrap = await client.fetch_bootstrap_static(session, raw_dir)
        await client.fetch_all_element_summaries(session, raw_dir, bootstrap)
        await client.fetch_fixtures(session, raw_dir)


def run_fpl_ingestion(raw_dir: Path) -> None:
    asyncio.run(_run_async(raw_dir))


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    run_fpl_ingestion(Path("data/raw"))
