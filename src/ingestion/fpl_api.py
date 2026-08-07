from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib.parse import urljoin

from loguru import logger

from config.pipeline import load_config


class FPLAPIClient:
    """Thin, idempotent wrapper around the public FPL API."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        if config is None:
            config = load_config()["fpl_api"]
        self.base_url = config["base_url"]
        self.timeout = config.get("timeout_seconds", 30)
        self.headers = config.get("headers", {})
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        self.session.mount("https://", HTTPAdapter(max_retries=3))

    def _url(self, endpoint: str) -> str:
        return urljoin(self.base_url, endpoint)

    def _save(self, payload: Any, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        tmp = destination.with_suffix(destination.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        tmp.replace(destination)
        logger.debug("Saved raw file {}", destination)

    def _should_download(self, destination: Path) -> bool:
        if not destination.exists():
            return True
        if destination.stat().st_size == 0:
            return True
        return False

    def fetch_bootstrap_static(self, raw_dir: Path) -> Path:
        endpoint = "bootstrap-static/"
        destination = raw_dir / "bootstrap_static.json"
        if not self._should_download(destination):
            logger.info("Skipping bootstrap-static (already exists)")
            return destination
        logger.info("Fetching bootstrap-static")
        resp = self.session.get(self._url(endpoint), timeout=self.timeout)
        resp.raise_for_status()
        self._save(resp.json(), destination)
        return destination

    def fetch_element_summary(self, element_id: int, raw_dir: Path) -> Path:
        endpoint = f"element-summary/{element_id}/"
        destination = raw_dir / f"element_summary_{element_id}.json"
        if not self._should_download(destination):
            logger.info("Skipping element-summary for element {}", element_id)
            return destination
        logger.info("Fetching element-summary for element {}", element_id)
        resp = self.session.get(self._url(endpoint), timeout=self.timeout)
        resp.raise_for_status()
        self._save(resp.json(), destination)
        time.sleep(0.05)
        return destination

    def fetch_all_element_summaries(self, raw_dir: Path) -> None:
        bootstrap_path = self.fetch_bootstrap_static(raw_dir)
        with bootstrap_path.open("r", encoding="utf-8") as fh:
            bootstrap = json.load(fh)
        elements = bootstrap.get("elements", [])
        logger.info("Fetching summaries for {} players", len(elements))
        for element in elements:
            try:
                self.fetch_element_summary(element["id"], raw_dir)
            except requests.RequestException as exc:
                logger.warning("Failed to fetch element {}: {}", element["id"], exc)


def run_ingestion(raw_dir: Path) -> None:
    config = load_config()
    client = FPLAPIClient(config)
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    client.fetch_bootstrap_static(raw_dir)
    client.fetch_all_element_summaries(raw_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_ingestion(Path("data/raw"))
