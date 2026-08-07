from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from bs4 import BeautifulSoup

from loguru import logger

from config.pipeline import load_config


def _headers() -> Dict[str, str]:
    return {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/123.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }


def fetch_page(url: str) -> Optional[str]:
    try:
        resp = requests.get(url, headers=_headers(), timeout=30)
        resp.raise_for_status()
        return resp.text
    except requests.RequestException as exc:
        logger.warning("FBref fetch failed for {}: {}", url, exc)
        return None


def parse_match_logs_table(html: str) -> List[Dict[str, Any]]:
    soup = BeautifulSoup(html, "html.parser")
    table = soup.select_one("#stats_standard_9")
    if table is None:
        logger.warning("Match logs table not found")
        return []
    headers = [
        th.get_text(strip=True) for th in table.find_all("th") if th.get("aria-label")
    ]
    rows: List[Dict[str, Any]] = []
    for row in table.find_all("tr")[1:]:
        cells = [td.get_text(strip=True) for td in row.find_all("td")]
        if not cells:
            continue
        row_data: Dict[str, Any] = {headers[i]: cells[i] for i in range(min(len(headers), len(cells)))}
        rows.append(row_data)
    return rows


def save_csv(rows: List[Dict[str, Any]], destination: Path) -> None:
    if not rows:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Saved FBref CSV: {}", destination)


def sync_fbref(raw_dir: Path) -> None:
    config = load_config().get("fbref", {})
    seasons = config.get("seasons", ["2024-2025"])
    base_url = config.get("base_url", "https://fbref.com")
    raw_dir = Path(raw_dir)

    urls = [
        f"{base_url}/en/comps/9/Premier-League-Stats/{season}"
        for season in seasons
    ]
    for url in urls:
        html = fetch_page(url)
        if not html:
            continue
        rows = parse_match_logs_table(html)
        season_slug = url.split("/")[-1].replace(" ", "_")
        destination = raw_dir / f"fbref_match_logs_{season_slug}.csv"
        save_csv(rows, destination)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sync_fbref(Path("data/raw"))
