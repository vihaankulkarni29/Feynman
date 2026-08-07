from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp
import pandas as pd
from bs4 import BeautifulSoup
from loguru import logger
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

from config.pipeline import load_config
from src.ingestion._common import immutable_write_csv


FBREF_BASE = "https://fbref.com"
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0.0.0 Safari/537.36"
)


def _headers() -> Dict[str, str]:
    return {
        "User-Agent": UA,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
    }


def _season_url(season: str) -> str:
    return f"{FBREF_BASE}/en/comps/9/Premier-League-Stats/{season}"


def _safe_text(cell) -> str:
    return cell.get_text(strip=True) if cell else ""


def parse_table_to_df(table) -> pd.DataFrame:
    headers = []
    header_rows = table.find_all("tr")
    for row in header_rows:
        ths = row.find_all("th")
        if ths:
            for th in ths:
                aria = th.get("aria-label")
                text = th.get_text(strip=True)
                headers.append(aria or text)
    if not headers:
        return pd.DataFrame()

    rows = []
    for row in table.find_all("tr")[1:]:
        cells = row.find_all(["td", "th"])
        if not cells:
            continue
        row_data = {headers[i]: _safe_text(cells[i]) for i in range(min(len(headers), len(cells)))}
        rows.append(row_data)
    return pd.DataFrame(rows)


async def fetch_page(session: aiohttp.ClientSession, url: str) -> Optional[str]:
    try:
        async with session.get(url, headers=_headers(), timeout=30) as resp:
            if resp.status == 403:
                logger.warning("FBref blocked request (403). Cloudflare challenge may require Selenium. URL: {}", url)
                return None
            if resp.status != 200:
                logger.warning("FBref fetch failed for {}: status {}", url, resp.status)
                return None
            text = await resp.text()
            if "Just a moment..." in text or "cloudflare" in text.lower():
                logger.warning("FBref returned Cloudflare challenge page for {}", url)
                return None
            return text
    except Exception as exc:
        logger.warning("FBref fetch exception for {}: {}", url, exc)
        return None


def fetch_page_with_selenium(url: str, wait_time: Optional[int] = None) -> Optional[str]:
    config = load_config().get("tactical", {}).get("selenium", {})
    if wait_time is None:
        wait_time = config.get("wait_time_seconds", 15)

    chrome_options = Options()
    if config.get("headless", True):
        chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option("useAutomationExtension", False)
    chrome_options.add_argument(f"user-agent={UA}")
    chrome_options.add_argument("--window-size=1920,1080")

    driver = None
    try:
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        driver.set_page_load_timeout(30)
        driver.get(url)

        WebDriverWait(driver, wait_time).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "table"))
        )
        html = driver.page_source
        if "Just a moment..." in html or "cloudflare" in html.lower():
            logger.warning(
                "FBref Selenium fallback still hit Cloudflare for {}. "
                "Cloudflare clearance may require manual browser interaction or proxy.",
                url,
            )
            return None
        logger.info("FBref Selenium fallback succeeded for {}", url)
        return html
    except Exception as exc:
        logger.warning("FBref Selenium fallback failed for {}: {}", url, exc)
        return None
    finally:
        if driver:
            try:
                driver.quit()
            except Exception:
                pass


async def fetch_squad_tactical_table(season: str, table_id: str, session: aiohttp.ClientSession) -> pd.DataFrame:
    url = _season_url(season)
    html = await fetch_page(session, url)
    if not html:
        logger.info("Falling back to Selenium for {} season={}", table_id, season)
        html = await asyncio.to_thread(fetch_page_with_selenium, url)
    if not html:
        return pd.DataFrame()

    soup = BeautifulSoup(html, "html.parser")
    table = soup.select_one(f"#{table_id}")
    if table is None:
        logger.warning("FBref table #{} not found for season {}", table_id, season)
        return pd.DataFrame()

    df = parse_table_to_df(table)
    df["season"] = season
    df["source_table"] = table_id
    logger.info("Extracted {} rows from #{} for season {}", len(df), table_id, season)
    return df


async def _run_async(raw_dir: Path) -> None:
    config = load_config().get("tactical", {})
    seasons = config.get("seasons", [])
    tables = config.get("fbref_tables", [])

    if not seasons:
        logger.info("No tactical seasons configured. Skipping FBref sync.")
        return
    if not tables:
        logger.warning("No fbref_tables configured. Skipping FBref sync.")
        return

    raw_dir = Path(raw_dir)
    all_frames: List[pd.DataFrame] = []

    async with aiohttp.ClientSession() as session:
        for season in seasons:
            for table_id in tables:
                df = await fetch_squad_tactical_table(season, table_id, session)
                if not df.empty:
                    all_frames.append(df)

    if not all_frames:
        logger.warning(
            "No FBref tactical data fetched. FBref is protected by Cloudflare and may block automated requests. "
            "Options: (1) Run this script interactively with a real browser profile, "
            "(2) Manually download tables from fbref.com and place in data/raw/historical/, "
            "(3) Use a proxy service with valid Cloudflare clearance."
        )
        df = pd.DataFrame()
    else:
        df = pd.concat(all_frames, ignore_index=True)

    immutable_write_csv(df, raw_dir, config.get("raw_filename", "tactical_metrics").replace(".csv", ""))


def run_tactical_ingestion(raw_dir: Path) -> None:
    asyncio.run(_run_async(raw_dir))


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    run_tactical_ingestion(Path("data/raw"))
