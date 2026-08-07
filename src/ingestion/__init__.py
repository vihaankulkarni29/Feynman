from __future__ import annotations

from pathlib import Path

from src.ingestion.fpl_api import run_fpl_ingestion
from src.ingestion.vaastav_sync import run_vaastav_ingestion
from src.ingestion.tactical_sync import run_tactical_ingestion


async def run_ingestion(raw_dir: Path) -> None:
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    run_fpl_ingestion(raw_dir)
    run_vaastav_ingestion(raw_dir)
    run_tactical_ingestion(raw_dir)
