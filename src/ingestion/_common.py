from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def immutable_write_json(payload: Any, raw_dir: Path, basename: str) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    timestamp = _utc_timestamp()
    versioned = raw_dir / f"{basename}.{timestamp}.json"
    current = raw_dir / f"{basename}.json"

    tmp = versioned.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        import json
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    tmp.replace(versioned)

    tmp2 = current.with_suffix(".tmp")
    with tmp2.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    tmp2.replace(current)

    return current


def immutable_write_csv(df: pd.DataFrame, raw_dir: Path, basename: str) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    timestamp = _utc_timestamp()
    versioned = raw_dir / f"{basename}.{timestamp}.csv"
    current = raw_dir / f"{basename}.csv"

    tmp = versioned.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(versioned)

    tmp2 = current.with_suffix(".tmp")
    df.to_csv(tmp2, index=False)
    tmp2.replace(current)

    return current
