from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List
import requests
from datetime import datetime

SNAPSHOT_INDEX = Path("data/_snapshots/index.json")

def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def latest_snapshot_entry() -> Dict[str, Any]:
    """Return the last entry from data/_snapshots/index.json."""
    if not SNAPSHOT_INDEX.exists():
        raise FileNotFoundError("Snapshot index not found. Run fetch_raw.py first.")
    index = read_json(SNAPSHOT_INDEX)
    if not isinstance(index, list) or len(index) == 0:
        raise RuntimeError("Snapshot index is empty.")
    return index[-1]  # last appended is most recent


CACHE_DIR = Path("data/raw/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def _cache_path(name: str) -> Path:
    return CACHE_DIR / f"{name}.json"

def fetch_with_cache(url: str, name: str, force: bool = False):
    """
    Download a small reference table with simple JSON and cache it to data/raw/cache.
    Returns parsed JSON (list of dicts).
    """
    path = _cache_path(name)
    if path.exists() and not force:
        return read_json(path)
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return data

def safe_get(d: dict, *keys, default=None):
    """Traverse nested dicts safely: safe_get(x, 'a','b') == x.get('a',{}).get('b')."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, default)
    return cur