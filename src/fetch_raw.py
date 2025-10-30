"""
fetch_raw.py
-------------
Downloads SpaceX launch data and stores a dated snapshot
with SHA256 integrity tracking.
"""

import hashlib, json, os, datetime, requests
from pathlib import Path
from rich.console import Console

console = Console()

DATA_RAW = Path("data/raw")
SNAPSHOT_INDEX = Path("data/_snapshots/index.json")
API_URL = "https://api.spacexdata.com/v4/launches"


def sha256sum(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def save_snapshot():
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_INDEX.parent.mkdir(parents=True, exist_ok=True)

    # Timestamped filename
    now = datetime.datetime.utcnow()
    fname = f"launches_{now:%Y%m%d_%H%M%S}.json"
    fpath = DATA_RAW / fname

    console.print(f"Fetching [cyan]{API_URL}[/cyan] ...")
    resp = requests.get(API_URL, timeout=60)
    resp.raise_for_status()

    data = resp.json()
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    sha = sha256sum(fpath)
    count = len(data)
    entry = {
        "file": str(fpath),
        "rows": count,
        "sha256": sha,
        "timestamp_utc": now.isoformat(timespec="seconds"),
    }

    # Append to index.json
    if SNAPSHOT_INDEX.exists():
        with open(SNAPSHOT_INDEX, "r", encoding="utf-8") as f:
            index = json.load(f)
    else:
        index = []

    index.append(entry)
    with open(SNAPSHOT_INDEX, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

    console.print(
        f"[green]Snapshot saved:[/green] {fpath.name} "
        f"({count} records, sha256={sha[:12]}...)"
    )
    return fpath


if __name__ == "__main__":
    try:
        save_snapshot()
    except Exception as e:
        console.print(f"[red]Fetch failed:[/red] {e}")
        raise SystemExit(1)
