"""
manifest.py
------------
Creates a manifest.json to record schema, columns, and source hashes.
This ensures the feature table is reproducible.
"""

from __future__ import annotations
from pathlib import Path
import hashlib, json, pandas as pd
from rich.console import Console

console = Console()

FEATURE_DIR = Path("data/features")
RAW_DIR = Path("data/raw")
MANIFEST_PATH = Path("data/manifest.json")

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def _latest_file(pattern: str) -> Path:
    import glob
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern}")
    return Path(files[-1])

def build_manifest():
    feature_path = _latest_file(str(FEATURE_DIR / "launch_features_*.parquet"))
    raw_path = _latest_file(str(RAW_DIR / "raw/launches_*.json")) if (RAW_DIR / "raw").exists() else None

    df = pd.read_parquet(feature_path)
    manifest = {
        "feature_file": feature_path.as_posix(),
        "rows": len(df),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "sha256_feature": _sha256(feature_path),
        "sha256_raw": _sha256(raw_path) if raw_path else None,
    }

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    console.print(f"[green]Manifest locked:[/green] {MANIFEST_PATH}")
    console.print(json.dumps(manifest, indent=2))

if __name__ == "__main__":
    build_manifest()
