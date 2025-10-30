"""
build_features.py
-----------------
Selects and orders canonical feature columns from the processed dataset.
Converts labels to 0/1, drops unlabeled rows,
and saves to data/features/ + preview CSV in reports/.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import glob
import pandas as pd
from rich.console import Console

console = Console()

PROCESSED_DIR = Path("data/processed")
FEATURE_DIR   = Path("data/features")
REPORTS_DIR   = Path("reports")

FEATURE_COLUMNS = [
    "flight_number",     # Int
    "year",              # Int
    "payload_mass_kg",   # Float
    "orbit",             # Str
    "launch_site",       # Str
    "booster_version",   # Str
    "reuse_count",       # Int
    "is_weekend",        # Int (0/1)
    "success",           # Int label
]


def _latest_processed_path() -> Path:
    files = sorted(glob.glob(str(PROCESSED_DIR / "launches_processed_*.feather")))
    if not files:
        raise FileNotFoundError("No processed files found. Run clean_transform first.")
    return Path(files[-1])


def build_features() -> Path:
    ppath = _latest_processed_path()
    console.print(f"Building features from [cyan]{ppath}[/cyan]")
    df = pd.read_feather(ppath)

    # Convert success to 0/1 Int64, drop NA
    df["success"] = (
        df["success"]
        .map(lambda x: 1 if x is True else (0 if x is False else pd.NA))
        .astype("Int64")
    )
    before = len(df)
    df = df.dropna(subset=["success"])
    dropped = before - len(df)

    # Ensure all expected columns exist
    for c in FEATURE_COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA

    feats = df[FEATURE_COLUMNS].copy()

    # Enforce numeric types
    for c in ["flight_number", "year", "reuse_count", "is_weekend", "success"]:
        feats[c] = pd.to_numeric(feats[c], errors="coerce").astype("Int64")

    # Save outputs
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    feature_path = FEATURE_DIR / f"launch_features_{stamp}.parquet"
    preview_path = REPORTS_DIR / f"feature_head_{stamp}.csv"

    feats.to_parquet(feature_path, index=False)
    feats.head(10).to_csv(preview_path, index=False)

    console.print(f"[green]Features saved:[/green] {feature_path.name} (rows={len(feats)}, dropped_missing_label={dropped})")
    console.print(f"[green]Preview saved:[/green]  {preview_path.name}")
    return feature_path


if __name__ == "__main__":
    try:
        build_features()
    except Exception as e:
        console.print(f"[red]Feature build failed:[/red] {e}")
        raise SystemExit(1)
