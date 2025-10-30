"""
validate_raw.py
---------------
Validates the latest raw SpaceX launches snapshot for shape, fields,
and basic types using pandas + pandera. Writes a log to logs/validate_raw.log
and exits nonzero on failure.
"""
from __future__ import annotations

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import pandera as pa
from pandera import Column, Check
from rich.console import Console

from src.utils import latest_snapshot_entry

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="pandera")


console = Console()
LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "validate_raw.log"
REQUIRED_FIELDS = [
    "date_utc",
    "rocket",
    "success",
    "payloads",
    "cores",
    "flight_number",
    "launchpad",
]

def setup_logging():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=str(LOG_FILE),
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

def load_raw_dataframe(path: Path) -> pd.DataFrame:
    # JSON file is a list of dicts
    df = pd.read_json(path, orient="records")
    return df

def build_schema() -> pa.DataFrameSchema:
    return pa.DataFrameSchema(
        columns={
            "date_utc": Column(pa.String, nullable=False),
            "rocket": Column(pa.String, nullable=False),
            "success": Column(pa.Object, nullable=True),  # relaxed
            "payloads": Column(pa.Object, nullable=True),
            "cores": Column(pa.Object, nullable=True),
            "flight_number": Column(pa.Int, nullable=False, checks=Check.ge(1)),
            "launchpad": Column(pa.String, nullable=False),
        },
        coerce=True,    # let Pandera cast dtypes for us
        strict=False,
    )



def run_validation() -> None:
    setup_logging()
    snap = latest_snapshot_entry()
    raw_path = Path(snap["file"])
    console.print(f"Validating latest snapshot: [cyan]{raw_path}[/cyan]")

    if not raw_path.exists():
        msg = f"Snapshot file missing: {raw_path}"
        logging.error(msg)
        console.print(f"[red]{msg}[/red]")
        raise SystemExit(1)

    df = load_raw_dataframe(raw_path)
    logging.info(f"Loaded {len(df)} rows from {raw_path}")

    # --- Basic shape sanity ---
    if len(df) < 100:
        msg = f"Too few rows: {len(df)} (expected >= 100)"
        logging.error(msg)
        console.print(f"[red]{msg}[/red]")
        raise SystemExit(1)

    # --- Required fields present ---
    missing = [c for c in REQUIRED_FIELDS if c not in df.columns]
    if missing:
        msg = f"Missing required fields: {missing}"
        logging.error(msg)
        console.print(f"[red]{msg}[/red]")
        raise SystemExit(1)

    # --- Pandera column-level checks ---
    # Force correct dtype for success: pandas 'boolean' supports NA without upcasting to float64
    if "success" in df.columns:
        df["success"] = df["success"].apply(
        lambda x: True if x is True else False if x is False else None
    )
    schema = build_schema()
    try:
        schema.validate(df[REQUIRED_FIELDS], lazy=True)
    except pa.errors.SchemaErrors as err:
        # Write details to log
        logging.error("Pandera validation failed:\n%s", err.failure_cases)
        console.print("[red]Pandera validation failed; see logs/validate_raw.log[/red]")
        raise SystemExit(1)

    # --- Extra semantic checks ---
    # date_utc parseable for most rows
    parsed = pd.to_datetime(df["date_utc"], errors="coerce", utc=True)
    ok_ratio = parsed.notna().mean()
    if ok_ratio < 0.95:
        msg = f"Too many unparseable date_utc values: only {ok_ratio:.1%} parseable"
        logging.error(msg)
        console.print(f"[red]{msg}[/red]")
        raise SystemExit(1)

    # success must be True/False/None only (allow NA)
    if "success" in df.columns:
        allowed = {True, False, None}
        bad_mask = ~df["success"].isin(allowed)
        # if dtype is nullable boolean, NA shows up as <NA>, treat as None
        bad_mask &= ~df["success"].isna()
        if bad_mask.any():
            msg = f"Invalid values in 'success' at rows: {df.index[bad_mask].tolist()[:10]}..."
            logging.error(msg)
            console.print(f"[red]{msg}[/red]")
            raise SystemExit(1)

    # payloads and cores should be list-like when present
    def is_listlike(x: Any) -> bool:
        return isinstance(x, list) or pd.isna(x)

    bad_payloads = ~df["payloads"].map(is_listlike)
    bad_cores = ~df["cores"].map(is_listlike)
    if bad_payloads.any() or bad_cores.any():
        msg = (
            f"Non-list values: payloads_bad={int(bad_payloads.sum())}, "
            f"cores_bad={int(bad_cores.sum())}"
        )
        logging.error(msg)
        console.print(f"[red]{msg}[/red]")
        raise SystemExit(1)

    logging.info("RAW_VALIDATION_OK")
    console.print("[green]RAW_VALIDATION_OK[/green]")

if __name__ == "__main__":
    try:
        run_validation()
    except SystemExit as e:
        raise
    except Exception as e:
        setup_logging()
        logging.exception("Unexpected failure: %s", e)
        console.print(f"[red]Unexpected error[/red]: {e}")
        raise SystemExit(1)
