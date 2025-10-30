"""
clean_transform.py
------------------
Builds a flattened, normalized processed table from the latest raw snapshot.
- Caches lookup tables (launchpads, rockets, payloads)
- Flattens nested structures
- Normalizes categories and types
- Writes data/processed/launches_processed_YYYYMMDD.feather
- Writes reports/processed_YYYYMMDD.txt
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import math

import pandas as pd
from rich.console import Console

from src.utils import latest_snapshot_entry, read_json, fetch_with_cache, safe_get

console = Console()

# API endpoints for lookups
LAUNCHPADS_URL = "https://api.spacexdata.com/v4/launchpads"
ROCKETS_URL    = "https://api.spacexdata.com/v4/rockets"
PAYLOADS_URL   = "https://api.spacexdata.com/v4/payloads"

PROCESSED_DIR = Path("data/processed")
REPORTS_DIR   = Path("reports")

# --- small helpers -----------------------------------------------------------

def _dt_parts(date_utc: str) -> Tuple[int, int]:
    """Return (year, is_weekend) from ISO UTC string."""
    ts = pd.to_datetime(date_utc, errors="coerce", utc=True)
    if pd.isna(ts):
        return (pd.NA, pd.NA)
    # weekend: Saturday(5) or Sunday(6)
    is_weekend = int(ts.weekday() >= 5)
    return (int(ts.year), is_weekend)

def _simplify_orbit(raw: Optional[str]) -> str:
    """Map raw orbit strings to a small vocabulary."""
    if not raw or not isinstance(raw, str):
        return "Unknown"
    s = raw.upper()
    if any(k in s for k in ["LEO", "VLEO"]): return "LEO"
    if "MEO" in s: return "MEO"
    if "GTO" in s: return "GTO"
    if "GEO" in s: return "GEO"
    if "SSO" in s or "PSO" in s: return "SSO"
    if "HEO" in s or "HCO" in s: return "HEO"
    return raw.upper()

def _sum_payload_mass_and_orbit(payload_ids: List[str], payload_map: Dict[str, Dict[str, Any]]) -> Tuple[float, str]:
    """Sum payload mass_kg and choose an orbit label."""
    total_mass = 0.0
    chosen_orbit = "Unknown"
    for pid in (payload_ids or []):
        info = payload_map.get(pid, {})
        mass = info.get("mass_kg")
        if isinstance(mass, (int, float)) and not math.isnan(mass):
            total_mass += float(mass)
        orbit = _simplify_orbit(info.get("orbit"))
        if chosen_orbit == "Unknown" and orbit:
            chosen_orbit = orbit
    return (total_mass if total_mass != 0.0 else float("nan"), chosen_orbit)

def _reuse_count(cores_field: Any) -> int:
    """Count how many cores in this launch were marked reused=True."""
    if not isinstance(cores_field, list):
        return 0
    return int(sum(1 for c in cores_field if isinstance(c, dict) and c.get("reused") is True))

# --- main --------------------------------------------------------------------

def build_processed() -> Path:
    snap = latest_snapshot_entry()
    raw_path = Path(snap["file"])
    console.print(f"Processing snapshot: [cyan]{raw_path}[/cyan]")

    # Load raw launches (list[dict])
    launches = read_json(raw_path)
    df = pd.DataFrame(launches)

    # Fetch caches
    launchpads = fetch_with_cache(LAUNCHPADS_URL, "launchpads")
    rockets    = fetch_with_cache(ROCKETS_URL,    "rockets")
    payloads   = fetch_with_cache(PAYLOADS_URL,   "payloads")

    # Build maps for fast lookup
    launchpad_map = {lp.get("id"): (lp.get("name") or lp.get("full_name") or "Unknown") for lp in launchpads}
    rocket_map    = {rk.get("id"): (rk.get("name") or "Unknown") for rk in rockets}
    payload_map   = {pl.get("id"): {"mass_kg": pl.get("mass_kg"), "orbit": pl.get("orbit")} for pl in payloads}

    # Compute columns
    # Ensure these exist even if missing in some records
    for col in ["flight_number","date_utc","payloads","cores","launchpad","rocket","success"]:
        if col not in df.columns:
            df[col] = pd.NA

    # Year + weekend
    parts = df["date_utc"].apply(_dt_parts)
    df["year"] = parts.apply(lambda t: t[0])
    df["is_weekend"] = parts.apply(lambda t: t[1])

    # Payload mass + orbit
    mass_orbit = df["payloads"].apply(lambda ids: _sum_payload_mass_and_orbit(ids, payload_map))
    df["payload_mass_kg"] = mass_orbit.apply(lambda t: t[0])
    df["orbit"] = mass_orbit.apply(lambda t: t[1])

    # Launch site
    df["launch_site"] = df["launchpad"].map(lambda x: launchpad_map.get(x, "Unknown"))

    # Booster version (use rocket name as a stable proxy)
    df["booster_version"] = df["rocket"].map(lambda x: rocket_map.get(x, "Unknown"))

    # Reuse count
    df["reuse_count"] = df["cores"].apply(_reuse_count)

    # Keep and order the processed columns
    processed_cols = [
        "flight_number","date_utc","year","is_weekend",
        "payload_mass_kg","orbit","launch_site","booster_version",
        "reuse_count","success"
    ]
    proc = df[processed_cols].copy()

    # Types: flight_number numeric, year numeric, is_weekend 0/1 int, reuse_count int
    proc["flight_number"] = pd.to_numeric(proc["flight_number"], errors="coerce").astype("Int64")
    proc["year"]          = pd.to_numeric(proc["year"], errors="coerce").astype("Int64")
    proc["is_weekend"]    = pd.to_numeric(proc["is_weekend"], errors="coerce").fillna(0).astype("Int64")
    proc["reuse_count"]   = pd.to_numeric(proc["reuse_count"], errors="coerce").fillna(0).astype("Int64")

    # Save
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = PROCESSED_DIR / f"launches_processed_{stamp}.feather"
    proc.reset_index(drop=True).to_feather(out_path)

    # Report
    report_path = REPORTS_DIR / f"processed_{stamp}.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Processed file: {out_path}\n")
        f.write(f"Rows: {len(proc)}\n\n")
        f.write("Null counts:\n")
        f.write(proc.isna().sum().to_string())
        f.write("\n\nDtypes:\n")
        f.write(proc.dtypes.to_string())

    console.print(f"[green]Processed saved:[/green] {out_path.name}")
    console.print(f"[green]Report saved:[/green]    {report_path.name}")
    return out_path

if __name__ == "__main__":
    try:
        build_processed()
    except Exception as e:
        console.print(f"[red]Processing failed:[/red] {e}")
        raise SystemExit(1)
