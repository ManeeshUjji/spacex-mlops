# apps/api/preprocess.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline


def _load_snapshot_df(model_dir: Path) -> pd.DataFrame:
    """
    Try to load a tiny dataframe from data_snapshot.json to infer types/categories
    when the manifest doesn't spell them out. Works with dict-of-lists or list-of-rows.
    """
    snap = model_dir / "data_snapshot.json"
    if not snap.exists():
        return pd.DataFrame()
    try:
        payload = json.loads(snap.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return pd.DataFrame(payload)
        elif isinstance(payload, list):
            return pd.DataFrame(payload)
    except Exception:
        pass
    return pd.DataFrame()


def _infer_cols_from_manifest_or_snapshot(
    manifest: Dict[str, Any], model_dir: Path
) -> Tuple[List[str], List[str]]:
    """
    Robustly infer numerical and categorical columns from:
      - manifest["numerical"] / manifest["categorical"], or
      - manifest["columns"] with dtype, or
      - manifest["columns"] as plain names + snapshot-based inference.
    """
    # Case A: explicit lists
    if isinstance(manifest.get("numerical"), list) or isinstance(manifest.get("categorical"), list):
        nums = list(manifest.get("numerical", []))
        cats = list(manifest.get("categorical", []))
        return nums, cats

    cols = manifest.get("columns")

    # Case B: columns with dtype dicts
    if isinstance(cols, list) and cols and isinstance(cols[0], dict):
        nums, cats = [], []
        for col in cols:
            name = col.get("name")
            if not isinstance(name, str):
                continue
            dtype = str(col.get("dtype", "")).lower()
            if dtype in {"int", "int64", "float", "float64", "number", "numeric"}:
                nums.append(name)
            elif dtype in {"cat", "category", "categorical", "string", "object"}:
                cats.append(name)
            else:
                # unknown dtype → fall back to snapshot inference
                pass
        # If anything is still undecided, do a late pass with snapshot
        undecided = [
            c.get("name") for c in cols
            if isinstance(c, dict) and isinstance(c.get("name"), str)
            and c.get("name") not in nums + cats
        ]
        if undecided:
            df = _load_snapshot_df(model_dir)
            for c in undecided:
                if c in df.columns:
                    if pd.api.types.is_numeric_dtype(df[c]):
                        nums.append(c)
                    else:
                        cats.append(c)
        return nums, cats

    # Case C: columns is a list of names only → infer from snapshot
    if isinstance(cols, list) and (not cols or isinstance(cols[0], str)):
        df = _load_snapshot_df(model_dir)
        nums, cats = [], []
        for c in cols:
            if isinstance(c, str):
                if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
                    nums.append(c)
                else:
                    cats.append(c)
        return nums, cats

    # Fallback: nothing found
    return [], []


def _categories_from_manifest_or_snapshot(
    manifest: Dict[str, Any], model_dir: Path, cat_cols: List[str]
) -> List[List[str]]:
    """
    Prefer categories in manifest (if provided per column), else use snapshot uniques.
    If nothing available, return [] for that column so OHE learns on the fly and ignores unknowns.
    """
    cats_by_col: dict[str, List[str]] = {}

    # Look for per-column categories in manifest["columns"]
    cols = manifest.get("columns")
    if isinstance(cols, list):
        for item in cols:
            if isinstance(item, dict) and "name" in item and "categories" in item:
                name = item["name"]
                categories = item.get("categories")
                if isinstance(name, str) and isinstance(categories, list):
                    cats_by_col[name] = list(map(str, categories))

    # Fill from snapshot if missing
    df = _load_snapshot_df(model_dir)
    if not df.empty:
        for c in cat_cols:
            if c not in cats_by_col and c in df.columns:
                cats_by_col[c] = sorted(map(str, df[c].dropna().unique().tolist()))

    categories: List[List[str]] = []
    for c in cat_cols:
        categories.append(cats_by_col.get(c, []))  # [] → OHE will still work with handle_unknown="ignore"
    return categories


def make_runtime_preprocessor(feature_manifest: Dict[str, Any], model_dir: Path) -> ColumnTransformer:
    """
    Build a ColumnTransformer that:
      - passes numeric columns through
      - one-hot encodes categorical columns (training categories if available)
    Works with a variety of manifest shapes.
    """
    num_cols, cat_cols = _infer_cols_from_manifest_or_snapshot(feature_manifest, model_dir)

    categories = _categories_from_manifest_or_snapshot(feature_manifest, model_dir, cat_cols)

    cat_enc = OneHotEncoder(
        handle_unknown="ignore",
        categories=categories if any(categories) else "auto",
        sparse_output=False,
        dtype=np.float64,
    )

    ct = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", cat_enc, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return ct


def ensure_column_order(df: "pd.DataFrame", manifest: Dict[str, Any]) -> "pd.DataFrame":
    """Reorder to manifest['order'] if present; otherwise leave as-is."""
    order = manifest.get("order")
    if isinstance(order, list) and all(isinstance(c, str) for c in order):
        present = [c for c in order if c in df.columns]
        extras = [c for c in df.columns if c not in present]
        return df[present + extras]
    return df
