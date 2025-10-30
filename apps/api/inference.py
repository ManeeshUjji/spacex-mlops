# apps/api/inference.py
from __future__ import annotations

import time
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd

from .loader import load_bundle
from .explainer import make_explanation  # optional, keep if you use /explain

# ---------------------------------------------------------------------
# Load champion bundle once at import (fast cold start)
# loader.load_bundle() should give us an object with at least:
#   .model  -> either a fitted sklearn estimator, OR a "package" dict from train_baselines.py
#   .feature_manifest (optional)
#   .train_config (optional)
#   .model_dir (optional)
# ---------------------------------------------------------------------
_BUNDLE = load_bundle()

# simple counters for /metrics
_METRICS = {
    "requests_total": 0,
    "errors_total": 0,
    "latencies_ms": [],
    "timestamps": [],
}


# ---------- helpers that work with *either* style of artifact ----------

def _is_package(model_obj: Any) -> bool:
    """True if loader returned the dict that train_baselines.py writes."""
    return isinstance(model_obj, dict) and all(
        k in model_obj
        for k in ["model", "feature_columns", "categorical_columns", "numeric_medians"]
    )


def _package_parts(pkg: Dict[str, Any]) -> Tuple[Any, List[str], List[str], Dict[str, float], Dict[str, Dict[str, Any]]]:
    """Unpack the training package dict into parts."""
    model = pkg["model"]
    feat_cols = list(pkg["feature_columns"])
    cat_cols = list(pkg["categorical_columns"])
    medians = dict(pkg["numeric_medians"])
    encoders = dict(pkg.get("encoders", {}))  # {col: {"classes_": [...], "missing_token": "__MISSING__"}}
    return model, feat_cols, cat_cols, medians, encoders


def _build_label_maps(encoders: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    """
    Turn saved LabelEncoder 'classes_' into mapping dicts per column.
    Unseen categories map to the saved 'missing_token' if present,
    else to index 0 (which is at least deterministic).
    """
    maps = {}
    for col, meta in encoders.items():
        classes = list(meta.get("classes_", []))
        m = {v: i for i, v in enumerate(classes)}
        missing = meta.get("missing_token")
        if missing is not None and missing not in m and classes:
            # if trainer never saw the missing token, append it to end
            m[missing] = len(classes)
        maps[col] = {"_map": m, "_missing": missing if missing is not None else None}
    return maps


def _encode_categoricals(df: pd.DataFrame, cat_cols: List[str], label_maps: Dict[str, Dict[str, Any]]) -> None:
    for c in cat_cols:
        if c not in df.columns:
            df[c] = None
        # cast to pandas "string" (like trainer)
        s = df[c].astype("string")
        meta = label_maps.get(c, {"_map": {}, "_missing": None})
        lut = meta["_map"]
        missing_tok = meta["_missing"]
        # replace NA with missing token first
        if missing_tok is not None:
            s = s.fillna(missing_tok)
        # vectorized map; unknowns -> missing token (if defined) else NaN -> then 0
        s = s.map(lut)
        if s.isna().any():
            if missing_tok is not None and missing_tok in lut:
                s = s.fillna(lut[missing_tok])
            else:
                s = s.fillna(0)
        df[c] = s.astype("int64")


def _impute_numerics(df: pd.DataFrame, medians: Dict[str, float], exclude: List[str]) -> None:
    for c in df.columns:
        if c in exclude:
            continue
        # try numeric cast like trainer (also handles nullable Int64)
        try:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        except Exception:
            pass
        fill = float(medians.get(c, 0.0))
        df[c] = df[c].fillna(fill).astype("float64")


def _make_dataframe(payload: Dict[str, Any], cols: List[str]) -> pd.DataFrame:
    """1-row DataFrame with exactly the feature columns in order."""
    row = {c: payload.get(c, None) for c in cols}
    return pd.DataFrame([row], columns=cols)


def _predict_with_any_model(model: Any, X: pd.DataFrame) -> Tuple[int, float]:
    """
    Works for sklearn classifiers that have predict_proba, decision_function,
    or at least predict.
    """
    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(X)[:, 1][0])
        label = int(proba >= 0.5 if not hasattr(model, "classes_") else model.classes_[np.argmax([1 - proba, proba])])
        # The label line above is defensive; we still return 0/1 below:
        label = int(proba >= 0.5)
        return label, proba

    if hasattr(model, "decision_function"):
        score = float(model.decision_function(X)[0])
        # crude sigmoid-ish mapping to (0,1) for a probability-like number
        proba = 1.0 / (1.0 + np.exp(-score))
        label = int(score >= 0.0)
        return label, proba

    # last resort
    y = model.predict(X)
    label = int(y[0])
    proba = float(label)  # 0 or 1
    return label, proba


# -------------------------- main entrypoint ----------------------------

def predict_one(payload: Dict[str, Any]) -> Tuple[int, float, Dict[str, Any]]:
    """
    Returns: (label, prob_1, debug_info)
    Will use the "package" artifact (train_baselines.py) if present,
    otherwise falls back to feeding the loader's .model directly.
    """
    t0 = time.perf_counter()

    model_obj = _BUNDLE.model  # whatever loader attached
    debug: Dict[str, Any] = {"used_package": False}

    try:
        # Path A: training "package" dict (preferred)
        if _is_package(model_obj):
            debug["used_package"] = True
            inner_model, feat_cols, cat_cols, medians, encoders = _package_parts(model_obj)
            label_maps = _build_label_maps(encoders)

            df = _make_dataframe(payload, feat_cols)
            # encode categoricals (exactly like training)
            _encode_categoricals(df, cat_cols, label_maps)
            # impute + cast numerics (training used medians + float64)
            _impute_numerics(df, medians, exclude=cat_cols)

            label, proba = _predict_with_any_model(inner_model, df)
            return label, proba, {"cols": feat_cols, "dtypes": df.dtypes.astype(str).to_dict()}

        # Path B: “old” artifact (e.g., a Pipeline that expects raw columns)
        # In this case we’ll just build a one-row DataFrame from whatever
        # feature manifest exists (if any), else from payload keys.
        feat_cols = None
        if hasattr(_BUNDLE, "feature_manifest") and isinstance(_BUNDLE.feature_manifest, dict):
            # try to respect order if manifest tracks it
            cols = _BUNDLE.feature_manifest.get("columns") or _BUNDLE.feature_manifest.get("feature_columns")
            if cols:
                feat_cols = list(cols)
        if feat_cols is None:
            feat_cols = list(payload.keys())

        df = _make_dataframe(payload, feat_cols)
        label, proba = _predict_with_any_model(model_obj, df)
        return label, proba, {"cols": feat_cols, "dtypes": df.dtypes.astype(str).to_dict()}

    finally:
        # update metrics
        dt_ms = (time.perf_counter() - t0) * 1000.0
        _METRICS["requests_total"] += 1
        _METRICS["latencies_ms"].append(dt_ms)
        _METRICS["timestamps"].append(time.time())


def rolling_stats(window: int = 60) -> Dict[str, Any]:
    """Tiny in-memory stats just for demo health/metrics."""
    hist = _METRICS["latencies_ms"][-window:]
    n = len(hist)
    if n == 0:
        return {"requests": _METRICS["requests_total"], "p50_ms": None, "p95_ms": None}
    arr = np.array(hist, dtype=float)
    return {
        "requests": _METRICS["requests_total"],
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
    }


# Optional: SHAP/LIME, if you wired /explain
def make_local_explanation(payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return make_explanation(_BUNDLE, payload)
    except Exception as e:
        return {"error": str(e)}
