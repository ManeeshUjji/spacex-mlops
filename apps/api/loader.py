# apps/api/loader.py
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib

from .settings import settings

log = logging.getLogger(__name__)


@dataclass
class Bundle:
    model_dir: Path
    model: Any
    feature_manifest: Dict[str, Any] | None
    train_config: Dict[str, Any] | None
    metrics: Dict[str, Any] | None
    model_version: str


# ---------- paths & metadata ----------

def _pointer_path() -> Path:
    # Most repos use registry/champion/pointer.json
    candidates = [
        Path("registry/champion/pointer.json"),
        Path("registry/registry/champion/pointer.json"),  # legacy/alt layout
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Champion pointer.json not found. Expected one of: "
        + ", ".join(str(p) for p in candidates)
        + ". You can also set MODEL_DIR_OVERRIDE to bypass the pointer."
    )


def _models_root() -> Path:
    return Path("registry/models")


def resolve_champion_dir() -> Path:
    ptr = json.loads(_pointer_path().read_text(encoding="utf-8"))
    version = ptr["champion_version"]
    model_dir = _models_root() / version
    if not model_dir.exists():
        raise RuntimeError(f"Resolved champion dir does not exist: {model_dir}")
    return model_dir


def _read_json_if_exists(p: Path) -> Optional[Dict[str, Any]]:
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            log.warning("Could not parse %s", p)
    return None


def load_meta_only() -> Tuple[Path, Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    model_dir = resolve_champion_dir()
    feature_manifest = _read_json_if_exists(model_dir / "feature_manifest.json")
    train_config = _read_json_if_exists(model_dir / "train_config.json")
    metrics = _read_json_if_exists(model_dir / "metrics.json")
    return model_dir, feature_manifest, train_config, metrics


# ---------- artifact loading ----------

# We’ll try these in order; the first one that exists wins.
_PACKAGE_CANDIDATES = [
    "package.joblib",
    "package.pkl",
    "model_package.joblib",
    "model_package.pkl",
    # many folks also just call the package “model.joblib”; we’ll inspect its type
    "model.joblib",
    "model.pkl",
]

def _load_any(path: Path) -> Any:
    return joblib.load(path)

def _pick_artifact(model_dir: Path) -> Tuple[Any, str]:
    """Return (artifact_object, filename_used). Prefer a *package dict* if present."""
    last_err = None
    for fname in _PACKAGE_CANDIDATES:
        p = model_dir / fname
        if not p.exists():
            continue
        try:
            obj = _load_any(p)
            # If it looks like our training package dict, use it immediately.
            if isinstance(obj, dict) and {"model", "feature_columns", "categorical_columns", "numeric_medians"} <= set(obj):
                return obj, fname
            # Otherwise keep as a fallback candidate (likely a fitted Pipeline)
            # but still return it if nothing better shows up.
            if fname in ("model.joblib", "model.pkl"):
                fallback = (obj, fname)
            else:
                return obj, fname
        except Exception as e:
            last_err = e
    if 'fallback' in locals():
        return fallback  # type: ignore[misc]
    if last_err:
        raise last_err
    raise FileNotFoundError(f"No model artifact found in {model_dir} (tried: {', '.join(_PACKAGE_CANDIDATES)})")


def load_bundle() -> Bundle:
    # Allow a manual override via env (handy during debugging)
    override = settings.model_dir_override
    if override:
        model_dir = Path(override)
        if not model_dir.exists():
            raise RuntimeError(f"Override model_dir does not exist: {model_dir}")
        feature_manifest = _read_json_if_exists(model_dir / "feature_manifest.json")
        train_config = _read_json_if_exists(model_dir / "train_config.json")
        metrics = _read_json_if_exists(model_dir / "metrics.json")
    else:
        model_dir, feature_manifest, train_config, metrics = load_meta_only()

    artifact, used_fname = _pick_artifact(model_dir)
    log.info("Loaded model artifact: %s/%s (type=%s)", model_dir, used_fname, type(artifact))

    # If the artifact is a raw sklearn Pipeline whose preprocessing stage isn't fitted,
    # we still return it — inference.py has logic to fall back to manual preprocessing
    # when a *package dict* is provided. Preferring the package above avoids that issue.

    # Figure out a version label for responses/health
    version = model_dir.name

    return Bundle(
        model_dir=model_dir,
        model=artifact,
        feature_manifest=feature_manifest,
        train_config=train_config,
        metrics=metrics,
        model_version=version,
    )
