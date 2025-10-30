# apps/api/main.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import logging

from .schemas import (
    HealthResponse,
    PredictRequest,
    PredictResponse,
    MetricsResponse,
)
from .loader import load_meta_only
from .inference import predict_one, rolling_stats, _BUNDLE  # _BUNDLE is the loaded model bundle

# -----------------------------------------------------------------------------
# App
# -----------------------------------------------------------------------------
app = FastAPI(
    title="SpaceX Landing Prediction API",
    version="0.4.0",
    description="Phase 3: health + real predictions + basic metrics",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Basic logging
logging.basicConfig(level=logging.INFO)


# -----------------------------------------------------------------------------
# Startup logging (what model did we load?)
# -----------------------------------------------------------------------------
@app.on_event("startup")
def _log_model_info() -> None:
    try:
        model = _BUNDLE.model
        logging.info("Loaded model type: %s", type(model))
        for attr in ("predict_proba", "predict", "decision_function"):
            logging.info("Has %s: %s", attr, hasattr(model, attr))

        version = (
            _BUNDLE.train_config.get("version")
            or getattr(_BUNDLE.model_dir, "name", None)
            or "unknown"
        )
        logging.info("Champion model version: %s", version)
        logging.info("Model directory: %s", _BUNDLE.model_dir)
    except Exception as e:
        logging.exception("Failed to introspect loaded model: %s", e)


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """
    Returns status plus champion training metadata (version, last_trained_at, metrics snapshot).
    """
    try:
        model_dir, fm, cfg, met = load_meta_only()
        last_trained_at = met.get("generated_at")

        # prefer the winning candidate metrics if available
        metrics_val = met.get("metrics_val", {})
        winner_key = met.get("candidate_selection", {}).get("winner_model_key")
        if winner_key and winner_key in metrics_val:
            summary = metrics_val[winner_key]
        else:
            # fallback: first metrics dict or empty
            summary = next(iter(metrics_val.values()), met.get("metrics_snapshot", {}))

        version = cfg.get("version") or getattr(model_dir, "name", None) or "unknown"
        return HealthResponse(
            status="ok",
            model_version=version,
            last_trained_at=last_trained_at,
            metrics_summary=summary or {},
        )
    except Exception as e:
        logging.exception("Health check failed: %s", e)
        # degraded but still returns JSON for probes
        return HealthResponse(
            status="degraded",
            model_version="unknown",
            last_trained_at=None,
            metrics_summary={},
        )


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    """
    Predict landing success with strict validation and identical preprocessing
    to training (feature order, encoders, fills).
    """
    try:
        result = predict_one(payload.model_dump())
        return JSONResponse(result)
    except ValueError as ve:
        # schema-valid but semantically invalid (e.g., out of domain)
        logging.warning("Validation error during prediction: %s", ve)
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        # Hide internals from clients; logs have full traceback
        logging.exception("Prediction failed: %s", e)
        raise HTTPException(
            status_code=500,
            detail="Prediction failed; check inputs and model readiness.",
        )


@app.get("/metrics", response_model=MetricsResponse)
def metrics() -> MetricsResponse:
    """
    Basic in-memory counters/latency stats for the current process.
    """
    return MetricsResponse(**rolling_stats())
