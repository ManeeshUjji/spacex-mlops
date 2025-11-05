from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import os, json, time
from typing import Optional
import joblib
from pathlib import Path

app = FastAPI(title="SpaceX Launch Success API", version="1.0.0")

# ---- Input schema (adjust if your Phase 3 schema differs) ----
class LaunchFeatures(BaseModel):
    payload_mass_kg: float = Field(..., ge=0)
    orbit: str
    launch_site: str
    booster_version: str
    reuse_count: int = Field(..., ge=0)
    flight_number: int = Field(..., ge=1)

# ---- Model loading helpers ----
def _champion_dir() -> Path:
    override = os.getenv("MODEL_DIR_OVERRIDE", "").strip()
    if override:
        return Path(override)
    # Try pointer.json under registry/champion
    champion_root = Path("/app/registry/champion")
    pointer = champion_root / "pointer.json"
    if pointer.exists():
        try:
            meta = json.loads(pointer.read_text())
            target = meta.get("target_dir")
            if target:
                return (champion_root / target).resolve()
        except Exception:
            pass
    # Fallback to a conventional path
    return Path("/app/registry/models/latest")

def _try_load_model():
    mdir = _champion_dir()
    if not mdir.exists():
        return None, str(mdir)
    # look for common filenames
    for name in ["model.joblib", "model.pkl", "clf.joblib", "clf.pkl"]:
        cand = mdir / name
        if cand.exists():
            try:
                model = joblib.load(cand)
                return model, str(mdir)
            except Exception:
                pass
    return None, str(mdir)

MODEL, MODEL_DIR = _try_load_model()

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "model_dir": MODEL_DIR
    }

@app.post("/predict")
def predict(x: LaunchFeatures):
    t0 = time.time()
    # If no model, return a deterministic stub so the endpoint still works
    if MODEL is None:
        # Simple stub logic to stay predictable for smoke tests
        score = 0.5
        return {
            "landed": score >= 0.5,
            "prob": score,
            "model_version": "stub-no-model",
            "latency_ms": round((time.time() - t0) * 1000, 2)
        }

    try:
        # Example vectorization — adapt to your trained model’s expected order
        vec = [[
            x.payload_mass_kg,
            x.reuse_count,
            x.flight_number
        ]]
        # If your model needs encoded strings, you should wrap a full pipeline during training.
        # Here we attempt proba if available, else decision_function, else predict.
        if hasattr(MODEL, "predict_proba"):
            prob = float(MODEL.predict_proba(vec)[0][1])
        elif hasattr(MODEL, "decision_function"):
            raw = float(MODEL.decision_function(vec)[0])
            # simple squash
            import math
            prob = 1.0 / (1.0 + math.exp(-raw))
        else:
            pred = int(MODEL.predict(vec)[0])
            prob = float(pred)

        return {
            "landed": prob >= 0.5,
            "prob": prob,
            "model_version": Path(MODEL_DIR).name or "unknown",
            "latency_ms": round((time.time() - t0) * 1000, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"inference_error: {e}")
