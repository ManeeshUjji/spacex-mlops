from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Literal, Optional
model_config = {"protected_namespaces": ()}

class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "error"]
    model_version: str
    last_trained_at: Optional[str] = None
    metrics_summary: dict
    
    # Allow fields starting with "model_"
    model_config = ConfigDict(protected_namespaces=())
    
class PredictRequest(BaseModel):
    flight_number: int = Field(ge=1, le=10000)
    year: int = Field(ge=2006, le=2035)
    payload_mass_kg: float = Field(ge=0, le=200000)
    orbit: str = Field(min_length=1, max_length=50)
    launch_site: str = Field(min_length=1, max_length=100)
    booster_version: str = Field(min_length=1, max_length=50)
    reuse_count: int = Field(ge=0, le=100)
    is_weekend: Literal[0, 1]

    model_config = ConfigDict(extra="forbid")

    @field_validator("orbit", "launch_site", "booster_version")
    @classmethod
    def normalize_string(cls, v: str) -> str:
        v = (v or "").strip()
        if v == "" or v.lower() in {"na", "n/a", "null", "none", "unknown"}:
            return "unknown"
        return v


class PredictResponse(BaseModel):
    landed: int = Field(description="Predicted landing success: 1 or 0")
    prob: float = Field(ge=0, le=1, description="Probability of success")
    explanation: str
    model_version: str
    latency_ms: float
    model_config = ConfigDict(protected_namespaces=())
    
class MetricsResponse(BaseModel):
    requests_total: int
    errors_total: int
    p50_ms: float
    p95_ms: float
    last_minute_rps: float
