# apps/api/settings.py
from __future__ import annotations
from pydantic_settings import BaseSettings, SettingsConfigDict
model_config = {"protected_namespaces": ()}

class Settings(BaseSettings):
    # Optional override of the champion directory (absolute or relative).
    # Example: "registry/models/v20251030-0333"
    model_dir_override: str | None = None

    # Pydantic v2: use model_config only (do NOT keep an inner class Config)
    model_config = SettingsConfigDict(
        protected_namespaces=("settings_",),  # keep only 'settings_' reserved; allow model_* fields elsewhere
        env_prefix="APP_",                   # env vars like APP_MODEL_DIR_OVERRIDE
        extra="ignore",                      # ignore unexpected env/fields
    )


settings = Settings()
