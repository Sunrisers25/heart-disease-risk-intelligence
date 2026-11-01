"""Centralized backend configuration management."""

from typing import List, Union
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from src.config import DB_PATH


class Settings(BaseSettings):
    """Application settings loaded from environment variables with safe clinical defaults."""

    # Project metadata
    PROJECT_NAME: str = "Heart Disease Risk Intelligence API"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = (
        "Production-grade Clinical Decision Support REST API powered by scikit-learn "
        "and ACC/AHA Evidence-Based Guidelines."
    )
    API_V1_PREFIX: str = "/api/v1"
    DEBUG: bool = False

    # Security & CORS
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8501",  # Streamlit default frontend
        "http://127.0.0.1:8501",
    ]

    # Optional authentication hook
    API_KEY_ENABLED: bool = False
    API_KEY: str = ""

    # Model inference settings
    DEFAULT_CALIBRATE: bool = False
    ENABLE_AUDIT_LOGGING: bool = True

    # Database configuration
    DATABASE_URL: str = f"sqlite:///{DB_PATH}"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    @field_validator("ALLOWED_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",") if i.strip()]
        elif isinstance(v, list):
            return v
        return [str(v)]


settings = Settings()
