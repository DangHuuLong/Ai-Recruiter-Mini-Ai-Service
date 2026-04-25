from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "ai-recruiter-mini-ai-service"
    app_env: str = "development"
    app_version: str = "0.1.0"
    app_port: int = 8000

    log_level: str = "INFO"

    ai_provider: str = "mock"
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-3-flash-preview"

    request_timeout_seconds: int = 30

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()