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

    similarity_model_path: str = ""
    similarity_base_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    similarity_model_version: str = ""
    similarity_scoring_weight: float = 0.0
    similarity_score_threshold: float = 0.0
    similarity_fallback_mode: str = "base_model"

    section_classifier_model_path: str = ""
    section_classifier_base_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    section_classifier_fallback_mode: str = "regex_only"

    enable_pdf_ocr_fallback: bool = False
    pdf_ocr_engine: str = "paddleocr"
    pdf_ocr_mode: str = "targeted"
    pdf_ocr_languages: str = "en"
    pdf_ocr_max_pages: int = 2
    pdf_ocr_min_confidence: float = 0.5

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()