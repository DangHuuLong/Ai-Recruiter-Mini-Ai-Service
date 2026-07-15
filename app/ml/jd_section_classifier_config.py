from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class JdSectionClassifierConfig(BaseModel):
    model_path: str = ""
    base_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    fallback_mode: Literal["regex_only", "model_with_regex_fallback"] = "regex_only"


def get_jd_section_classifier_config() -> JdSectionClassifierConfig:
    from app.core.config import get_settings

    s = get_settings()
    return JdSectionClassifierConfig(
        model_path=s.jd_section_classifier_model_path,
        base_model=s.jd_section_classifier_base_model,
        fallback_mode=s.jd_section_classifier_fallback_mode,
    )
