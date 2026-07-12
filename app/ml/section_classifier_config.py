from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class SectionClassifierConfig(BaseModel):
    model_path: str = ""
    base_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    fallback_mode: Literal["regex_only", "model_with_regex_fallback"] = "regex_only"


def get_section_classifier_config() -> SectionClassifierConfig:
    from app.core.config import get_settings

    s = get_settings()
    return SectionClassifierConfig(
        model_path=s.section_classifier_model_path,
        base_model=s.section_classifier_base_model,
        fallback_mode=s.section_classifier_fallback_mode,
    )
