from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SimilarityConfig(BaseModel):
    model_path: str = ""
    base_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    model_version: str = ""
    scoring_weight: float = Field(default=0.0, ge=0.0, le=1.0)
    score_threshold: float = Field(default=0.0, ge=0.0, le=100.0)
    fallback_mode: Literal["base_model", "rule_only"] = "base_model"

    def label_for_score(self, score: float) -> str:
        if score < 40:
            return "poor_match"
        if score < 60:
            return "weak_match"
        if score < 75:
            return "moderate_match"
        if score < 90:
            return "strong_match"
        return "excellent_match"


def get_similarity_config() -> SimilarityConfig:
    from app.core.config import get_settings

    s = get_settings()
    return SimilarityConfig(
        model_path=s.similarity_model_path,
        base_model=s.similarity_base_model,
        model_version=s.similarity_model_version,
        scoring_weight=s.similarity_scoring_weight,
        score_threshold=s.similarity_score_threshold,
        fallback_mode=s.similarity_fallback_mode,
    )
