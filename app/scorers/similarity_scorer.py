from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.schemas.evaluation import ScoreApplicationRequest

logger = logging.getLogger(__name__)


def score_cv_jd_similarity(request: ScoreApplicationRequest) -> float:
    from app.ml.similarity_model import get_similarity_model
    from app.scorers.cv_jd_scorer import job_text, resume_text

    model = get_similarity_model()
    cv_text = resume_text(request.resume)
    jd_text = job_text(request.job_description)

    embeddings = model.encode(
        [cv_text, jd_text],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    similarity = float(sum(float(a) * float(b) for a, b in zip(embeddings[0], embeddings[1])))
    return round(max(0.0, min(1.0, similarity)) * 100, 2)
