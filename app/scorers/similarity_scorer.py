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

    # CrossEncoder pair-scorer (see app/ml/similarity_model.py) — trained with
    # Identity activation on labels scaled to [0, 1] (score / 100), so predict()
    # returns a raw regression value approximating that same [0, 1] range
    # without an automatic sigmoid squash. Clip defensively since regression
    # output isn't hard-bounded.
    raw_score = float(model.predict([(cv_text, jd_text)])[0])
    return round(max(0.0, min(1.0, raw_score)) * 100, 2)
