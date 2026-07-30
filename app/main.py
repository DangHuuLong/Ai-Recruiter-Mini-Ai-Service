import logging
import os
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("FLAGS_enable_mkldnn", "0")

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

import app.services.paddleocr_runtime_patch  # noqa: F401
from app.api.health import router as health_router
from app.api.parse import router as parse_router
from app.api.score import router as score_router
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.schemas.common import ApiErrorResponse, ErrorItem

configure_logging()

settings = get_settings()
logger = logging.getLogger(__name__)


def _warm_up_ml_models() -> None:
    """Eagerly load every configured ML model once at startup, before the
    server accepts traffic.

    Previously each model loaded lazily (@lru_cache) on its first request.
    lru_cache does serialize concurrent callers within a single process, but
    provides no protection at all across worker PROCESSES (each process has
    its own interpreter and its own cache) — if several requests land on
    different workers right after a cold start, multiple processes can end
    up loading the same HuggingFace model concurrently. Observed in
    production as an intermittent "Cannot copy out of meta tensor" error
    from torch during that window, with automatic fallback to the
    rule-based scorer (never a crash, but silently 0% ML-blended for those
    requests despite SIMILARITY_SCORING_WEIGHT > 0).

    Paying the load cost once here, before `yield`, removes that window
    entirely. If a model is disabled (fallback_mode=*_only) this is a no-op
    for it; if loading genuinely fails, it's logged loudly at startup
    instead of silently mid-request, and lazy loading still retries on the
    first real request (lru_cache does not cache exceptions)."""
    from app.ml.jd_section_classifier_model import get_jd_section_classifier_model
    from app.ml.section_classifier_model import get_section_classifier_model
    from app.ml.similarity_model import get_similarity_model

    for name, loader in (
        ("similarity (CV-JD CrossEncoder)", get_similarity_model),
        ("section_classifier (CV)", get_section_classifier_model),
        ("jd_section_classifier (JD)", get_jd_section_classifier_model),
    ):
        try:
            loader()
            logger.info("Warmed up ML model: %s", name)
        except RuntimeError as exc:
            logger.info("Skipped ML model warm-up (disabled): %s — %s", name, exc)
        except Exception:
            logger.exception("Failed to warm up ML model: %s — will retry lazily on first request", name)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info("AI service started")
    _warm_up_ml_models()
    yield
    logger.info("AI service stopped")


app = FastAPI(
    title="AI Recruiter Mini AI Service",
    description="Standalone AI service for resume parsing, job description parsing, and CV-JD scoring.",
    version=settings.app_version,
    lifespan=lifespan,
)

app.include_router(health_router)
app.include_router(parse_router)
app.include_router(score_router)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Conform to the ApiErrorResponse contract documented in
    docs/ai-service-contract.md (Section 3) — without this handler, FastAPI's
    default behavior returns {"detail": [...]}, which does not match what the
    Backend is told to expect."""
    errors = [
        ErrorItem(
            field=".".join(str(part) for part in error["loc"] if part != "body") or None,
            message=error["msg"],
        )
        for error in exc.errors()
    ]
    return JSONResponse(
        status_code=422,
        content=ApiErrorResponse(message="Request validation failed", errors=errors).model_dump(),
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Same contract-conformance fix as validation_exception_handler, for
    HTTPException raised directly by route handlers (e.g. DocumentTextExtractionError
    in app/api/parse.py)."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ApiErrorResponse(message=str(exc.detail), errors=[]).model_dump(),
    )
