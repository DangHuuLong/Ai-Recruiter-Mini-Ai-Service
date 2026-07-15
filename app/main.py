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


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info("AI service started")
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
