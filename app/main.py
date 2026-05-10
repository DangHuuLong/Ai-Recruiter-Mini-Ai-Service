import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI

import app.services.paddleocr_runtime_patch  # noqa: F401
from app.api.health import router as health_router
from app.api.parse import router as parse_router
from app.api.score import router as score_router
from app.core.config import get_settings
from app.core.logging import configure_logging

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
