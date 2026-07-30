from fastapi import APIRouter

from app.core.config import get_settings
from app.schemas.common import ApiResponse

router = APIRouter(prefix="/health", tags=["health"])


@router.get("", response_model=ApiResponse[dict])
def health_check() -> ApiResponse[dict]:
    settings = get_settings()

    return ApiResponse(
        success=True,
        message="AI service is healthy",
        data={
            "service": settings.app_name,
            "status": "healthy",
            "version": settings.app_version,
            "environment": settings.app_env,
            "aiProvider": settings.ai_provider,
        },
    )