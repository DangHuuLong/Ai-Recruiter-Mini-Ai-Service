from fastapi import APIRouter

from app.schemas.common import ApiResponse
from app.schemas.evaluation import EvaluationResult, ScoreApplicationRequest
from app.services.scoring_service import scoring_service

router = APIRouter(prefix="/score", tags=["score"])


@router.post("/application", response_model=ApiResponse[EvaluationResult])
def score_application(
    request: ScoreApplicationRequest,
) -> ApiResponse[EvaluationResult]:
    result = scoring_service.score_application(request)

    return ApiResponse(
        success=True,
        message="Application scored successfully",
        data=result,
    )