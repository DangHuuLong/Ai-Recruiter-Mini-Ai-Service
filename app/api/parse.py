from fastapi import APIRouter, HTTPException

from app.schemas.common import ApiResponse
from app.schemas.job_description import (
    ParseJobDescriptionRequest,
    ParsedJobDescriptionData,
)
from app.schemas.resume import ParseResumeRequest, ParseResumeResult
from app.services.document_text_extraction_service import DocumentTextExtractionError
from app.services.parsing_service import parsing_service

router = APIRouter(prefix="/parse", tags=["parse"])


@router.post("/resume", response_model=ApiResponse[ParseResumeResult])
def parse_resume(request: ParseResumeRequest) -> ApiResponse[ParseResumeResult]:
    try:
        parsed_resume = parsing_service.parse_resume(request)
    except DocumentTextExtractionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return ApiResponse(
        success=True,
        message="Resume parsed successfully",
        data=parsed_resume,
    )


@router.post(
    "/job-description",
    response_model=ApiResponse[ParsedJobDescriptionData],
)
def parse_job_description(
    request: ParseJobDescriptionRequest,
) -> ApiResponse[ParsedJobDescriptionData]:
    try:
        parsed_job_description = parsing_service.parse_job_description(request)
    except DocumentTextExtractionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return ApiResponse(
        success=True,
        message="Job description parsed successfully",
        data=parsed_job_description,
    )