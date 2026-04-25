from fastapi import APIRouter

from app.schemas.common import ApiResponse
from app.schemas.job_description import (
    ParseJobDescriptionRequest,
    ParsedJobDescriptionData,
)
from app.schemas.resume import ParseResumeRequest, ParsedResumeData
from app.services.parsing_service import parsing_service

router = APIRouter(prefix="/parse", tags=["parse"])


@router.post("/resume", response_model=ApiResponse[ParsedResumeData])
def parse_resume(request: ParseResumeRequest) -> ApiResponse[ParsedResumeData]:
    parsed_resume = parsing_service.parse_resume(request.raw_text)

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
    parsed_job_description = parsing_service.parse_job_description(request.raw_text)

    return ApiResponse(
        success=True,
        message="Job description parsed successfully",
        data=parsed_job_description,
    )