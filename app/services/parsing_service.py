from app.parsers.job_description_parser import parse_job_description
from app.parsers.resume_parser import parse_resume
from app.schemas.job_description import ParsedJobDescriptionData
from app.schemas.resume import ParseResumeRequest, ParseResumeResult, ParsedResumeData
from app.services.document_text_extraction_service import document_text_extraction_service

RESUME_PARSER_VERSION = "ai-document-resume-parser-v1"


class ParsingService:
    def parse_resume(self, request: ParseResumeRequest) -> ParseResumeResult:
        extraction_result = document_text_extraction_service.extract_from_signed_url(
            signed_url=str(request.signed_url),
            file_type=request.file_type,
            file_name=request.file_name,
        )
        raw_text = self._sanitize_raw_text(extraction_result.raw_text)
        parsed_resume = parse_resume(raw_text)

        warnings = list(extraction_result.warnings)
        confidence = 0.9 if raw_text else 0.0

        return ParseResumeResult(
            raw_text=raw_text,
            parsed_data=parsed_resume,
            parser_version=RESUME_PARSER_VERSION,
            warnings=warnings,
            confidence=confidence,
            text_extraction_method=extraction_result.method,
        )

    def parse_resume_text(self, raw_text: str) -> ParsedResumeData:
        return parse_resume(self._sanitize_raw_text(raw_text))

    def parse_job_description(self, raw_text: str) -> ParsedJobDescriptionData:
        return parse_job_description(self._sanitize_raw_text(raw_text))

    def _sanitize_raw_text(self, raw_text: str) -> str:
        return (raw_text or "").replace("\x00", "")


parsing_service = ParsingService()
