from app.parsers.job_description_parser import parse_job_description_mock
from app.parsers.resume_parser import parse_resume
from app.schemas.job_description import ParsedJobDescriptionData
from app.schemas.resume import ParsedResumeData


class ParsingService:
    def parse_resume(self, raw_text: str) -> ParsedResumeData:
        return parse_resume(self._sanitize_raw_text(raw_text))

    def parse_job_description(self, raw_text: str) -> ParsedJobDescriptionData:
        return parse_job_description_mock(self._sanitize_raw_text(raw_text))

    def _sanitize_raw_text(self, raw_text: str) -> str:
        return (raw_text or "").replace("\x00", "")


parsing_service = ParsingService()