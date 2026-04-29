from app.parsers.job_description_parser import parse_job_description_mock
from app.parsers.resume_parser import parse_resume
from app.schemas.job_description import ParsedJobDescriptionData
from app.schemas.resume import ParsedResumeData


class ParsingService:
    def parse_resume(self, raw_text: str) -> ParsedResumeData:
        return parse_resume(raw_text)

    def parse_job_description(self, raw_text: str) -> ParsedJobDescriptionData:
        return parse_job_description_mock(raw_text)


parsing_service = ParsingService()