import re
from pathlib import Path

from app.parsers.job_description_parser import parse_job_description
from app.parsers.resume_parser import parse_resume
from app.schemas.job_description import ParsedJobDescriptionData
from app.schemas.resume import ParseResumeRequest, ParseResumeResult, ParsedResumeData
from app.services.document_text_extraction_service import document_text_extraction_service

RESUME_PARSER_VERSION = "ai-document-resume-parser-v1"
BAD_TEXT_MARKERS = ("·", "ï", "¿", "ˇ", "�")
NAME_STOP_WORDS = {
    "cv",
    "resume",
    "fullstack",
    "full",
    "stack",
    "developer",
    "frontend",
    "backend",
    "software",
    "engineer",
    "intern",
    "internship",
    "mobile",
    "web",
    "pdf",
    "doc",
    "docx",
}
ROLE_TITLE_TOKENS = {
    "developer",
    "engineer",
    "manager",
    "designer",
    "analyst",
    "consultant",
    "intern",
    "student",
}
NON_NAME_TOKENS = {
    "able",
    "admin",
    "api",
    "application",
    "applications",
    "authentication",
    "building",
    "communicate",
    "contribute",
    "core",
    "data",
    "development",
    "effectively",
    "environment",
    "environments",
    "experience",
    "feature",
    "features",
    "frontend",
    "github",
    "improve",
    "interfaces",
    "mobile",
    "projects",
    "reading",
    "remote",
    "skills",
    "software",
    "storage",
    "technical",
    "user",
    "workflows",
    "working",
}
SUMMARY_TAIL_PREFIXES = ("valuable", "software", "products", "solutions", "systems")


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
        recovered_name = self._recover_full_name_from_file_name(parsed_resume, request.file_name)
        if recovered_name:
            parsed_resume.personal.full_name = recovered_name
            warnings.append("Recovered full_name from file name because extracted PDF text name was missing, corrupted, or invalid")

        parsed_resume.summary = self._recover_summary_tail(parsed_resume.summary, raw_text)
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

    def _recover_full_name_from_file_name(self, parsed_resume: ParsedResumeData, file_name: str | None) -> str | None:
        current_name = parsed_resume.personal.full_name
        if current_name and not self._looks_corrupted(current_name) and not self._looks_invalid_person_name(current_name):
            return None

        candidate = self._name_candidate_from_file_name(file_name)
        if not candidate:
            return None

        return candidate

    def _looks_corrupted(self, value: str | None) -> bool:
        return bool(value) and any(marker in value for marker in BAD_TEXT_MARKERS)

    def _looks_invalid_person_name(self, value: str | None) -> bool:
        if not value:
            return True

        cleaned = re.sub(r"\s+", " ", value).strip()
        if not cleaned:
            return True

        if any(marker in cleaned for marker in BAD_TEXT_MARKERS):
            return True

        if any(token in cleaned.lower() for token in ["@", "http://", "https://", "phone:", "date:", "address:"]):
            return True

        words = cleaned.split()
        if not 2 <= len(words) <= 5:
            return True

        normalized_words = [re.sub(r"[^A-Za-zÀ-ỹ]", "", word).lower() for word in words]
        normalized_words = [word for word in normalized_words if word]
        if len(normalized_words) != len(words):
            return True

        if any(word in NON_NAME_TOKENS or word in ROLE_TITLE_TOKENS for word in normalized_words):
            return True

        # Real names from PDF headers are usually title case or uppercase. Fully lowercase
        # phrases such as "communicate effectively in remote" are body text, not names.
        if all(word == word.lower() for word in words):
            return True

        alpha_chars = re.sub(r"[^A-Za-zÀ-ỹ]", "", cleaned)
        if not alpha_chars:
            return True

        return False

    def _name_candidate_from_file_name(self, file_name: str | None) -> str | None:
        if not file_name:
            return None

        stem = Path(file_name).stem
        stem = re.sub(r"[_\-.]+", " ", stem)
        tokens = []
        for raw_token in stem.split():
            tokens.extend(self._split_camel_case(raw_token))

        clean_tokens = []
        for token in tokens:
            normalized = re.sub(r"[^A-Za-zÀ-ỹ]", "", token).strip()
            if not normalized:
                continue
            if normalized.lower() in NAME_STOP_WORDS:
                continue
            clean_tokens.append(normalized)

        if not 2 <= len(clean_tokens) <= 5:
            return None

        return " ".join(token[:1].upper() + token[1:] for token in clean_tokens)

    def _split_camel_case(self, value: str) -> list[str]:
        if not value:
            return []

        spaced = re.sub(r"(?<=[a-zà-ỹ])(?=[A-ZÀ-Ỹ])", " ", value)
        spaced = re.sub(r"(?<=[A-ZÀ-Ỹ])(?=[A-ZÀ-Ỹ][a-zà-ỹ])", " ", spaced)
        return spaced.split()

    def _recover_summary_tail(self, summary: str | None, raw_text: str) -> str | None:
        if not summary:
            return summary

        cleaned = summary.strip()
        if not cleaned or cleaned.endswith((".", "!", "?")):
            return cleaned

        lines = [line.strip() for line in (raw_text or "").splitlines() if line.strip()]
        summary_last_line = cleaned.splitlines()[-1].strip()
        try:
            last_index = max(index for index, line in enumerate(lines) if line == summary_last_line)
        except ValueError:
            return cleaned

        for line in lines[last_index + 1 : last_index + 40]:
            if not self._is_valid_summary_tail_candidate(line):
                continue
            normalized = line.lower().strip()
            if normalized.startswith(SUMMARY_TAIL_PREFIXES):
                return f"{cleaned} {line}".strip()

        return cleaned

    def _is_valid_summary_tail_candidate(self, line: str) -> bool:
        normalized = line.lower().strip()
        if not normalized:
            return False
        if any(marker in line for marker in BAD_TEXT_MARKERS):
            return False
        if any(token in normalized for token in ["@", "http://", "https://", "phone:", "date:", "address:"]):
            return False
        if normalized in {"github", "technical skills", "education", "projects", "experience"}:
            return False
        if any(token in normalized.split() for token in ROLE_TITLE_TOKENS):
            return False
        if len(line.split()) > 6:
            return False
        return True


parsing_service = ParsingService()
