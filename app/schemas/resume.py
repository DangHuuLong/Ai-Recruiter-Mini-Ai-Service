import re

from pydantic import BaseModel, Field, HttpUrl, model_validator

GPA_RE = re.compile(
    r"\b(?:Cumulative\s+)?GPA\s*[:\-]?\s*([0-9]+(?:\.[0-9]+)?)\s*/\s*([0-9]+(?:\.[0-9]+)?)",
    re.IGNORECASE,
)


class ParseResumeRequest(BaseModel):
    resume_id: str
    file_name: str | None = None
    file_type: str | None = None
    signed_url: HttpUrl | None = None
    raw_text: str | None = None
    checksum: str | None = None

    @model_validator(mode="after")
    def _require_file_or_raw_text(self) -> "ParseResumeRequest":
        if self.signed_url is None and self.raw_text is None:
            raise ValueError("Either signed_url or raw_text must be provided")
        if self.signed_url is not None and (self.file_name is None or self.file_type is None):
            raise ValueError("file_name and file_type are required when signed_url is provided")
        return self


class ResumePersonalInfo(BaseModel):
    full_name: str | None = None
    email: str | None = None
    phone: str | None = None
    location: str | None = None
    linkedin_url: str | None = None
    github_url: str | None = None
    portfolio_url: str | None = None


class ResumeSkill(BaseModel):
    name: str
    normalized_name: str
    category: str | None = None
    evidence: str | None = None
    level: str | None = None


class ResumeEducation(BaseModel):
    institution: str | None = None
    degree: str | None = None
    field_of_study: str | None = None
    start_year: int | None = None
    end_year: int | None = None
    gpa: str | None = None
    gpa_scale: str | None = None
    description: str | None = None

    @model_validator(mode="after")
    def populate_gpa_from_description(self) -> "ResumeEducation":
        if self.gpa and self.gpa_scale:
            return self

        match = GPA_RE.search(self.description or "")
        if not match:
            return self

        if not self.gpa:
            self.gpa = match.group(1)
        if not self.gpa_scale:
            self.gpa_scale = match.group(2)
        return self


class ResumeExperience(BaseModel):
    company: str | None = None
    role: str | None = None
    location: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    duration_months: int | None = None
    responsibilities: list[str] = Field(default_factory=list)
    technologies: list[str] = Field(default_factory=list)


class ResumeProject(BaseModel):
    name: str | None = None
    role: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    description: str | None = None
    technologies: list[str] = Field(default_factory=list)
    urls: list[str] = Field(default_factory=list)


class ResumeCertification(BaseModel):
    name: str | None = None
    issuer: str | None = None
    issued_year: int | None = None
    url: str | None = None


class ResumeAchievement(BaseModel):
    title: str | None = None
    description: str | None = None
    year: int | None = None


class ResumeLanguage(BaseModel):
    name: str
    proficiency: str | None = None


class ParsedResumeData(BaseModel):
    personal: ResumePersonalInfo = Field(default_factory=ResumePersonalInfo)
    summary: str | None = None
    skills: list[ResumeSkill] = Field(default_factory=list)
    education: list[ResumeEducation] = Field(default_factory=list)
    experience: list[ResumeExperience] = Field(default_factory=list)
    projects: list[ResumeProject] = Field(default_factory=list)
    certifications: list[ResumeCertification] = Field(default_factory=list)
    achievements: list[ResumeAchievement] = Field(default_factory=list)
    languages: list[ResumeLanguage] = Field(default_factory=list)


class ParseResumeResult(BaseModel):
    raw_text: str
    parsed_data: ParsedResumeData
    parser_version: str
    warnings: list[str] = Field(default_factory=list)
    confidence: float | None = None
    text_extraction_method: str
