from pydantic import BaseModel, Field, HttpUrl, model_validator

class ParseJobDescriptionRequest(BaseModel):
    raw_text: str | None = None
    file_name: str | None = None
    file_type: str | None = None
    signed_url: HttpUrl | None = None

    @model_validator(mode="after")
    def _require_raw_text_or_file(self) -> "ParseJobDescriptionRequest":
        if self.raw_text is None and self.signed_url is None:
            raise ValueError("Either raw_text or signed_url must be provided")
        if self.signed_url is not None and (self.file_name is None or self.file_type is None):
            raise ValueError("file_name and file_type are required when signed_url is provided")
        return self

class JobSkill(BaseModel):
    name: str
    normalized_name: str | None = None
    is_core: bool = False
    weight_hint: float | None = None


class ParsedJobDescriptionData(BaseModel):
    title: str | None = None
    seniority: str | None = None
    employment_type: str | None = None
    responsibilities: list[str] = Field(default_factory=list)
    requirements: list[str] = Field(default_factory=list)
    nice_to_have: list[str] = Field(default_factory=list)
    required_skills: list[JobSkill] = Field(default_factory=list)
    preferred_skills: list[JobSkill] = Field(default_factory=list)
    min_experience_years: int | None = None
    education_requirement: str | None = None
    domain_keywords: list[str] = Field(default_factory=list)