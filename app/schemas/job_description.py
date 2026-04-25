from pydantic import BaseModel, Field

class ParseJobDescriptionRequest(BaseModel):
    raw_text: str

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