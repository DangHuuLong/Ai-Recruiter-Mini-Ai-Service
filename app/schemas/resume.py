from pydantic import BaseModel, Field

class ParseResumeRequest(BaseModel):
    raw_text: str

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
    normalized_name: str | None = None
    category: str | None = None
    evidence: str | None = None


class ResumeEducation(BaseModel):
    school: str | None = None
    degree: str | None = None
    field_of_study: str | None = None
    start_date: str | None = None
    end_date: str | None = None


class ResumeExperience(BaseModel):
    company: str | None = None
    title: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    description: str | None = None
    skills: list[str] = Field(default_factory=list)


class ResumeProject(BaseModel):
    name: str | None = None
    description: str | None = None
    technologies: list[str] = Field(default_factory=list)
    role: str | None = None


class ResumeCertification(BaseModel):
    name: str
    issuer: str | None = None
    issued_date: str | None = None


class ParsedResumeData(BaseModel):
    personal: ResumePersonalInfo = Field(default_factory=ResumePersonalInfo)
    summary: str | None = None
    skills: list[ResumeSkill] = Field(default_factory=list)
    education: list[ResumeEducation] = Field(default_factory=list)
    experience: list[ResumeExperience] = Field(default_factory=list)
    projects: list[ResumeProject] = Field(default_factory=list)
    certifications: list[ResumeCertification] = Field(default_factory=list)
    achievements: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)