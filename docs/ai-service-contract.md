# AI Service Contract

## 1. Purpose

This document defines the API contract between the Backend service and the AI service.

The AI service is an internal service. It is not called directly by the Frontend.

### Communication flow

```txt
Frontend
  ↓
Backend API
  ↓
Backend AiService
  ↓
AI Service
```

The AI service is responsible for:

- Parsing resume text
- Parsing job description text
- Scoring a parsed resume against a parsed job description

At the current stage, the AI service provides deterministic MVP endpoints. Resume parsing uses rule-based extraction, while job description parsing and scoring still use lightweight deterministic logic. The response shape should remain stable so the Backend can integrate with it before LLM/embedding-based logic is implemented.

---

## 2. Base URL

### Local development base URL

```txt
http://localhost:8000
```

### Backend environment variable

```txt
AI_SERVICE_URL=http://localhost:8000
```

---

## 3. Response Format

### Success response format

```json
{
  "success": true,
  "message": "Operation completed successfully",
  "data": {}
}
```

### Error response format

```json
{
  "success": false,
  "message": "Error message",
  "errors": [
    {
      "field": "fieldName",
      "message": "Validation message"
    }
  ]
}
```

---

## 4. Health Check

### Endpoint

```txt
GET /health
```

### Purpose

Used by the Backend to check whether the AI service is running.

### Response

```json
{
  "success": true,
  "message": "AI service is healthy",
  "data": {
    "service": "ai-recruiter-mini-ai-service",
    "status": "healthy",
    "version": "0.1.0",
    "environment": "development",
    "aiProvider": "mock"
  }
}
```

---

## 5. Parse Resume

### Endpoint

```txt
POST /parse/resume
```

### Purpose

Parses raw resume text into structured resume data.

The AI service does not receive uploaded files directly in this contract. The Backend is responsible for file upload, file storage, and raw text extraction before calling this endpoint.

The current implementation is a deterministic MVP parser. It uses rule-based text normalization, section splitting, and field extractors rather than an LLM. The response shape remains stable and must match `ParsedResumeData`.

### Request Body

```json
{
  "raw_text": "Nguyen Van A is a backend developer with experience in Python, FastAPI, PostgreSQL, Redis, and REST API development."
}
```

### Response Body

```json
{
  "success": true,
  "message": "Resume parsed successfully",
  "data": {
    "personal": {
      "full_name": "Nguyen Van A",
      "email": "nguyen@example.com",
      "phone": "+84912345678",
      "location": "Da Nang",
      "linkedin_url": null,
      "github_url": "https://github.com/nguyenvana",
      "portfolio_url": null
    },
    "summary": null,
    "skills": [
      {
        "name": "Python",
        "normalized_name": "python",
        "category": "language",
        "evidence": "Python, FastAPI, PostgreSQL"
      }
    ],
    "education": [
      {
        "institution": "University of Technology",
        "degree": "Bachelor",
        "field_of_study": "Computer Science",
        "start_year": 2019,
        "end_year": 2023,
        "description": "University of Technology - Bachelor of Computer Science, 2019 - 2023"
      }
    ],
    "experience": [],
    "projects": [
      {
        "name": "AI Recruiter",
        "description": "CV screening API using Python, FastAPI and PostgreSQL. Used Docker and Redis for local development.",
        "technologies": ["Python", "FastAPI", "PostgreSQL", "Docker", "Redis"],
        "url": "https://github.com/test/ai-recruiter"
      }
    ],
    "certifications": [],
    "achievements": [],
    "languages": []
  }
}
```

### Data Contract

- `data` must match the `ParsedResumeData` schema.
- Resume JSON payloads use `snake_case`.
- List fields such as `skills`, `education`, `experience`, `projects`, `certifications`, `achievements`, and `languages` default to empty lists.
- The parser may return `null` for unknown optional values.

### Main Fields

- `personal` — name, email, phone, location, LinkedIn, GitHub, portfolio
- `summary` — profile summary or objective when a clear summary section exists
- `skills` — extracted technical skills with normalized names, category, and evidence
- `education` — institution, degree, field of study, year range, and description
- `experience` — company, role, dates, duration, responsibilities, and technologies
- `projects` — project name, description, technologies, and URL
- `certifications`
- `achievements`
- `languages`

---

## 6. Current Resume Parser Behavior

The MVP resume parser currently supports:

- English and basic Vietnamese section headings, including `Học vấn`, `Học tập`, `Kỹ năng`, `Kinh nghiệm`, `Dự án`, and `Ngôn ngữ`.
- Vietnamese accent normalization that preserves `Đ/đ` as `D/d` for keyword matching.
- Personal information extraction for full name, email, phone, location, LinkedIn, GitHub, and portfolio.
- GitHub profile normalization from repository URLs, for example `https://github.com/nhkkhaii/QLDA` can produce `https://github.com/nhkkhaii` for `personal.github_url`.
- Candidate-owned GitHub profile selection when multiple repository owners appear in project links.
- Portfolio URL detection for domains such as `github.io`, `super.site`, `vercel.app`, `netlify.app`, `.dev`, `.me`, `.io`, and `.app`.
- Education block extraction across wrapped lines, including institution, degree approximation, field of study, start year, end year, GPA inside description, and Vietnamese student-status text such as `Sinh viên năm 3`.
- Stacked experience parsing for company → role → date layouts and company/role/date layouts with wrapped detail lines.
- Date normalization for common separators and flexible resume date formats.
- Project block extraction using field-aware grouping.
- Project titles followed immediately by URL lines, for example:

```txt
Personal Portfolio
https://nhkkhaii.github.io/portfolio/
Frontend Developer
Technologies: ReactJS, Bootstrap
```

- Project repository URL assignment into `projects[].url` without keeping URL-only lines at the start of `projects[].description`.
- GitHub repository URL preservation for project links, for example `https://github.com/nhkkhaii/CinemaNHK`.
- Project details such as `Link Github`, `Công nghệ sử dụng`, `Mô tả chức năng`, `Vai trò`, `Customer`, `Admin`, `Auth`, and `Trạng thái` are grouped into the current project description instead of being treated as separate projects.
- Certification, achievement, and language extraction with support for section-specific fallback behavior.

---

## 7. Current MVP Limits

- The parser is deterministic and rule-based, so unusual layouts can still require new heuristics.
- OCR and scanned CV handling are not implemented.
- Highly visual PDFs depend on the Backend's raw text extraction quality before parsing.
- Project metadata such as role, status, period, and feature list is currently folded into `projects[].description` because the contract still exposes a compact project schema.
- `GitHub` can appear as a tooling skill or project technology when it is present in project text.
- Summary extraction may still include extra content if the CV layout does not expose clear section boundaries.

---

## 8. Parse Job Description

### Endpoint

```txt
POST /parse/job-description
```

### Purpose

Parses raw job description text into structured job description data.

### Request Body

```json
{
  "raw_text": "We are looking for a Backend Developer with Python, FastAPI, PostgreSQL, Docker, Redis, and REST API experience."
}
```

### Response Body

```json
{
  "success": true,
  "message": "Job description parsed successfully",
  "data": {
    "title": "Backend Developer",
    "seniority": "junior",
    "employment_type": "full-time",
    "responsibilities": ["Develop and maintain backend APIs"],
    "requirements": ["Experience with Python and FastAPI"],
    "nice_to_have": ["Docker experience"],
    "required_skills": [
      {
        "name": "Python",
        "normalized_name": "python",
        "is_core": true,
        "weight_hint": 1.0
      }
    ],
    "preferred_skills": [],
    "min_experience_years": 1,
    "education_requirement": null,
    "domain_keywords": ["backend", "rest api"]
  }
}
```

### Main Fields

- `title`
- `seniority`
- `employment_type`
- `responsibilities`
- `requirements`
- `nice_to_have`
- `required_skills`
- `preferred_skills`
- `min_experience_years`
- `education_requirement`
- `domain_keywords`

---

## 9. Score Application

### Endpoint

```txt
POST /score/application
```

### Purpose

Scores a parsed resume against a parsed job description.

The Backend should call this endpoint after it already has:

- Parsed resume data
- Parsed job description data
- Scoring criteria configuration

### Response Body

```json
{
  "success": true,
  "message": "Application scored successfully",
  "data": {
    "overall_score": 75.0,
    "summary": "The candidate is a potential fit for the role.",
    "criteria": [],
    "skills": [],
    "explanation": "The candidate matches the main backend direction but is missing some required database evidence.",
    "skill_gap_summary": "Main missing skill: PostgreSQL.",
    "interview_questions": [],
    "evidence_map": {}
  }
}
```

### Main Fields

- `overall_score`
- `summary`
- `criteria`
- `skills`
- `explanation`
- `skill_gap_summary`
- `interview_questions`
- `evidence_map`

---

## 10. Backend Integration Notes

The Backend should call the AI service through an internal `AiService`.

Expected Backend methods:

- `checkHealth()`
- `parseResume()`
- `parseJobDescription()`
- `scoreApplication()`

Expected mapping:

| Backend method | AI service endpoint |
| --- | --- |
| `checkHealth()` | `GET /health` |
| `parseResume()` | `POST /parse/resume` |
| `parseJobDescription()` | `POST /parse/job-description` |
| `scoreApplication()` | `POST /score/application` |

The Backend should handle:

- AI service timeout
- AI service unavailable
- Invalid AI service response
- AI service error response

The Frontend must not call the AI service directly.

---

## 11. Naming Convention

The current AI service contract uses `snake_case` in JSON payloads.

Examples:

- `raw_text`
- `full_name`
- `normalized_name`
- `overall_score`
- `skill_gap_summary`
- `interview_questions`

If the Backend and Frontend use `camelCase`, the Backend is responsible for mapping AI service fields into the API response format expected by the Frontend.

Conversion examples:

- `overall_score` → `overallScore`
- `skill_gap_summary` → `skillGapSummary`
- `interview_questions` → `interviewQuestions`

---

## 12. MVP Parser Notes

During the MVP parser phase:

- Endpoints should return stable response shapes.
- Parser logic is rule-based and deterministic.
- The goal is to validate the Backend ↔ AI service contract and provide usable first-pass parsing.
- OCR, semantic matching, embeddings, and LLM-based parsing are not required yet.
- The deterministic parser implementation should be replaceable without changing the API contract.

### Recent parser improvements

The latest parser updates improved real CV support for frontend/backend developer resumes:

- Better stacked experience extraction from two-column PDF text layouts.
- Better project recovery when `Projects` appears near `Career History` in extracted PDF text.
- Better project URL assignment and preservation.
- Better GitHub and portfolio personal link normalization.
- Better support for GitHub Pages portfolio URLs and repository URLs.
- Better handling of project titles followed by URL lines.
- Cleaner project descriptions by removing URL-only lines after extracting `projects[].url`.
- Additional regression tests for frontend developer CV layouts and hyperlink/project URL cases.

### Current test status

Run tests with:

```bash
python -m pytest
```

The latest local target after these parser updates is:

```txt
20+ tests passing
```

If new parser tests are added, update this count after running the full suite.
