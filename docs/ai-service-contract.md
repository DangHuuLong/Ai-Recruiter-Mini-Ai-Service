# AI Service Contract

## 1. Purpose

This document defines the API contract between the Backend service and the AI service.

The AI service is an internal service. It is not called directly by the Frontend.

### Communication flow:

```
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

### Local development base URL:

```
http://localhost:8000
```

### Backend environment variable:

```
AI_SERVICE_URL=http://localhost:8000
```

---

## 3. Response Format

### Success response format:

```json
{
  "success": true,
  "message": "Operation completed successfully",
  "data": {}
}
```

### Error response format:

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

```
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

```
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

### Main fields

- `personal` — name, email, phone, location, LinkedIn, GitHub, portfolio
- `summary` — profile summary or objective when a clear summary section exists
- `skills` — extracted technical skills with normalized names, category, and evidence
- `education` — institution, degree, field of study, year range, and description
- `experience` — company, role, dates, duration, responsibilities, and technologies
- `projects` — project name, description, technologies, and URL
- `certifications`
- `achievements`
- `languages`

### Current Resume Parser Behavior

The MVP resume parser now supports:

- English and basic Vietnamese section headings, including `Học vấn`, `Học tập`, `Kỹ năng`, `Kinh nghiệm`, `Dự án`, and `Ngôn ngữ`.
- Vietnamese accent normalization that preserves `Đ/đ` as `D/d` for keyword matching.
- Personal location extraction from labels such as `location`, `address`, `địa chỉ`, `quê quán`, and common city names.
- Education block extraction across wrapped lines, including institution, degree approximation, field of study, start year, end year, GPA inside description, and Vietnamese student-status text such as `Sinh viên năm 3`.
- Project block extraction using field-aware grouping. Project details such as `Link Github`, `Công nghệ sử dụng`, `Mô tả chức năng`, `Vai trò`, `Customer`, `Admin`, `Auth`, and `Trạng thái` are grouped into the current project description instead of being treated as separate projects.
- Inline project header parsing, for example `AI Recruiter - CV screening API ... https://github.com/...`.
- Controlled fallback experience parsing for resumes without explicit section headers, while avoiding false positives from simple profile titles such as `Developer`.

### Current MVP Limits

- The parser is deterministic and rule-based, so unusual layouts can still require new heuristics.
- OCR and scanned CV handling are not implemented.
- Project metadata such as role, status, period, and feature list is currently folded into `projects[].description` because the contract still exposes a compact project schema.
- `GitHub` can appear as a tooling skill or project technology when it is present in project text.
- Highly visual PDFs may require better upstream text extraction before parsing.

---

## 6. Parse Job Description

### Endpoint

```
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
    "responsibilities": [
      "Develop and maintain backend APIs"
    ],
    "requirements": [
      "Experience with Python and FastAPI"
    ],
    "nice_to_have": [
      "Docker experience"
    ],
    "required_skills": [
      {
        "name": "Python",
        "normalized_name": "python",
        "is_core": true,
        "weight_hint": 1.0
      },
      {
        "name": "FastAPI",
        "normalized_name": "fastapi",
        "is_core": true,
        "weight_hint": 1.0
      }
    ],
    "preferred_skills": [
      {
        "name": "Docker",
        "normalized_name": "docker",
        "is_core": false,
        "weight_hint": 0.5
      }
    ],
    "min_experience_years": 1,
    "education_requirement": null,
    "domain_keywords": [
      "backend",
      "rest api"
    ]
  }
}
```

### Data Contract

- `data` must match the `ParsedJobDescriptionData` schema

### Main fields:

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

## 7. Score Application

### Endpoint

```
POST /score/application
```

### Purpose

Scores a parsed resume against a parsed job description.

The Backend should call this endpoint after it already has:

- Parsed resume data
- Parsed job description data
- Scoring criteria configuration

### Request Body

```json
{
  "resume": {
    "personal": {
      "full_name": "Nguyen Van A",
      "email": null,
      "phone": null,
      "location": null,
      "linkedin_url": null,
      "github_url": null,
      "portfolio_url": null
    },
    "summary": "Mock parsed resume profile.",
    "skills": [
      {
        "name": "Python",
        "normalized_name": "python",
        "category": "backend",
        "evidence": "Mentioned in resume text"
      },
      {
        "name": "FastAPI",
        "normalized_name": "fastapi",
        "category": "backend",
        "evidence": "Mentioned in resume text"
      }
    ],
    "education": [],
    "experience": [],
    "projects": [],
    "certifications": [],
    "achievements": [],
    "languages": []
  },
  "job_description": {
    "title": "Backend Developer",
    "seniority": "junior",
    "employment_type": "full-time",
    "responsibilities": [
      "Develop and maintain backend APIs"
    ],
    "requirements": [
      "Experience with Python and FastAPI"
    ],
    "nice_to_have": [
      "Docker experience"
    ],
    "required_skills": [
      {
        "name": "Python",
        "normalized_name": "python",
        "is_core": true,
        "weight_hint": 1.0
      },
      {
        "name": "FastAPI",
        "normalized_name": "fastapi",
        "is_core": true,
        "weight_hint": 1.0
      },
      {
        "name": "PostgreSQL",
        "normalized_name": "postgresql",
        "is_core": true,
        "weight_hint": 0.8
      }
    ],
    "preferred_skills": [
      {
        "name": "Docker",
        "normalized_name": "docker",
        "is_core": false,
        "weight_hint": 0.5
      }
    ],
    "min_experience_years": 1,
    "education_requirement": null,
    "domain_keywords": [
      "backend",
      "rest api"
    ]
  },
  "config": {
    "criteria": [
      {
        "criterion": "SKILLS_MATCH",
        "weight": 0.35
      },
      {
        "criterion": "EXPERIENCE_RELEVANCE",
        "weight": 0.3
      },
      {
        "criterion": "PROJECT_RELEVANCE",
        "weight": 0.15
      },
      {
        "criterion": "EDUCATION_CERTIFICATION",
        "weight": 0.1
      },
      {
        "criterion": "KEYWORD_DOMAIN_ALIGNMENT",
        "weight": 0.1
      }
    ]
  }
}
```

### Response Body

```json
{
  "success": true,
  "message": "Application scored successfully",
  "data": {
    "overall_score": 75.0,
    "summary": "The candidate is a potential fit for the role.",
    "criteria": [
      {
        "criterion": "SKILLS_MATCH",
        "weight": 0.35,
        "score_normalized": 0.75,
        "reason": "The candidate matches several required backend skills.",
        "evidence": [
          "Python found in resume skills",
          "FastAPI found in resume skills"
        ]
      }
    ],
    "skills": [
      {
        "skill_name": "Python",
        "normalized_skill_name": "python",
        "type": "MATCHED",
        "importance": "HIGH",
        "evidence": "Python found in resume skills",
        "note": null
      },
      {
        "skill_name": "PostgreSQL",
        "normalized_skill_name": "postgresql",
        "type": "MISSING",
        "importance": "HIGH",
        "evidence": null,
        "note": "Required by the job description but not found in resume skills"
      }
    ],
    "explanation": "The candidate matches the main backend direction but is missing some required database evidence.",
    "skill_gap_summary": "Main missing skill: PostgreSQL.",
    "interview_questions": [
      {
        "question": "Can you describe how you used FastAPI in a real backend project?",
        "category": "backend",
        "linked_skill": "FastAPI",
        "difficulty": "MEDIUM",
        "rationale": "FastAPI is one of the required skills for this role.",
        "display_order": 1
      }
    ],
    "evidence_map": {
      "skills": [
        "Python found in resume skills",
        "FastAPI found in resume skills"
      ],
      "missing_skills": [
        "PostgreSQL required by JD but not found in resume"
      ]
    }
  }
}
```

### Data Contract

- `data` must match the `EvaluationResult` schema

### Main fields:

- `overall_score`
- `summary`
- `criteria`
- `skills`
- `explanation`
- `skill_gap_summary`
- `interview_questions`
- `evidence_map`

---

## 8. Backend Integration Notes

The Backend should call the AI service through an internal `AiService`.

### Expected Backend methods:

- `checkHealth()`
- `parseResume()`
- `parseJobDescription()`
- `scoreApplication()`

### Expected mapping:

| Backend method | AI service endpoint |
| --- | --- |
| `checkHealth()` | `GET /health` |
| `parseResume()` | `POST /parse/resume` |
| `parseJobDescription()` | `POST /parse/job-description` |
| `scoreApplication()` | `POST /score/application` |

### The Backend should handle:

- AI service timeout
- AI service unavailable
- Invalid AI service response
- AI service error response

### Important:

The Frontend must not call the AI service directly.

---

## 9. Naming Convention

The current AI service contract uses `snake_case` in JSON payloads.

### Examples:

- `raw_text`
- `full_name`
- `normalized_name`
- `overall_score`
- `skill_gap_summary`
- `interview_questions`

If the Backend and Frontend use `camelCase`, the Backend is responsible for mapping AI service fields into the API response format expected by the Frontend.

### Conversion examples:

- `overall_score` → `overallScore`
- `skill_gap_summary` → `skillGapSummary`
- `interview_questions` → `interviewQuestions`

---

## 10. MVP Parser Notes

### During the MVP parser phase

- Endpoints should return stable response shapes.
- Parser logic is rule-based and deterministic.
- The goal is to validate the Backend ↔ AI service contract and provide usable first-pass parsing.
- OCR, semantic matching, embeddings, and LLM-based parsing are not required yet.
- The deterministic parser implementation should be replaceable without changing the API contract.

### Recent parser improvements

The resume parser was updated to improve Vietnamese CV support and reduce project/education parsing errors:

- Added `Học tập` as an education heading.
- Improved Vietnamese accent normalization for `Đ/đ`.
- Added location extraction from Vietnamese profile labels such as `Quê quán`.
- Improved project grouping so feature lines are not split into separate projects.
- Improved inline project header parsing and project URL preservation.
- Improved education extraction for multi-line Vietnamese education blocks.
- Added a regression test for Vietnamese project and education parsing.

### Current test status

```txt
20 passed
```
