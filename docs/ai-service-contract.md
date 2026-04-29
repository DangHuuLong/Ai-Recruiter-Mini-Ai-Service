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
      }
    ],
    "education": [],
    "experience": [],
    "projects": [],
    "certifications": [],
    "achievements": [],
    "languages": []
  }
}
```

### Data Contract

- `data` must match the `ParsedResumeData` schema

### Main fields:

- `personal`
- `summary`
- `skills`
- `education`
- `experience`
- `projects`
- `certifications`
- `achievements`
- `languages`

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

### During the MVP parser phase:

- Endpoints should return stable response shapes
- Parser logic can be rule-based and deterministic
- The goal is to validate API contract and provide usable first-pass parsing
- OCR, semantic matching, embeddings, or LLM integration are not required yet

The deterministic parser implementation should be replaceable without changing the API contract.
