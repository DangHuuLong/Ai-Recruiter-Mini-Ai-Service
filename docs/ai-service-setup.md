# AI Service Setup

## 1. Purpose

This document describes the current setup of the AI service for the AI Recruiter Mini project.

The AI service is a standalone Python service built with FastAPI. It provides deterministic MVP endpoints for health checks, resume parsing, job description parsing, and application scoring.

This document reflects the current implementation state and the latest parser work completed on 2026-04-30.

---

## 2. Current Tech Stack

| Tool | Purpose |
| --- | --- |
| Python | Main programming language |
| FastAPI | API framework |
| Uvicorn | ASGI server for running FastAPI |
| Pydantic | Data validation and response schema |
| Pydantic Settings | Environment configuration |
| python-dotenv | Load environment variables from `.env` |
| Pytest | Testing framework |
| HTTPX | Test client dependency used by FastAPI tests |

---

## 3. Current Folder Structure

```txt
ai-recruiter-mini-ai-service
├── app
│   ├── api
│   ├── core
│   ├── normalizers
│   ├── parsers
│   │   ├── extractors
│   │   └── normalizers
│   ├── schemas
│   ├── scorers
│   ├── services
│   ├── utils
│   └── main.py
├── docs
├── samples
├── scripts
├── tests
├── .env.example
├── pytest.ini
├── README.md
└── requirements.txt
```

---

## 4. Folder Responsibilities

| Folder | Responsibility |
| --- | --- |
| `app` | Main application source code |
| `app/api` | FastAPI route handlers |
| `app/core` | Global configuration and logging setup |
| `app/schemas` | Shared Pydantic schemas and parser/scoring contracts |
| `app/parsers` | Resume/JD parsing orchestration, text readers, section splitting, extractors, and normalizers |
| `app/normalizers` | Shared text/skill normalization helpers when applicable |
| `app/scorers` | CV-JD scoring logic |
| `app/services` | Service orchestration logic |
| `app/utils` | Shared utility functions |
| `tests` | Automated tests |
| `samples` | Sample input files for testing |
| `scripts` | Local development scripts |
| `docs` | Project documentation |

---

## 5. Environment Configuration

Environment variables are defined in `.env.example`.

```env
APP_NAME=ai-recruiter-mini-ai-service
APP_ENV=development
APP_VERSION=0.1.0
APP_PORT=8000

LOG_LEVEL=INFO

AI_PROVIDER=mock
GEMINI_API_KEY=
GEMINI_MODEL=gemini-3-flash-preview

REQUEST_TIMEOUT_SECONDS=30
```

For local development, create a `.env` file from `.env.example`:

```powershell
Copy-Item .env.example .env
```

The `.env` file is ignored by Git and must not be committed.

---

## 6. Configuration and Logging

Configuration is handled in:

```txt
app/core/config.py
```

Logging is configured in:

```txt
app/core/logging.py
```

The project uses cached settings through `get_settings()` to avoid repeatedly constructing configuration objects.

---

## 7. Common Response Schema

Common response schemas are defined in:

```txt
app/schemas/common.py
```

Success response format:

```json
{
  "success": true,
  "message": "AI service is healthy",
  "data": {}
}
```

Error response format:

```json
{
  "success": false,
  "message": "Invalid request",
  "errors": [
    {
      "field": "raw_text",
      "message": "Field is required"
    }
  ]
}
```

---

## 8. FastAPI Application Entry Point

The main FastAPI app is defined in:

```txt
app/main.py
```

Responsibilities:

- Configure logging
- Load application settings
- Create the FastAPI app instance
- Register API routers
- Handle startup and shutdown logging through FastAPI lifespan

Local docs:

```txt
http://localhost:8000/docs
http://localhost:8000/openapi.json
```

---

## 9. Health Check Endpoint

Endpoint:

```txt
GET /health
```

Expected response:

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

## 10. Local Setup

Create virtual environment:

```bash
python -m venv .venv
```

Activate on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run service:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Or run with the local script:

```powershell
.\scripts\run-local.ps1
```

---

## 11. Test Configuration

Pytest is configured in:

```txt
pytest.ini
```

Current configuration:

```ini
[pytest]
pythonpath = .
testpaths = tests
```

Useful commands:

```bash
python -m pytest
python -m pytest tests/test_parse_resume.py
python -m pytest -vv
```

Current test files:

| File | Coverage |
| --- | --- |
| `tests/test_health.py` | Health endpoint |
| `tests/test_parse_resume.py` | Resume parsing endpoint and regression cases |
| `tests/test_parse_job_description.py` | Job description parsing endpoint |
| `tests/test_score_application.py` | Application scoring endpoint |
| `tests/test_schemas.py` | Pydantic schema defaults and validation |
| `tests/test_file_readers.py` | PDF/DOCX readers with mocked dependencies |
| `tests/test_parser_utilities.py` | Duration normalization and section splitting |

---

## 12. Schema Layer

The schema layer defines structured data contracts used by the AI service.

Main data flow:

```txt
Raw resume text / raw JD text
  ↓
Parse and normalize
  ↓
ParsedResumeData / ParsedJobDescriptionData
  ↓
ScoreApplicationRequest
  ↓
EvaluationResult
```

Schema groups:

| Schema group | Purpose |
| --- | --- |
| Resume schemas | Parsed CV data |
| Job description schemas | Parsed JD data |
| Evaluation schemas | CV-JD scoring input/output |
| Common schemas | Shared API response and error formats |

---

## 13. Deterministic MVP API Endpoints

Current endpoints:

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `POST` | `/parse/resume` | Parses raw resume text into structured resume data |
| `POST` | `/parse/job-description` | Parses raw job description text into structured job description data |
| `POST` | `/score/application` | Scores a parsed resume against a parsed job description |

Implementation notes:

- Resume parsing is deterministic and rule-based.
- Job description parsing and scoring still use lightweight deterministic MVP logic.
- The API response shape remains stable and schema-compatible.
- The service does not use a real LLM, embeddings, OCR, or external AI provider yet.

---

## 14. Resume Parser Implementation Status

The resume parser is now a deterministic MVP parser, not a mock-only parser.

Implemented parser capabilities:

- PDF text extraction through `app/parsers/pdf_reader.py`
- DOCX text extraction through `app/parsers/docx_reader.py`
- Text normalization with optional line preservation
- Vietnamese accent normalization with explicit `Đ/đ` to `D/d` handling
- Resume section splitting for English and basic Vietnamese headings, including `Học tập`
- Email, phone, LinkedIn, GitHub, portfolio, and location extraction
- Skill extraction with aliases, categories, and normalized names
- Education extraction for institution, degree, field, start year, end year, GPA, and wrapped multi-line education blocks
- Controlled experience fallback for resumes without clear section headers, while avoiding false positives from profile titles
- Project extraction using block grouping and field-aware parsing
- Inline project header parsing, including URL preservation
- Certification, achievement, and language extraction
- `ParsedResumeData` schema-compatible output

Known MVP limits:

- The parser is rule-based and deterministic.
- OCR and scanned CV handling are not implemented.
- Highly visual or unusual CV layouts may still require better upstream text extraction from Backend.
- Skill coverage depends on the local skill catalog and aliases.
- Project metadata such as period, role, status, and feature list is currently folded into `projects[].description` because the project schema remains compact.
- Tooling terms such as `GitHub` may appear in skills or project technologies when present in the source text.

---

## 15. Work Completed on 2026-04-30

The following work was completed to improve frontend developer CV parsing quality and align with Backend PDF hyperlink extraction.

### Resume personal link parsing

- Normalized GitHub repository URLs into candidate profile URLs for `personal.github_url`.
- Added candidate-owned GitHub profile selection when multiple repository owners appear in project links.
- Preserved GitHub repository URLs separately in `projects[].url`.
- Improved portfolio URL detection for domains such as `github.io` and `super.site`.
- Fixed GitHub Pages portfolio handling so URLs such as `https://nhkkhaii.github.io/portfolio/` are not incorrectly reduced to `https://github.io/portfolio/`.

### Project extraction improvements

- Added support for project titles followed immediately by URL lines.
- Preserved project repository URLs for each project.
- Prevented URL-only lines from being copied into the beginning of `projects[].description`.
- Improved project grouping for frontend developer PDF text layouts.
- Added regression coverage for the `Ninh Hoang Khai` CV case with four projects:
  - `Personal Portfolio`
  - `KaiSneaker – E-commerce Website`
  - `CinemaNHK – Movie Ticket Booking System`
  - `Project Management – Construction Management System`

### Experience and layout robustness

- Improved stacked experience parsing for company → role → date layouts.
- Improved fallback parsing for wrapped PDF lines while avoiding false experience entries with missing company names.
- Added handling for page-break/layout noise in parsed project descriptions where possible.

### Test coverage

- Added regression tests for personal link normalization.
- Added regression tests for candidate-owned GitHub profile matching.
- Added regression tests for project title + URL layouts.
- Added regression tests to ensure project descriptions do not start with extracted URLs.

Run the full suite after changes:

```bash
python -m pytest
```
