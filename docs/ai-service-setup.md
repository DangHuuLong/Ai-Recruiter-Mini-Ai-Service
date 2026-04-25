# AI Service Setup

## 1. Purpose

This document describes the current setup of the AI service for the AI Recruiter Mini project.

The AI service is a standalone Python service built with FastAPI. At the current stage, the service only contains the base project structure, environment configuration, logging setup, common response schema, health check endpoint, and basic test setup.

This document only reflects the current implementation state.

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

```
ai-recruiter-mini-ai-service
├── app
│   ├── api
│   │   ├── __init__.py
│   │   └── health.py
│   ├── core
│   │   ├── __init__.py
│   │   ├── config.py
│   │   └── logging.py
│   ├── normalizers
│   │   └── __init__.py
│   ├── parsers
│   │   └── __init__.py
│   ├── schemas
│   │   ├── __init__.py
│   │   └── common.py
│   ├── scorers
│   │   └── __init__.py
│   ├── services
│   │   └── __init__.py
│   ├── utils
│   │   └── __init__.py
│   ├── __init__.py
│   └── main.py
├── docs
│   └── ai-service-setup.md
├── samples
├── scripts
│   └── run-local.ps1
├── tests
│   ├── conftest.py
│   └── test_health.py
├── .env.example
├── .gitignore
├── pytest.ini
├── README.md
└── requirements.txt
```

---

## 4. Folder Responsibilities

| Folder | Responsibility |
| --- | --- |
| app | Main application source code |
| app/api | FastAPI route handlers |
| app/core | Global configuration and logging setup |
| app/schemas | Shared Pydantic schemas |
| app/parsers | Placeholder for resume and job description parsing logic |
| app/normalizers | Placeholder for text and skill normalization logic |
| app/scorers | Placeholder for CV-JD scoring logic |
| app/services | Placeholder for service orchestration logic |
| app/utils | Shared utility functions |
| tests | Automated tests |
| samples | Sample input files for later testing |
| scripts | Local development scripts |
| docs | Project documentation |

---

## 5. Environment Configuration

Environment variables are defined in:

```
.env.example
```

### Current variables:

```
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

The `.env` file is ignored by Git and should not be committed.

---

## 6. Configuration File

Configuration is handled in:

```
app/core/config.py
```

This file defines the `Settings` class using `pydantic-settings`.

### Current responsibilities:

- Load environment variables from `.env`
- Provide application name, version, environment, and port
- Provide logging level
- Provide AI provider configuration
- Provide request timeout configuration

The project uses a cached settings function:

```python
@lru_cache
def get_settings() -> Settings:
    return Settings()
```

This avoids creating a new settings instance repeatedly across the application.

---

## 7. Logging Configuration

Logging is configured in:

```
app/core/logging.py
```

### Current log format:

```
timestamp | level | module | message
```

### Example:

```
2026-04-25 11:00:00 | INFO | app.main | AI service started
```

The log level is controlled by:

```
LOG_LEVEL=INFO
```

---

## 8. Common Response Schema

Common response schemas are defined in:

```
app/schemas/common.py
```

### Current schemas:

```python
class ErrorItem(BaseModel):
    field: str | None = None
    message: str


class ApiResponse(BaseModel, Generic[T]):
    success: bool
    message: str
    data: T | None = None


class ApiErrorResponse(BaseModel):
    success: bool = False
    message: str
    errors: list[ErrorItem] = []
```

These schemas are used to keep API responses consistent.

### Current success response format:

```json
{
  "success": true,
  "message": "AI service is healthy",
  "data": {}
}
```

### Current error response format:

```json
{
  "success": false,
  "message": "Invalid request",
  "errors": [
    {
      "field": "rawText",
      "message": "Field is required"
    }
  ]
}
```

---

## 9. FastAPI Application Entry Point

The main FastAPI app is defined in:

```
app/main.py
```

### Current responsibilities:

- Configure logging
- Load application settings
- Create the FastAPI app instance
- Register API routers
- Handle startup and shutdown logging through FastAPI lifespan

The app currently registers:

- `GET /health`

FastAPI documentation is available at:

```
http://localhost:8000/docs
```

OpenAPI JSON is available at:

```
http://localhost:8000/openapi.json
```

---

## 10. Health Check Endpoint

The health check route is defined in:

```
app/api/health.py
```

### Endpoint:

```
GET /health
```

### Expected response:

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

This endpoint is used to verify that the AI service is running.

---

## 11. Local Setup

### Create virtual environment:

```bash
python -m venv .venv
```

### Activate virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Install dependencies:

```bash
pip install -r requirements.txt
```

### Run the service:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Or run with the local script:

```powershell
.\scripts\run-local.ps1
```

### Health check URL:

```
http://localhost:8000/health
```

### Swagger URL:

```
http://localhost:8000/docs
```

---

## 12. Test Configuration

Pytest is configured in:

```
pytest.ini
```

### Current configuration:

```ini
[pytest]
pythonpath = .
testpaths = tests
```

The project also includes:

```
tests/conftest.py
```

This file ensures that the project root is added to `sys.path` during test execution, which helps avoid import issues on Windows.

### Run tests:

```bash
python -m pytest
```

### Current test file:

```
tests/test_health.py
```

### Current test coverage:

- Verify that `GET /health` returns HTTP 200
- Verify that the response has `success = true`
- Verify that the health status is `healthy`
- Verify that the service name is correct

### Expected result:

```
1 passed
```

---

## 13. Git Ignore Rules

The `.gitignore` file excludes local and generated files such as:

```
.venv/
.env
__pycache__/
.pytest_cache/
*.log
.vscode/
.idea/
build/
dist/
```

### Important rule:

```
.env must not be committed.
```

Only `.env.example` should be committed.

---

## 14. Schema Layer

The schema layer defines the structured data contracts used by the AI service.

Schemas are used to represent data after raw inputs have been extracted and normalized. They help ensure that the AI service returns predictable objects to the Backend instead of uncontrolled or inconsistent data.

### Current schema groups:

| Schema group | Purpose |
| --- | --- |
| Resume schemas | Represent parsed resume data after extracting information from a CV |
| Job description schemas | Represent parsed job description data after analyzing JD text |
| Evaluation schemas | Represent the input and output structure for CV-JD scoring |
| Common schemas | Represent shared API response and error response formats |

### Main data flow:

```
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

### Schema details:

- The resume schema groups information such as personal details, skills, education, experience, projects, certifications, achievements, and languages.
- The job description schema groups information such as title, seniority, responsibilities, requirements, required skills, preferred skills, experience requirement, education requirement, and domain keywords.
- The evaluation schema groups scoring output such as overall score, score breakdown, matched or missing skills, explanation, skill gap summary, interview questions, and evidence mapping.
- These schemas are not database models. They are API-level objects used by the AI service and Backend to exchange structured data consistently.

---

## 15. Schema Testing

Schema tests verify that the main data objects can be created correctly and safely.

### Purpose:

The purpose of these tests is to make sure that:

- Schema objects can be initialized with valid default values
- List fields default to empty lists instead of `None`
- Required fields are enforced where needed
- The schema contract is stable before building mock parse and scoring endpoints

### Run all tests:

```bash
python -m pytest
```

The schema tests are part of the basic safety check before adding parser, normalizer, scorer, and API endpoint logic.

---

## 16. Mock API Endpoints

The AI service currently provides mock endpoints for parsing and scoring.

These endpoints are used to validate the contract between the Backend and the AI service before real AI logic is implemented.

Current mock endpoints:

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `POST` | `/parse/resume` | Parses raw resume text into structured resume data |
| `POST` | `/parse/job-description` | Parses raw job description text into structured job description data |
| `POST` | `/score/application` | Scores a parsed resume against a parsed job description |

The mock implementation is deterministic and schema-based. It does not use a real LLM, embeddings, OCR, or external AI provider yet.