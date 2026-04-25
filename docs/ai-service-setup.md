# AI Service Setup Documentation

## 1. Purpose

This document tracks the setup process for the AI service of the AI Recruiter Mini project.

The AI service is responsible for providing independent AI-related capabilities for the recruitment screening pipeline, including:

- Resume parsing
- Job description parsing
- CV-JD scoring
- Explanation generation
- Interview question generation
- Skill gap analysis

In the first setup phase, the service will use mock endpoints and stable request/response schemas so that the Backend can integrate with it before real AI logic is implemented.

---

## 2. Repository Information

**Repository name:**

```
ai-recruiter-mini-ai-service
```

**Main branches:**

| Branch | Purpose |
|--------|---------|
| `main` | Stable production-ready branch |
| `develop` | Integration branch for development |
| `chore/ai-service-setup` | Initial setup branch for the AI service foundation |

---

## 3. Initial Setup Commands

### Clone Repository

```bash
git clone https://github.com/<your-username>/ai-recruiter-mini-ai-service.git
cd ai-recruiter-mini-ai-service
```

### Create Initial Commit

```bash
echo "# AI Recruiter Mini AI Service" > README.md
git add README.md
git commit -m "chore: initialize ai service repository"
```

### Push Main Branch

```bash
git branch -M main
git push -u origin main
```

### Create Develop Branch

```bash
git checkout -b develop
git push -u origin develop
```

### Create Setup Branch

```bash
git checkout develop
git pull origin develop
git checkout -b chore/ai-service-setup
git push -u origin chore/ai-service-setup
```

---

## 4. Project Setup Goal

The goal of this setup branch is to create the foundation for a standalone Python AI service using **FastAPI**.

The initial service should include:

- ✅ FastAPI application bootstrap
- ✅ Health check endpoint
- ✅ Environment configuration
- ✅ Logging configuration
- ✅ Request and response schemas
- ✅ Mock resume parser endpoint
- ✅ Mock job description parser endpoint
- ✅ Mock CV-JD scoring endpoint
- ✅ Sample CV/JD input files
- ✅ Local run script
- ✅ Basic tests

---

## 5. Planned Folder Structure

```
ai-recruiter-mini-ai-service/
├── app/
│   ├── main.py                           # FastAPI app entry point
│   ├── core/
│   │   ├── config.py                     # Configuration management
│   │   └── logging.py                    # Logging setup
│   ├── api/
│   │   ├── health.py                     # Health check endpoint
│   │   ├── parse.py                      # Parsing endpoints
│   │   └── score.py                      # Scoring endpoints
│   ├── parsers/
│   │   ├── resume_parser.py              # Resume parsing logic
│   │   └── job_description_parser.py     # Job description parsing logic
│   ├── normalizers/
│   │   ├── skill_normalizer.py           # Skill normalization
│   │   └── text_normalizer.py            # Text normalization utilities
│   ├── scorers/
│   │   └── cv_jd_scorer.py               # CV-JD scoring logic
│   ├── schemas/
│   │   ├── common.py                     # Common request/response schemas
│   │   ├── resume.py                     # Resume schemas
│   │   ├── job_description.py            # Job description schemas
│   │   └── evaluation.py                 # Evaluation schemas
│   ├── services/
│   │   ├── ai_provider_service.py        # AI provider integration
│   │   ├── parsing_service.py            # Parsing service orchestration
│   │   └── scoring_service.py            # Scoring service orchestration
│   └── utils/
│       └── text.py                       # Text utility functions
├── tests/                                # Test suite
│   ├── conftest.py                       # Pytest configuration
│   ├── test_health.py                    # Health endpoint tests
│   ├── test_parsers.py                   # Parser tests
│   └── test_scorers.py                   # Scorer tests
├── samples/                              # Sample input files
│   ├── sample_resume.pdf
│   ├── sample_resume.docx
│   └── sample_job_description.txt
├── scripts/                              # Utility scripts
│   └── run.sh                            # Local run script
├── docs/                                 # Documentation
│   └── API.md                            # API documentation
├── .env.example                          # Environment variables template
├── .gitignore                            # Git ignore rules
├── requirements.txt                      # Python dependencies
├── pyproject.toml                        # Python project configuration
├── README.md                             # Project README
└── .github/                              # GitHub workflows
    └── workflows/
        └── tests.yml                     # CI/CD testing workflow
```

---

## 6. Technology Stack

| Technology | Purpose |
|-----------|---------|
| **Python 3.10+** | Programming language |
| **FastAPI** | Web framework |
| **Pydantic** | Data validation |
| **Pytest** | Testing framework |
| **python-dotenv** | Environment configuration |
| **Uvicorn** | ASGI server |

---

## 7. Key Endpoints (Mock Phase)

### Health Check

```
GET /health
```

**Response:**

```json
{
  "status": "healthy",
  "version": "0.1.0"
}
```

### Parse Resume

```
POST /api/parse/resume
```

**Request:**

```json
{
  "file_path": "/path/to/resume.pdf"
}
```

**Response:**

```json
{
  "success": true,
  "data": {
    "contact_info": { ... },
    "experience": [ ... ],
    "education": [ ... ],
    "skills": [ ... ]
  }
}
```

### Parse Job Description

```
POST /api/parse/job-description
```

**Request:**

```json
{
  "text": "Job description text..."
}
```

**Response:**

```json
{
  "success": true,
  "data": {
    "title": "...",
    "description": "...",
    "required_skills": [ ... ],
    "preferred_skills": [ ... ]
  }
}
```

### Score CV vs JD

```
POST /api/score/evaluate
```

**Request:**

```json
{
  "resume_data": { ... },
  "job_data": { ... }
}
```

**Response:**

```json
{
  "success": true,
  "data": {
    "overall_score": 75.5,
    "criteria_scores": {
      "skills_match": 80.0,
      "experience_relevance": 70.0,
      "project_relevance": 75.0,
      "education_certification": 85.0,
      "keyword_domain_alignment": 65.0
    },
    "matched_skills": [ ... ],
    "missing_skills": [ ... ]
  }
}
```

---

## 8. Setup Status Checklist

| Task | Status |
|------|--------|
| Create empty repository | ⏳ Pending |
| Clone repository locally | ⏳ Pending |
| Create initial commit | ⏳ Pending |
| Create `develop` branch | ⏳ Pending |
| Create `chore/ai-service-setup` branch | ⏳ Pending |
| Create `docs` folder | ⏳ Pending |
| Create setup documentation | ⏳ Pending |
| Initialize Python project | ⏳ Pending |
| Create `pyproject.toml` | ⏳ Pending |
| Create `requirements.txt` | ⏳ Pending |
| Install FastAPI dependencies | ⏳ Pending |
| Create `app/main.py` | ⏳ Pending |
| Create health endpoint | ⏳ Pending |
| Create configuration system | ⏳ Pending |
| Create logging system | ⏳ Pending |
| Create base schemas | ⏳ Pending |
| Create mock parsing endpoints | ⏳ Pending |
| Create mock scoring endpoint | ⏳ Pending |
| Create sample input files | ⏳ Pending |
| Create local run script | ⏳ Pending |
| Create basic tests | ⏳ Pending |
| Create API documentation | ⏳ Pending |
| Create GitHub workflows | ⏳ Pending |

---

## 9. Development Workflow

### Setup Local Development Environment

```bash
# Clone the repository
git clone https://github.com/DangHuuLong/Ai-Recruiter-Mini-Ai-Service.git
cd ai-recruiter-mini-ai-service

# Checkout setup branch
git checkout chore/ai-service-setup

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the service
python -m uvicorn app.main:app --reload --port 8001
```

The service will be available at `http://localhost:8001`.

Health check: `http://localhost:8001/health`  
API documentation: `http://localhost:8001/docs`

### Running Tests

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=app

# Run specific test file
pytest tests/test_health.py

# Run in watch mode
pytest-watch
```

---

## 10. Environment Configuration

Create a `.env` file based on `.env.example`:

```env
# Application
APP_NAME=AI Recruiter Mini - AI Service
APP_VERSION=0.1.0
DEBUG=True

# API
API_HOST=0.0.0.0
API_PORT=8001

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/app.log

# AI Provider (will be configured later)
AI_PROVIDER=mock
AI_API_KEY=
```

---

## 11. Key Files to Create

### `requirements.txt`

```
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.4.2
pydantic-settings==2.0.3
python-dotenv==1.0.0
pytest==7.4.3
pytest-cov==4.1.0
pytest-watch==4.2.0
```

### `pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=65.0"]
build-backend = "setuptools.build_meta"

[project]
name = "ai-recruiter-mini-ai-service"
version = "0.1.0"
description = "AI Service for AI Recruiter Mini platform"
authors = [
    {name = "Your Name", email = "your.email@example.com"}
]
requires-python = ">=3.10"
dependencies = [
    "fastapi==0.104.1",
    "uvicorn[standard]==0.24.0",
    "pydantic==2.4.2",
    "pydantic-settings==2.0.3",
    "python-dotenv==1.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest==7.4.3",
    "pytest-cov==4.1.0",
    "pytest-watch==4.2.0",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
```

---

## 12. Integration with Backend

The backend (`Ai-Recruiter-Mini-Backend`) will communicate with this AI service via HTTP requests.

### Configuration in Backend

The backend will need to configure the AI service base URL:

```env
AI_SERVICE_BASE_URL=http://localhost:8001
```

### API Calls from Backend

The backend will call endpoints like:

```typescript
// Example: Parse a resume
const response = await fetch('http://localhost:8001/api/parse/resume', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ file_path: '/path/to/resume.pdf' })
});
```

---

## 13. Next Steps

1. **Initialize Repository** — Create GitHub repository and clone locally
2. **Setup Python Project** — Create virtual environment and install dependencies
3. **Bootstrap FastAPI** — Create main application and health check endpoint
4. **Create Schemas** — Define request/response data structures
5. **Implement Mock Endpoints** — Create mock parsing and scoring endpoints
6. **Add Configuration** — Setup environment variables and logging
7. **Write Tests** — Add basic test coverage for endpoints
8. **Create Documentation** — Document API and setup process
9. **Merge to Develop** — Create pull request and merge setup branch to develop
10. **Iterate** — Add real AI logic and additional features

---

## 14. References

- **FastAPI Documentation:** https://fastapi.tiangolo.com/
- **Pydantic Documentation:** https://docs.pydantic.dev/
- **Pytest Documentation:** https://docs.pytest.org/
- **Backend Repository:** https://github.com/DangHuuLong/Ai-Recruiter-Mini-Backend
- **Frontend Repository:** https://github.com/DangHuuLong/Ai-Recruiter-Mini-Frontend

---

## 15. Contact & Support

For questions or updates regarding the AI service setup:

- Create an issue in the repository
- Reference this documentation
- Link related issues from other repositories (backend, frontend)