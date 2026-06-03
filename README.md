# 🤖 AI Recruiter — Mini AI Service

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)
![Pydantic](https://img.shields.io/badge/Pydantic-E92063?style=flat&logo=pydantic&logoColor=white)
![Pytest](https://img.shields.io/badge/Pytest-0A9EDC?style=flat&logo=pytest&logoColor=white)

> Standalone AI service for resume parsing, job description analysis, and explainable CV-JD scoring. Supports the AI Recruiter Mini recruitment screening platform.

📦 **Repository:** [github.com/DangHuuLong/Ai-Recruiter-Mini-Ai-Service](https://github.com/DangHuuLong/Ai-Recruiter-Mini-Ai-Service)  
🔗 **Backend Repository:** [github.com/DangHuuLong/Ai-Recruiter-Mini-Backend](https://github.com/DangHuuLong/Ai-Recruiter-Mini-Backend)  
🌐 **Frontend Repository:** [github.com/DangHuuLong/Ai-Recruiter-Mini-Frontend](https://github.com/DangHuuLong/Ai-Recruiter-Mini-Frontend)

---

## ✨ Overview

**AI Recruiter Mini AI Service** is a standalone Python microservice that provides intelligent document analysis and scoring capabilities for the recruitment screening pipeline.

### Responsibilities

- **Resume Parsing** — Extract structured data from PDF/DOCX resumes via signed URL
- **Job Description Parsing** — Analyze and structure job requirements from raw text
- **Application Scoring** — Score a candidate against a job with 5-criteria weighted evaluation
- **Evidence-Based Evaluation** — Explainable scoring with matched/partial/missing skill classification and auto-generated interview questions
- **Document Text Extraction** — Multi-layer PDF/DOCX extraction with OCR fallback for scanned documents

### Architecture

```
Frontend (Next.js)
    ↓
Backend (NestJS)
    ├── Resume Upload & Storage
    ├── Job Description Management
    └── Application Management
         ↓
    AI Service (FastAPI) ← This Repository
         ├── Document Text Extraction (PDF/DOCX/OCR)
         ├── Resume Parsing
         ├── JD Parsing
         ├── CV-JD Scoring
         └── Evidence & Interview Question Generation
```

**Key Principle:** The Frontend calls the Backend only. The Backend calls the AI Service internally. The AI Service is not exposed directly to the Frontend.

---

## 🛠️ Technology Stack

| Technology | Purpose | Version |
|-----------|---------|---------|
| **Python** | Programming language | 3.10+ |
| **FastAPI** | Web framework & async support | 0.136.1 |
| **Uvicorn** | ASGI server | 0.46.0 |
| **Pydantic** | Data validation & schemas | 2.13.3 |
| **Pydantic Settings** | Environment configuration | 2.14.0 |
| **python-dotenv** | Load `.env` variables | 1.2.2 |
| **httpx** | Async HTTP client | 0.28.1 |
| **PyMuPDF (fitz)** | PDF text extraction | >=1.24.0 |
| **pypdf** | PDF reading fallback | >=5.0.0 |
| **python-docx** | DOCX text extraction | >=1.1.2 |
| **sentence-transformers** | Semantic skill embeddings | >=3.0.0,<4.0.0 |
| **PyYAML** | YAML configuration parsing | 6.0.3 |
| **Pytest** | Testing framework | 9.0.3 |

---

## 📁 Project Structure

```
ai-recruiter-mini-ai-service/
├── app/                              # Main application source code
│   ├── main.py                       # FastAPI entry point with lifespan management
│   ├── api/                          # API route handlers
│   │   ├── health.py                 # GET /health
│   │   ├── parse.py                  # POST /parse/resume, POST /parse/job-description
│   │   └── score.py                  # POST /score/application
│   ├── core/                         # Core configuration and setup
│   │   ├── config.py                 # Pydantic Settings (env vars)
│   │   └── logging.py                # Logging configuration
│   ├── schemas/                      # Pydantic data models
│   │   ├── common.py                 # Generic ApiResponse<T> and error schemas
│   │   ├── resume.py                 # Resume parsing schemas
│   │   ├── job_description.py        # JD parsing schemas
│   │   └── evaluation.py             # Scoring schemas
│   ├── parsers/                      # Document parsing and field extraction
│   │   ├── resume_parser.py          # Resume parsing orchestrator
│   │   ├── job_description_parser.py # JD parsing orchestrator
│   │   ├── pdf_reader.py             # PDF text extraction
│   │   ├── docx_reader.py            # DOCX text extraction
│   │   ├── normalizer.py             # Text normalization
│   │   ├── section_splitter.py       # Resume section detection
│   │   ├── extractors/               # Field-specific extractors
│   │   │   ├── skills.py
│   │   │   ├── education.py
│   │   │   ├── experience.py
│   │   │   ├── projects.py
│   │   │   ├── email_phone.py
│   │   │   ├── links.py
│   │   │   ├── achievements.py
│   │   │   └── languages.py
│   │   └── normalizers/
│   │       ├── skill_normalizer.py
│   │       └── duration_normalizer.py
│   ├── scorers/
│   │   └── cv_jd_scorer.py           # Deterministic 5-criteria CV-JD scoring
│   ├── services/                     # Service orchestration layer
│   │   ├── parsing_service.py        # Resume/JD parsing orchestration
│   │   ├── scoring_service.py        # Scoring service wrapper
│   │   ├── document_text_extraction_service.py  # PDF/DOCX download & extraction
│   │   └── paddleocr_runtime_patch.py
│   └── utils/                        # Utility functions
├── tests/                            # Test suite
│   ├── conftest.py                   # Pytest configuration and PATH setup
│   ├── test_health.py                # Health endpoint tests
│   ├── test_parse_resume.py          # Resume parsing + regression cases
│   ├── test_parse_job_description.py # JD parsing tests
│   ├── test_score_application.py     # Scoring endpoint tests
│   ├── test_schemas.py               # Pydantic schema validation
│   ├── test_file_readers.py          # PDF/DOCX reader unit tests
│   ├── test_parser_utilities.py      # Duration normalization, section splitter
│   ├── test_document_text_extraction_service.py
│   └── test_scoring_baseline.py      # Scoring baseline validation
├── docs/                             # Documentation
│   └── ai-service-setup.md           # Architecture and setup guide
├── samples/                          # Sample resumes and JDs for testing
├── datasets/                         # Training datasets
├── notebooks/                        # Jupyter notebooks (fine-tuning experiments)
├── training/                         # Training scripts for model fine-tuning
├── artifacts/                        # Generated model artifacts
├── scripts/                          # Development scripts
│   ├── run-local.ps1                 # Windows PowerShell local run script
│   ├── run-local.sh                  # Linux/macOS run script
│   └── reorder_resumes_by_group.py   # Dataset organization utility
├── .env.example                      # Environment template
├── .gitignore
├── pytest.ini                        # Pytest configuration
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

---

## 🚀 Getting Started

### ✅ Prerequisites

- **Python 3.10+** — Runtime environment
- **pip** — Package manager
- **Optional: PowerShell** — For running local scripts on Windows

### 📦 Installation

#### 1. Clone Repository

```bash
git clone https://github.com/DangHuuLong/Ai-Recruiter-Mini-Ai-Service.git
cd Ai-Recruiter-Mini-Ai-Service
```

#### 2. Create Virtual Environment

```bash
python -m venv .venv

# Linux/macOS:
source .venv/bin/activate

# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1

# Windows (CMD):
.venv\Scripts\activate.bat
```

#### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 4. Configure Environment

```bash
# Linux/macOS:
cp .env.example .env

# Windows (PowerShell):
Copy-Item .env.example .env
```

Edit `.env` with your configuration:

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

# OCR configuration (optional)
ENABLE_PDF_OCR_FALLBACK=false
PDF_OCR_ENGINE=paddleocr
PDF_OCR_MODE=targeted
PDF_OCR_LANGUAGES=en
PDF_OCR_MAX_PAGES=2
PDF_OCR_MIN_CONFIDENCE=0.5
```

### ▶️ Run the Service

#### Option 1: Direct Command

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

#### Option 2: PowerShell Script (Windows)

```powershell
.\scripts\run-local.ps1
```

#### Option 3: Bash Script (Linux/macOS)

```bash
./scripts/run-local.sh
```

The service starts at `http://localhost:8000`.

---

## 📚 API Documentation

Interactive documentation (Swagger UI): `http://localhost:8000/docs`  
Raw OpenAPI schema: `http://localhost:8000/openapi.json`

---

## 🔌 API Endpoints

### 1. Health Check

```
GET /health
```

**Response:**

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

### 2. Parse Resume

```
POST /parse/resume
```

Downloads the resume from a signed URL and returns fully structured data.

**Request:**

```json
{
  "resume_id": "uuid-string",
  "file_name": "john_doe.pdf",
  "file_type": "pdf",
  "signed_url": "https://storage.example.com/signed-url",
  "checksum": "optional-hash"
}
```

**Response:**

```json
{
  "success": true,
  "message": "Resume parsed successfully",
  "data": {
    "raw_text": "John Doe ...",
    "parsed_data": {
      "personal": {
        "full_name": "John Doe",
        "email": "john@example.com",
        "phone": "+84 123 456 789",
        "location": "Ho Chi Minh City",
        "links": [
          { "type": "github", "url": "https://github.com/johndoe" }
        ]
      },
      "summary": "Backend developer with 3 years of experience in Python and FastAPI.",
      "skills": [
        {
          "name": "Python",
          "normalized_name": "python",
          "category": "backend",
          "evidence": "Mentioned in resume text"
        }
      ],
      "education": [
        {
          "degree": "Bachelor of Science",
          "field": "Computer Science",
          "institution": "HCMUT",
          "start_year": 2018,
          "end_year": 2022
        }
      ],
      "experience": [
        {
          "title": "Backend Developer",
          "company": "Tech Corp",
          "start_date": "2022-06",
          "end_date": "present",
          "duration_months": 24,
          "description": "...",
          "skills_used": ["Python", "FastAPI", "PostgreSQL"]
        }
      ],
      "projects": [],
      "certifications": [],
      "achievements": [],
      "languages": []
    },
    "parser_version": "ai-document-resume-parser-v1",
    "warnings": [],
    "confidence": 0.9,
    "text_extraction_method": "pdf_text_extraction"
  }
}
```

---

### 3. Parse Job Description

```
POST /parse/job-description
```

**Request:**

```json
{
  "raw_text": "We are looking for a Backend Developer with 2+ years of experience. Required: Python, FastAPI, PostgreSQL. Nice to have: Docker, Redis."
}
```

**Response:**

```json
{
  "success": true,
  "message": "Job description parsed successfully",
  "data": {
    "title": "Backend Developer",
    "seniority": "junior",
    "employment_type": "full-time",
    "responsibilities": ["Develop and maintain backend APIs"],
    "requirements": ["2+ years of backend development experience"],
    "nice_to_have": ["Docker and containerization experience"],
    "required_skills": [
      { "name": "Python", "normalized_name": "python", "is_core": true, "weight_hint": 1.0 },
      { "name": "FastAPI", "normalized_name": "fastapi", "is_core": true, "weight_hint": 1.0 },
      { "name": "PostgreSQL", "normalized_name": "postgresql", "is_core": true, "weight_hint": 0.8 }
    ],
    "preferred_skills": [
      { "name": "Docker", "normalized_name": "docker", "is_core": false, "weight_hint": 0.5 }
    ],
    "min_experience_years": 2,
    "education_requirement": null,
    "domain_keywords": ["backend", "rest api", "databases"]
  }
}
```

---

### 4. Score Application

```
POST /score/application
```

Evaluates a parsed resume against a job description using 5 weighted criteria.

**Scoring Criteria:**

| Criterion | Default Weight | Description |
|-----------|---------------|-------------|
| `SKILLS_MATCH` | 35% | Exact/partial/missing skill classification |
| `EXPERIENCE_RELEVANCE` | 30% | Years of experience + skill overlap in work history |
| `PROJECT_RELEVANCE` | 15% | Skill evidence in projects |
| `EDUCATION_CERTIFICATION` | 10% | Degree field matching |
| `KEYWORD_DOMAIN_ALIGNMENT` | 10% | Domain term frequency in resume |

**Request:**

```json
{
  "resume": { "...parsed_data..." },
  "job_description": { "...parsed_jd_data..." },
  "config": {
    "criteria": [
      { "criterion": "SKILLS_MATCH", "weight": 0.35 },
      { "criterion": "EXPERIENCE_RELEVANCE", "weight": 0.30 },
      { "criterion": "PROJECT_RELEVANCE", "weight": 0.15 },
      { "criterion": "EDUCATION_CERTIFICATION", "weight": 0.10 },
      { "criterion": "KEYWORD_DOMAIN_ALIGNMENT", "weight": 0.10 }
    ]
  }
}
```

**Response:**

```json
{
  "success": true,
  "message": "Application scored successfully",
  "data": {
    "overall_score": 78.5,
    "summary": "Strong fit for the role with most required skills.",
    "criteria": [
      {
        "criterion": "SKILLS_MATCH",
        "weight": 0.35,
        "score_normalized": 0.8,
        "reason": "Candidate matches key backend technologies",
        "evidence": ["Python found in resume", "FastAPI found in resume"]
      }
    ],
    "skills": [
      {
        "skill_name": "Python",
        "normalized_skill_name": "python",
        "type": "MATCHED",
        "importance": "HIGH",
        "evidence": "Python found in resume skills"
      },
      {
        "skill_name": "PostgreSQL",
        "normalized_skill_name": "postgresql",
        "type": "MISSING",
        "importance": "HIGH",
        "note": "Required but not found in resume"
      }
    ],
    "explanation": "Candidate has strong backend fundamentals but lacks database experience.",
    "skill_gap_summary": "Main gap: PostgreSQL database experience.",
    "interview_questions": [
      {
        "question": "Can you describe your experience with PostgreSQL and database design?",
        "category": "technical",
        "linked_skill": "PostgreSQL",
        "difficulty": "MEDIUM",
        "rationale": "PostgreSQL is a core requirement for this role",
        "display_order": 1
      }
    ],
    "evidence_map": {
      "skills": ["Python found in resume", "FastAPI found in resume"],
      "missing_skills": ["PostgreSQL required but not found"]
    }
  }
}
```

---

## 🧪 Testing

```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=app

# Run a specific test file
python -m pytest tests/test_parse_resume.py

# Run in watch mode
pytest-watch
```

### Test Coverage

| Test File | Coverage |
|-----------|----------|
| `test_health.py` | Health endpoint structure and status |
| `test_parse_resume.py` | Resume parsing + regression cases (layouts, link normalization) |
| `test_parse_job_description.py` | JD parsing validation |
| `test_score_application.py` | Scoring with various CV-JD combinations |
| `test_schemas.py` | Pydantic schema defaults and validation |
| `test_file_readers.py` | PDF/DOCX readers (mocked dependencies) |
| `test_parser_utilities.py` | Duration normalization, section splitting |
| `test_document_text_extraction_service.py` | Text extraction service |
| `test_scoring_baseline.py` | Scoring logic baseline validation |

---

## ⚙️ Configuration

### Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `APP_NAME` | Application name | `ai-recruiter-mini-ai-service` |
| `APP_ENV` | Environment (`development`/`production`) | `development` |
| `APP_VERSION` | Application version | `0.1.0` |
| `APP_PORT` | Server port | `8000` |
| `LOG_LEVEL` | Logging level (`INFO`/`DEBUG`/`WARNING`) | `INFO` |
| `AI_PROVIDER` | AI provider (`mock`/`gemini`) | `mock` |
| `GEMINI_API_KEY` | Google Gemini API key | (empty) |
| `GEMINI_MODEL` | Gemini model name | `gemini-3-flash-preview` |
| `REQUEST_TIMEOUT_SECONDS` | Outbound request timeout | `30` |
| `ENABLE_PDF_OCR_FALLBACK` | Enable OCR for scanned PDFs | `false` |
| `PDF_OCR_ENGINE` | OCR engine (`paddleocr`) | `paddleocr` |
| `PDF_OCR_MODE` | OCR mode (`targeted`/`full`) | `targeted` |
| `PDF_OCR_LANGUAGES` | OCR language codes | `en` |
| `PDF_OCR_MAX_PAGES` | Max pages to OCR per document | `2` |
| `PDF_OCR_MIN_CONFIDENCE` | Minimum OCR confidence threshold | `0.5` |

### Logging Format

```
timestamp | level | module | message
2026-04-25 11:00:00 | INFO | app.main | AI service started
```

---

## 📡 Backend Integration

### Backend Configuration

```env
AI_SERVICE_URL=http://localhost:8000
```

### Integration Pattern

```python
class AiService:
    async def check_health(self):
        return await self.call('GET', '/health')

    async def parse_resume(self, resume_id, file_name, file_type, signed_url):
        return await self.call('POST', '/parse/resume', {
            'resume_id': resume_id,
            'file_name': file_name,
            'file_type': file_type,
            'signed_url': signed_url
        })

    async def parse_job_description(self, raw_text: str):
        return await self.call('POST', '/parse/job-description', {'raw_text': raw_text})

    async def score_application(self, resume, job_description, config):
        return await self.call('POST', '/score/application', {
            'resume': resume,
            'job_description': job_description,
            'config': config
        })
```

### Response Contract

All endpoints follow a consistent structure:

**Success:**
```json
{ "success": true, "message": "...", "data": {} }
```

**Error:**
```json
{ "success": false, "message": "...", "errors": [{ "field": "fieldName", "message": "..." }] }
```

---

## 🔄 Data Schemas

Schemas are defined in `app/schemas/`:

**Resume (`resume.py`)** — `ParsedResumeData`:
- `personal` — Name, email, phone, location, social links
- `summary` — Professional summary
- `skills` — Extracted skills with category and normalization
- `education` — Degree, institution, field, years
- `experience` — Job titles, companies, durations, skills used
- `projects` — Project names, descriptions, tech stack
- `certifications` — Professional certifications
- `achievements` — Awards and notable achievements
- `languages` — Spoken languages

**Job Description (`job_description.py`)** — `ParsedJobDescriptionData`:
- `title`, `seniority`, `employment_type`
- `responsibilities`, `requirements`, `nice_to_have`
- `required_skills`, `preferred_skills` (with weights)
- `min_experience_years`, `education_requirement`, `domain_keywords`

**Evaluation (`evaluation.py`)** — `EvaluationResult`:
- `overall_score` (0–100), `summary`
- `criteria` — Per-criterion breakdown with weights and evidence
- `skills` — `MATCHED` / `PARTIAL` / `MISSING` classification per skill
- `explanation`, `skill_gap_summary`
- `interview_questions` — Auto-generated with difficulty and rationale
- `evidence_map` — Links scores back to parsed resume data

---

## 📊 Current Status

| Feature | Status |
|---------|--------|
| Project setup & structure | ✅ Complete |
| FastAPI bootstrap & lifespan | ✅ Complete |
| Configuration system (Pydantic Settings) | ✅ Complete |
| Logging system | ✅ Complete |
| Response schemas (all endpoints) | ✅ Complete |
| Health check endpoint | ✅ Complete |
| Document text extraction (PDF/DOCX) | ✅ Complete |
| OCR fallback for scanned PDFs | ✅ Complete |
| Resume parsing endpoint | ✅ Complete |
| JD parsing endpoint | ✅ Complete |
| CV-JD scoring endpoint | ✅ Complete |
| Deterministic 5-criteria scorer | ✅ Complete |
| Evidence & interview question generation | ✅ Complete |
| Comprehensive test suite | ✅ Complete |
| Training datasets | ✅ Complete |
| Fine-tuning notebooks (Colab) | ✅ Complete |
| Real AI integration (Gemini) | 📋 Planned |
| Semantic embeddings (sentence-transformers) | 📋 Planned |
| Response caching | 📋 Planned |
| Authentication / API key guard | 📋 Planned |
| Monitoring & metrics | 📋 Planned |
| Docker containerization | 📋 Planned |

---

## 🧠 ML / Fine-Tuning

The `notebooks/`, `training/`, and `datasets/` directories contain work for improving parsing accuracy via model fine-tuning:

- **`datasets/`** — Labeled resume and JD samples organized by category
- **`training/`** — Fine-tuning scripts for similarity and extraction models
- **`notebooks/`** — Google Colab notebooks for running fine-tuning experiments
- **`artifacts/`** — Generated model checkpoints and outputs

See `docs/ai-service-setup.md` for architecture details and the Colab guide for running fine-tuning experiments.

---

## 📖 Documentation

- **[AI Service Setup](docs/ai-service-setup.md)** — Architecture, setup, and integration guide
- **Swagger UI** — `http://localhost:8000/docs`
- **OpenAPI JSON** — `http://localhost:8000/openapi.json`

---

## 🤝 Contributing

1. Activate the virtual environment before running any commands
2. Follow Python PEP 8 style guide
3. Add tests for new features or bug fixes
4. Run the full test suite before committing:
   ```bash
   python -m pytest
   ```
5. Keep API schemas stable — breaking changes require Backend coordination
6. Update this README and `docs/` when adding endpoints or configuration options

---

## 🐛 Troubleshooting

**Port already in use:**
```bash
uvicorn app.main:app --port 8001 --reload
# Then update AI_SERVICE_URL in Backend .env
```

**Windows — virtual environment activation fails:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\.venv\Scripts\Activate.ps1
```

**Import errors in tests:**  
The project sets `pythonpath = .` in `pytest.ini` — ensure tests are run from the project root.

**OCR not working:**  
Set `ENABLE_PDF_OCR_FALLBACK=true` and ensure `paddleocr` is installed. OCR is disabled by default to reduce startup time.

---

## 📞 Support

- **GitHub Issues:** [Create an issue](https://github.com/DangHuuLong/Ai-Recruiter-Mini-Ai-Service/issues)
- **Related Repos:** [Backend](https://github.com/DangHuuLong/Ai-Recruiter-Mini-Backend) · [Frontend](https://github.com/DangHuuLong/Ai-Recruiter-Mini-Frontend)

---

## 📄 License

This project is part of the AI Recruiter Mini suite.

---

<p align="center">
  Built with ❤️ by <a href="https://github.com/DangHuuLong">DangHuuLong</a> using FastAPI, Pydantic & Python
  <br/>
  <a href="https://github.com/DangHuuLong/Ai-Recruiter-Mini-Ai-Service">AI Service</a> •
  <a href="https://github.com/DangHuuLong/Ai-Recruiter-Mini-Backend">Backend</a> •
  <a href="https://github.com/DangHuuLong/Ai-Recruiter-Mini-Frontend">Frontend</a>
</p>
