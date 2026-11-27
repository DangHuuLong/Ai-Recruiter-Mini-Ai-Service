# AI Recruiter Mini — AI Service

A stateless FastAPI service that turns CVs and job descriptions into structured
data and scores how well a CV fits a JD. It is called only by the
[Backend](https://github.com/DangHuuLong/Ai-Recruiter-Mini-Backend). End users
never call it directly.

Related repositories:
[Backend](https://github.com/DangHuuLong/Ai-Recruiter-Mini-Backend) ·
[Frontend](https://github.com/DangHuuLong/Ai-Recruiter-Mini-Frontend)

---

## What it does

| Capability | How |
|---|---|
| Read documents | PDF (PyMuPDF, then pypdf as fallback) and DOCX (python-docx), downloaded from a signed URL. Plain text is also accepted. |
| Repair Vietnamese text | Detects VNI/TCVN3-style glyph corruption in PDF text. Optional PaddleOCR fallback for scanned or corrupted PDFs. |
| Parse CVs | Section splitting, then extractors for contact info, links, skills, experience, projects, education, certifications, achievements and languages. |
| Parse JDs | Responsibilities, requirements, nice-to-have, benefits, required and preferred skills. |
| Score a CV–JD pair | Five weighted rule-based criteria, optionally blended with a fine-tuned cross-encoder score. Returns matched/missing skills, strengths, weaknesses and template-based interview questions. |

What it does **not** do:

- It stores nothing. There is no database, queue or cache, and every request is independent.
- It does not call any LLM at runtime. `GEMINI_API_KEY`, `GEMINI_MODEL` and
  `AI_PROVIDER` exist in the settings but no code path uses them; `AI_PROVIDER`
  is only echoed by `/health`. LLM-based interview-question generation lives in the Backend.
- It does not score a batch. The Backend orchestrates N×M matrices by calling
  `/score/application` once per pair.

---

## API

All responses use one envelope:

```json
{ "success": true, "message": "…", "data": { } }
```

Errors, including 422 validation errors, use:

```json
{ "success": false, "message": "…", "errors": [{ "field": "…", "message": "…" }] }
```

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Service name, status, version, environment |
| `POST` | `/parse/resume` | Parse a CV. Body: `resume_id`, plus either `raw_text` or `signed_url` with `file_type`/`file_name` |
| `POST` | `/parse/job-description` | Parse a JD. Body: `raw_text`, or `signed_url` with `file_type`/`file_name` |
| `POST` | `/score/application` | Score a pair. Body: `resume` (parsed CV), `job_description` (parsed JD), `config.criteria` (list of criterion and weight) |

Only `PDF` and `DOCX` files are accepted through `signed_url`. A document that
yields no text returns `422`.

Interactive docs are served by FastAPI at `/docs`. The full field-level
contract is in [docs/ai-service-contract.md](docs/ai-service-contract.md).

---

## How scoring works

```
/score/application
  ├─ rule-based scorer (always runs)        app/scorers/cv_jd_scorer.py
  │    SKILLS_MATCH 0.35 · EXPERIENCE_RELEVANCE 0.30 · PROJECT_RELEVANCE 0.15
  │    EDUCATION_CERTIFICATION 0.10 · KEYWORD_DOMAIN_ALIGNMENT 0.10
  │    (weights come from config.criteria; these are the defaults)
  │
  └─ ML similarity (only if SIMILARITY_SCORING_WEIGHT > 0)
       cross-encoder(CV text, JD text) → clip to [0, 1] × 100
       final = (1 − w) × rule_score + w × ml_score
```

- If the ML step fails or its score is below `SIMILARITY_SCORE_THRESHOLD`, the
  rule-based result is returned unchanged.
- When blending happens, the response also carries `similarity_score`, and the
  summary text is regenerated so it quotes the blended score.
- Interview questions in this service are templates filled with skill names.
  They are not model-generated.

Details: [docs/similarity-scoring-design.md](docs/similarity-scoring-design.md).

---

## Models

Three fine-tuned models are tracked with **Git LFS** under `models/`:

| Model | Used for | Architecture | Test result |
|---|---|---|---|
| `cross-encoder-cv-jd-v0.6` | CV–JD similarity | `cross-encoder/ms-marco-MiniLM-L-12-v2` (~33M parameters) fine-tuned with MSE regression | 62.50% same-label accuracy, 65.15% within ±10 points, MAE 9.56, Spearman 0.864 (2,000 test pairs) |
| `section-classifier-v0.1` | CV section splitting | MiniLM-L6 line embeddings, structural features, logistic regression, Viterbi smoothing | 86% line accuracy |
| `jd-section-classifier-v0.1` | JD section splitting | Same recipe as the CV classifier, with 5 labels | 89% line accuracy |

Each model is off unless configured (see [Configuration](#configuration)). The
service falls back to rule-based scoring and regex section splitting when a
model is disabled or fails to load. All configured models are loaded once at
startup.

**Important limitation.** The similarity model was trained on 13,350 CV–JD pairs
whose scores were produced by an LLM, not by recruiters. It imitates an LLM
scorer cheaply and offline; it has not been validated against real hiring
decisions. Training history, including failed experiments:
[v0.1](docs/similarity-model-cross-encoder-v0.1.md) →
[v0.6](docs/similarity-model-cross-encoder-v0.6.md) →
[v0.7, discontinued](docs/similarity-model-cross-encoder-v0.7.md).

---

## Getting started

Requirements: **Python 3.12**, Git with **Git LFS** (to pull the model weights).

```bash
git lfs install
git clone https://github.com/DangHuuLong/Ai-Recruiter-Mini-Ai-Service.git
cd Ai-Recruiter-Mini-Ai-Service

python -m venv .venv
# Windows: .venv\Scripts\activate   ·   macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
# Optional: local OCR fallback (heavy; Linux/Docker recommended)
# pip install -r requirements-ocr.txt

cp .env.example .env
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open <http://localhost:8000/health> to check the service.

The first start downloads `sentence-transformers/all-MiniLM-L6-v2` and the
cross-encoder base model from Hugging Face if they are not cached yet.

---

## Configuration

Settings are read from environment variables or `.env` (`app/core/config.py`).

| Variable | Default | Notes |
|---|---|---|
| `APP_NAME`, `APP_ENV`, `APP_VERSION` | `ai-recruiter-mini-ai-service`, `development`, `0.1.0` | Reported by `/health` |
| `LOG_LEVEL` | `INFO` | |
| `SIMILARITY_MODEL_PATH` | empty | e.g. `models/cross-encoder-cv-jd-v0.6`. Empty means the base model is used. |
| `SIMILARITY_BASE_MODEL` | `cross-encoder/ms-marco-MiniLM-L-12-v2` | Used when no fine-tuned path is set or the path is missing |
| `SIMILARITY_MODEL_VERSION` | empty | Label only |
| `SIMILARITY_SCORING_WEIGHT` | `0.0` | Blend weight `w`. `0` disables ML scoring. |
| `SIMILARITY_SCORE_THRESHOLD` | `0.0` | ML scores below this are ignored |
| `SIMILARITY_FALLBACK_MODE` | `base_model` | `base_model` or `rule_only` |
| `SECTION_CLASSIFIER_MODEL_PATH` | empty | e.g. `models/section-classifier-v0.1` |
| `SECTION_CLASSIFIER_FALLBACK_MODE` | `regex_only` | `regex_only` or `model_with_regex_fallback` |
| `JD_SECTION_CLASSIFIER_MODEL_PATH` | empty | e.g. `models/jd-section-classifier-v0.1` |
| `JD_SECTION_CLASSIFIER_FALLBACK_MODE` | `regex_only` | `regex_only` or `model_with_regex_fallback` |
| `ENABLE_PDF_OCR_FALLBACK` | `false` | Needs `requirements-ocr.txt` installed |
| `PDF_OCR_LANGUAGES` | `en` (`.env.example`: `en+vi`) | Tried in order until one gives clean text |
| `PDF_OCR_ENGINE`, `PDF_OCR_MODE`, `PDF_OCR_MAX_PAGES`, `PDF_OCR_MIN_CONFIDENCE` | `paddleocr`, `targeted`, `2`, `0.5` | |

`.env.example` ships with every model turned off. To run the full pipeline,
which is the setup this project uses, add:

```env
SIMILARITY_MODEL_PATH=models/cross-encoder-cv-jd-v0.6
SIMILARITY_MODEL_VERSION=v0.6
SIMILARITY_SCORING_WEIGHT=0.5
SECTION_CLASSIFIER_MODEL_PATH=models/section-classifier-v0.1
SECTION_CLASSIFIER_FALLBACK_MODE=model_with_regex_fallback
JD_SECTION_CLASSIFIER_MODEL_PATH=models/jd-section-classifier-v0.1
JD_SECTION_CLASSIFIER_FALLBACK_MODE=model_with_regex_fallback
```

---

## Docker

```bash
docker build -t ai-recruiter-ai-service .
docker run -p 8000:8000 --env-file .env ai-recruiter-ai-service
```

- The image is based on `python:3.12-slim`, installs CPU-only PyTorch, runs as
  a non-root user and has a `/health` healthcheck.
- It bakes in the three models listed above and the MiniLM-L6 encoder, so it
  needs no network access at runtime.
- PaddleOCR is **not** installed in the image, so OCR fallback is unavailable
  there. The extraction code skips it gracefully.
- Model paths are not set inside the image. Pass them at runtime, as in the
  previous section.
- The listening port is taken from `PORT` (default `8000`).

---

## Tests

```bash
pytest
```

Tests live in `tests/` and cover document extraction, CV and JD parsing
(including ML-assisted JD splitting), schemas, the error-response contract and `/health`.

---

## Project layout

```
app/
  main.py          FastAPI app, startup model warm-up, error handlers
  api/             health, parse, score routes
  core/            settings, logging
  services/        document text extraction (PDF/DOCX/OCR), parsing, scoring
  parsers/         resume/JD parsers, section splitter, skill catalog (~1,260 skills), extractors
  scorers/         rule-based CV–JD scorer, cross-encoder similarity scorer
  ml/              model loaders and configs (similarity, CV/JD section classifiers)
  schemas/         Pydantic request/response models
models/            fine-tuned weights (Git LFS)
tests/             pytest suite
training/          dataset validation/splitting, fine-tuning and evaluation scripts
notebooks/         Colab notebooks for every training experiment
datasets/          versioned CV/JD/pair datasets (v0.1–v0.6)
scripts/           synthetic data generation and LLM labelling helpers
docs/              design notes, contracts, experiment reports
```

`training/`, `notebooks/`, `datasets/` and `scripts/` are research tooling.
They are excluded from the Docker image.

---

## Deployment

Pushing to `main` triggers `.github/workflows` to build the image (with Git LFS
checkout) and push it to `ghcr.io/danghuulong/ai-recruiter-mini-ai-service`,
tagged `latest` and with the commit SHA.

Branch flow: `feature/*`, `fix/*` and `docs/*` branches merge into `develop`.
A `release/x.y.z` branch is cut from `develop` and merged into `main` to deploy.

---

## Known limitations

- **Throughput.** The cross-encoder runs one full forward pass per CV–JD pair,
  on CPU, with no caching. Large N×M batches are slow; set
  `SIMILARITY_SCORING_WEIGHT=0` to fall back to rule-based scoring only.
- **Language.** The skill catalog and training data are mostly English, so
  Vietnamese or mixed-language CVs score less reliably.
- **Synthetic training data.** See the limitation in [Models](#models).
- **No input size limits.** Downloads use a 30 s timeout but no file-size
  cap; the Backend enforces upload limits.

---

## Documentation

| Topic | File |
|---|---|
| API contract | [docs/ai-service-contract.md](docs/ai-service-contract.md) |
| Setup notes | [docs/ai-service-setup.md](docs/ai-service-setup.md) |
| Document extraction | [docs/document-text-extraction-parser.md](docs/document-text-extraction-parser.md) |
| JD parser | [docs/job-descriptions.md](docs/job-descriptions.md) |
| Scoring design | [docs/similarity-scoring-design.md](docs/similarity-scoring-design.md) |
| Model integration | [docs/model-integration-guide.md](docs/model-integration-guide.md) |
| Dataset schema and labelling | [docs/dataset-schema.md](docs/dataset-schema.md), [docs/labeling-guide.md](docs/labeling-guide.md) |
| Training | [docs/training-pipeline.md](docs/training-pipeline.md), [docs/training-strategy.md](docs/training-strategy.md), [docs/fine-tuning-colab-guide.md](docs/fine-tuning-colab-guide.md) |
| Section classifiers | [docs/section-classifier-v0.1.md](docs/section-classifier-v0.1.md), [docs/jd-section-classifier-v0.1.md](docs/jd-section-classifier-v0.1.md) |
