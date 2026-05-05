# Evaluation and Parser Rollout Notes

## 1. Purpose

This document summarizes the implementation work from the deterministic application evaluation rollout through the latest document-based resume parser and evaluation scoring tightening changes.

Use this file as an implementation and deployment reference for Backend, Frontend, and AI service integration work. API request and response contracts are still documented in [`docs/ai-service-contract.md`](./ai-service-contract.md).

---

## 2. Rollout Timeline

### 2026-05-04 — Deterministic application evaluation

Implemented the first deterministic scoring baseline for `POST /score/application`.

Main implementation file:

```txt
app/scorers/cv_jd_scorer.py
```

Main test file:

```txt
tests/test_scoring_baseline.py
```

Key behavior added:

- Score a parsed resume against a parsed job description.
- Generate weighted criterion scores.
- Return matched, missing, and related skills.
- Generate strong points, weak points, skill gap summary, interview questions, and evidence map.
- Keep the response compatible with the `EvaluationResult` schema consumed by the Backend.

Default criteria:

| Criterion | Weight | Purpose |
| --- | ---: | --- |
| `SKILLS_MATCH` | `0.35` | Required and preferred skill overlap. |
| `EXPERIENCE_RELEVANCE` | `0.30` | Work experience duration and role-relevant evidence. |
| `PROJECT_RELEVANCE` | `0.15` | Project evidence aligned with JD skills/domain. |
| `EDUCATION_CERTIFICATION` | `0.10` | Education and certification fit. |
| `KEYWORD_DOMAIN_ALIGNMENT` | `0.10` | Meaningful keyword/domain overlap. |

Backend persistence expectations:

- Persist `overall_score`, `summary`, `explanation`, and status in `Evaluation`.
- Persist criterion rows in `EvaluationCriterionScore`.
- Persist matched/missing/partial skills in `EvaluationSkill`.
- Persist generated interview questions.
- Preserve `evidence_map` for traceability and debugging.

---

### 2026-05-05 — Document-based resume parsing

Refactored resume parsing so the AI service receives a signed file URL instead of raw resume text.

Main implementation files:

```txt
app/api/parse.py
app/schemas/resume.py
app/services/document_text_extraction_service.py
app/services/parsing_service.py
app/parsers/extractors/languages.py
app/parsers/extractors/projects_certifications.py
app/parsers/section_splitter.py
```

Main test file:

```txt
tests/test_parse_resume.py
```

Key behavior added:

- `POST /parse/resume` now accepts document metadata and a short-lived signed URL.
- AI service downloads the resume file directly from storage.
- AI service extracts text from supported document formats before parsing.
- The response includes both extracted `raw_text` and structured `parsed_data`.
- Parser metadata is returned through `parser_version`, `warnings`, `confidence`, and `text_extraction_method`.

Supported file extraction methods:

| File type | Extraction method | Notes |
| --- | --- | --- |
| `PDF` | `PDF_TEXT` | Uses PyMuPDF first, then pypdf fallback. OCR is not implemented. |
| `DOCX` | `DOCX_TEXT` | Uses `python-docx` paragraph extraction. |

Current request shape:

```json
{
  "resume_id": "res_123",
  "file_name": "candidate-cv.pdf",
  "file_type": "PDF",
  "signed_url": "https://storage.example.com/signed/resumes/res_123/cv.pdf",
  "checksum": "optional-file-checksum"
}
```

Current response shape:

```json
{
  "raw_text": "Candidate Name\nBackend Developer...",
  "parsed_data": {},
  "parser_version": "ai-document-resume-parser-v1",
  "warnings": [],
  "confidence": 0.9,
  "text_extraction_method": "PDF_TEXT"
}
```

Backend integration expectations:

1. Frontend uploads resume through Backend only.
2. Backend stores the file and creates a short-lived signed URL.
3. Backend calls `POST /parse/resume` with file metadata and the signed URL.
4. AI service downloads and extracts text.
5. AI service parses the extracted text.
6. Backend persists both `raw_text` and `parsed_data`.

Important limits:

- OCR/scanned resume parsing is not supported yet.
- Highly visual PDFs depend on selectable text extraction quality.
- The AI service should return warnings rather than silently pretending OCR succeeded.

---

### 2026-05-05 — Evaluation scoring tightening

Tightened deterministic scoring after evaluating a real application where the initial baseline produced an overly high score for a weak fit.

Main implementation file:

```txt
app/scorers/cv_jd_scorer.py
```

Problem observed:

- Candidate had approximately `0.6` years of parsed experience.
- JD required `3` years of full-stack software development experience.
- Candidate matched some skills such as React, TypeScript, PostgreSQL, REST API, Docker, and Git.
- Candidate missed critical required skills such as Node.js, NestJS, Next.js, Redis, and CI/CD.
- Initial scoring gave too much credit to project and experience overlap because generic tokens were being counted.

Fixes implemented:

- Added broader generic-token filtering for keyword, project, and experience scoring.
- Project relevance now prioritizes required skill evidence over generic text overlap.
- Project relevance is capped when most required skills are missing.
- Experience relevance now combines experience duration with required-skill evidence more strictly.
- Experience relevance is capped when parsed experience is below the JD `min_experience_years` requirement.
- Education scoring better recognizes CS/IT/related degree requirements.
- Evidence text now explains score caps, especially when experience is below the JD requirement.

Expected scoring behavior after tightening:

| Scenario | Expected behavior |
| --- | --- |
| Missing many required core skills | Overall score should remain low even if some generic domain words overlap. |
| Experience below `min_experience_years` | Experience score should be capped. |
| Project uses a different backend stack | Project score should not be high unless required JD skills are explicit. |
| IT/CS-related education is present | Education score should reflect degree-field fit. |

Example result after tightening:

```txt
SKILLS_MATCH = 0.47
EXPERIENCE_RELEVANCE = 0.21
PROJECT_RELEVANCE = 0.26
EDUCATION_CERTIFICATION = 0.75
KEYWORD_DOMAIN_ALIGNMENT = 0.29
overall_score = 36.77
summary = Weak fit
```

This is expected for a fresher/frontend-leaning profile against a mid-level full-stack JD that requires Node.js, NestJS, Next.js, Redis, CI/CD, and 3 years of experience.

---

## 3. Current AI Service Flow

```txt
Resume file upload
  ↓
Backend stores file asset
  ↓
Backend creates signed URL
  ↓
AI Service /parse/resume
  ↓
DocumentTextExtractionService downloads PDF/DOCX
  ↓
Text extraction returns raw_text + extraction metadata
  ↓
ParsingService parses raw_text into ParsedResumeData
  ↓
Backend stores raw_text + parsed_data
  ↓
Backend parses or loads ParsedJobDescriptionData
  ↓
AI Service /score/application
  ↓
cv_jd_scorer.py returns EvaluationResult
  ↓
Backend persists evaluation breakdown, skills, evidence, and questions
```

---

## 4. Deployment Checklist

Before merging or deploying related changes, verify:

- `POST /parse/resume` is called with `resume_id`, `file_name`, `file_type`, `signed_url`, and optional `checksum`.
- Signed URLs are short-lived but valid long enough for the AI service to download the file.
- Backend persists `raw_text`, `parsed_data`, `parser_version`, parse warnings, and parse status.
- Backend sends parsed resume and parsed JD data to `POST /score/application`.
- Backend persists all evaluation child rows: criteria, skills, and interview questions.
- Frontend displays weak-fit and skill-gap output clearly rather than only showing the overall score.
- Logs include enough traceability to connect resume id, application id, parser version, and evaluation result id.

Run the test suite locally:

```bash
python -m pytest
```

Focused tests:

```bash
python -m pytest tests/test_parse_resume.py
python -m pytest tests/test_scoring_baseline.py
```

---

## 5. Known Limitations and Follow-ups

### Resume parser follow-ups

- Add OCR fallback for scanned PDFs.
- Add stronger handling for complex two-column layouts.
- Expand project schema when Backend/Frontend need role, status, period, and feature fields separately.
- Add more regression tests from real-world CV layouts.

### Evaluation follow-ups

- Add calibration tests with known weak, partial, reasonable, and strong fit examples.
- Add explicit score bands to product documentation and Frontend labels.
- Consider weighting core required skills more heavily than non-core required skills.
- Consider an overall-score cap when a candidate misses more than half of core required skills.
- Keep evidence output human-readable and avoid exposing generic tokens as evidence.

Suggested score bands for current MVP:

| Score range | Label | Meaning |
| ---: | --- | --- |
| `0–39` | Weak fit | Major required skills or experience missing. |
| `40–59` | Partial fit | Some relevant evidence, but notable gaps. |
| `60–74` | Reasonable fit | Most key evidence present, some gaps remain. |
| `75–100` | Strong fit | Strong alignment with JD requirements. |

---

## 6. Ownership Notes

AI service owns:

- Resume document download from signed URL.
- Text extraction from supported document types.
- Resume parsing into `ParsedResumeData`.
- JD text parsing into `ParsedJobDescriptionData`.
- Deterministic application scoring into `EvaluationResult`.

Backend owns:

- File upload and storage.
- Signed URL generation.
- Calling AI service endpoints.
- Persistence of parse and evaluation results.
- Mapping AI service `snake_case` fields to Backend/Frontend response conventions when needed.

Frontend owns:

- Upload UX through Backend.
- Evaluation display UX.
- Showing score breakdown, missing skills, evidence, and interview questions in a way recruiters can understand.
