# Document Text Extraction and Parser Flow

## 1. Purpose

This document explains how the AI service turns an uploaded resume file into structured parser output.

It is written for both:

- human readers who need to understand where file text is extracted, cleaned, and parsed;
- AI assistants or coding agents that need implementation context before modifying parser behavior.

The focus is the file-to-text-to-parser flow used by resume parsing. Job description parsing is simpler because it usually starts from plain text.

---

## 2. High-Level Pipeline

```txt
Backend uploads file to storage
  -> Backend sends signed file URL to AI service
  -> DocumentTextExtractionService downloads the file
  -> File text is extracted from PDF / DOCX / TXT
  -> Extracted raw text is sanitized
  -> Resume parser splits sections and extracts fields
  -> Pydantic schemas validate response shape
  -> Backend stores parsedData and rawText
```

Main files:

| Area | File |
| --- | --- |
| API/service orchestration | `app/services/parsing_service.py` |
| File text extraction | `app/services/document_text_extraction_service.py` |
| Resume parser orchestration | `app/parsers/resume_parser.py` |
| Resume response schema | `app/schemas/resume.py` |
| Section splitting | `app/parsers/section_splitter.py` |
| Field extractors | `app/parsers/extractors/` |
| Normalizers | `app/parsers/normalizers/` |

---

## 3. Request Input Contract

File-based resume parsing receives a `ParseResumeRequest` from `app/schemas/resume.py`.

Important fields:

| Field | Meaning |
| --- | --- |
| `resume_id` | Resume record ID from Backend |
| `file_name` | Original file name |
| `file_type` | File type such as `PDF`, `DOCX`, or text-like formats |
| `signed_url` | Temporary URL used by AI service to download the file |
| `checksum` | Optional checksum from Backend/FileAsset |

The AI service should not assume the original file exists locally. Treat the signed URL as the source of file bytes.

---

## 4. File Download and Text Extraction

`DocumentTextExtractionService` is responsible for file download and text extraction.

Expected responsibilities:

1. Download file bytes from the signed URL.
2. Choose the extraction strategy based on file type.
3. Extract text while preserving enough line structure for the downstream parser.
4. Return extraction metadata, method, and warnings.

Typical strategies:

| File type | Expected strategy |
| --- | --- |
| PDF with text layer | Extract blocks / words from the PDF text layer |
| PDF with poor or missing text layer | Use OCR fallback when enabled and supported |
| DOCX | Extract paragraphs and text runs |
| TXT-like file | Decode bytes into text |

Extraction should return readable text plus warning metadata. Extraction failures should be explicit and non-silent.

---

## 5. PDF Text Layer Handling

PDF resumes are often visually structured with columns, text boxes, headers, and wrapped lines. The extraction layer must convert visual layout into a stable reading order.

Important goals:

- Preserve candidate name and contact info near the top.
- Preserve headings such as `OBJECTIVE`, `EDUCATION`, `WORK EXPERIENCE`, and `PROJECTS`.
- Avoid mixing left-column and right-column content in two-column CVs.
- Keep project title, date, role, description, and technology lines close enough for the parser to group them.
- Preserve URLs when possible.

Common issues:

| Issue | Example | Expected handling |
| --- | --- | --- |
| Wrapped lines | `Tailwind CSS,` then `shadcn/ui` | Keep adjacent lines available for parser merge logic |
| Two-column layout | Education left, experience right | Detect columns and order text consistently |
| Broken bullets | `" Backend: Node.js` | Sanitize into a normal bullet-like line |
| Date separated from project | `Trello Clone` then `2025 - 2026` | Keep date near the project block |
| Bad glyphs | `�` or mojibake | Add warning or fallback recovery where appropriate |

---

## 6. OCR Fallback Notes

OCR is a fallback, not the default path.

Use PDF text-layer parsing first because it is faster, deterministic, and preserves selectable text. OCR should only help when the text layer is missing or poor.

Current local runtime caveat:

- PaddleOCR/PaddlePaddle CPU inference can be unstable on Windows local environments.
- Windows local runtime may skip PaddleOCR fallback and rely on PDF text-layer parsing plus fallback heuristics.
- Linux or Docker is preferred for OCR-heavy validation.

Rules for OCR changes:

1. Do not make OCR mandatory for text-layer PDFs.
2. Treat OCR failures as non-fatal warnings when usable text already exists.
3. Avoid returning empty text only because OCR failed.
4. Keep warning messages explicit so developers know whether OCR was used or skipped.

---

## 7. Parser Service Orchestration

`ParsingService` connects file extraction to domain parsing.

Resume flow:

```txt
parse_resume(request)
  -> document_text_extraction_service.extract_from_signed_url(...)
  -> _sanitize_raw_text(extraction_result.raw_text)
  -> parse_resume(raw_text)
  -> recover/fix selected fields when needed
  -> return ParseResumeResult
```

Responsibilities in this layer:

- Normalize line endings.
- Remove null bytes.
- Clean obvious PDF artifacts before parsing.
- Preserve `raw_text` in a readable form for debugging and future reprocessing.
- Merge extraction warnings into the final parse result.
- Recover `full_name` from file name only when extracted text is missing or clearly invalid.

Keep `ParsingService` as orchestration and cleanup. Put domain-specific extraction rules in `app/parsers`.

---

## 8. Raw Text Sanitization

Sanitization should make extracted text easier to parse without destroying useful structure.

Current goals:

- Normalize `\r\n` and `\r` to `\n`.
- Remove null bytes.
- Replace broken PDF bullet markers where safe.
- Trim repeated spaces per line.
- Collapse excessive blank lines.
- Keep section and item boundaries readable.

Example before cleanup:

```txt
" Backend: Node.js, Express.js


" State & API: TanStack
Query
```

Example after cleanup:

```txt
- Backend: Node.js, Express.js

- State & API: TanStack
Query
```

Avoid aggressive cleanup that removes meaningful line breaks. Resume parsing depends heavily on line order and line grouping.

---

## 9. Resume Parser Architecture

The resume parser is deterministic and rule-based.

Typical flow:

```txt
raw_text
  -> normalize_text(..., preserve_lines=True)
  -> split_sections(...)
  -> extract personal info
  -> extract skills
  -> extract education
  -> extract experience
  -> extract projects
  -> extract certifications, achievements, languages
  -> create ParsedResumeData
```

Key principle:

> Text extraction should produce readable text. Resume parser should produce structured candidate data.

Do not mix file-format concerns into `resume_parser.py` unless the problem appears in normalized text and must be handled semantically.

---

## 10. Section Splitting

Section splitting maps headings to parser sections.

Common sections:

| Section | Example headings |
| --- | --- |
| Summary / Objective | `SUMMARY`, `OBJECTIVE`, `PROFILE` |
| Education | `EDUCATION`, `ACADEMIC BACKGROUND`, `Học tập`, `Học vấn` |
| Experience | `WORK EXPERIENCE`, `EXPERIENCE`, `Career History` |
| Projects | `PROJECTS`, `Dự án` |
| Skills | `SKILL`, `SKILLS`, `TECHNICAL SKILLS` |
| Certifications | `CERTIFICATIONS`, `CERTIFICATE`, `Certicate` |
| Awards | `HONORS & AWARDS`, `ACHIEVEMENTS` |
| Languages | `LANGUAGE`, `LANGUAGES`, `Ngôn ngữ` |

When adding headings, add tests. Section splitting changes can affect multiple extractors.

---

## 11. Extractor Responsibilities

Each extractor should focus on one domain area.

| Extractor | Responsibility |
| --- | --- |
| `email_phone.py` | Email and phone extraction |
| `links.py` | LinkedIn, GitHub, portfolio links |
| `skills.py` | Skill catalog, aliases, categories, evidence |
| `education.py` | Institution, degree, field, years, GPA |
| `experience.py` | Company, role, date range, responsibilities, technologies |
| `projects.py` | Project block grouping and layout recovery |
| `projects_certifications.py` | Project block parsing, URLs, technologies, certifications |
| `achievements.py` | Awards and achievements |
| `languages.py` | Languages and proficiency, including TOEIC enrichment |

Extractor functions should return plain dictionaries. `resume_parser.py` assembles the final Pydantic schema objects.

---

## 12. Skill Extraction Guidelines

Skill extraction is catalog-based, deterministic, and explainable.

Important output fields:

| Field | Meaning |
| --- | --- |
| `name` | Display name, for example `Node.js` |
| `normalized_name` | Stable matching key, for example `nodejs` |
| `category` | Skill group, for example `backend`, `frontend`, `database` |
| `evidence` | Source text line or nearby text proving the skill |
| `level` | Optional self-described level, if present in CV |

Skill extraction should handle:

- aliases such as `ReactJS` -> `React`;
- wrapped text such as `TanStack` + `Query` on adjacent lines;
- compound lines such as `Nodejs & Expressjs : Solid understanding`;
- short aliases carefully, for example `C` should not match inside `CSS`.

Avoid adding skills without evidence from the CV text.

---

## 13. Education and GPA Guidelines

Education extraction should support English and Vietnamese resumes.

Important fields:

| Field | Meaning |
| --- | --- |
| `institution` | School/university name |
| `degree` | Bachelor, Master, PhD, etc. |
| `field_of_study` | Computer Science, Information Technology, etc. |
| `start_year` | Start year as integer |
| `end_year` | End year as integer or null for present/current |
| `gpa` | GPA value only, for example `3.96` |
| `gpa_scale` | GPA scale, for example `4.0` |
| `description` | Original education block text |

Example source:

```txt
University of Science and Technology -
Nov 2023 - Jul 2027
The University of Danang
Information Technology - Bachelor Program
Cumulative GPA: 3.96/4.0
```

Expected output:

```json
{
  "institution": "University of Science and Technology - The University of Danang",
  "degree": "Bachelor",
  "field_of_study": "Information Technology",
  "start_year": 2023,
  "end_year": 2027,
  "gpa": "3.96",
  "gpa_scale": "4.0"
}
```

---

## 14. Project Date Handling

Project dates are normalized into `start_date` and `end_date` only.

The current schema intentionally does not include extra project date metadata such as `raw_date` or `date_precision`.

Examples:

| Source date | Normalized output |
| --- | --- |
| `2025 - 2026` | `start_date = 2025-01`, `end_date = 2026-01` |
| `11/2025 02/2026` | `start_date = 2025-11`, `end_date = 2026-02` |
| `04/2026 - nay` | `start_date = 2026-04`, `end_date = present` |

Reasoning:

- Backend and scoring only need comparable start/end values.
- Keeping the project schema compact avoids noisy null fields.
- If UI later needs exact source display, it can use stored `rawText` or add a separate display helper with Backend coordination.

Do not re-add project `raw_date` or `date_precision` without a clear UI/API requirement.

---

## 15. Language Extraction and TOEIC

Language extraction should include explicit language sections and certification-derived language proficiency.

Example source:

```txt
CERTIFICATIONS
2023: TOEIC 925/990
```

Expected language enrichment:

```json
{
  "name": "English",
  "proficiency": "TOEIC 925/990"
}
```

TOEIC can still remain in `certifications`. The language enrichment is useful for filtering, matching, and candidate profile display.

---

## 16. Response Contract

The final resume response is validated by Pydantic schemas in `app/schemas/resume.py`.

Important response groups:

| Group | Description |
| --- | --- |
| `raw_text` | Cleaned extracted text used for parsing |
| `parsed_data.personal` | Name, email, phone, location, social links |
| `parsed_data.summary` | Candidate objective/summary |
| `parsed_data.skills` | Skill list with normalized names, categories, evidence, and optional level |
| `parsed_data.education` | Education blocks with GPA fields |
| `parsed_data.experience` | Work experience blocks |
| `parsed_data.projects` | Project blocks with dates, role, description, technologies, URLs |
| `parsed_data.certifications` | Certifications and issued years |
| `parsed_data.achievements` | Awards and achievements |
| `parsed_data.languages` | Languages and proficiency |

Schema stability matters because Backend stores and consumes `parsedData`. Any new field or removed field should be coordinated with Backend and covered by tests.

---

## 17. Warnings and Fallbacks

Warnings should explain non-fatal parser or extraction issues.

Examples:

- OCR fallback skipped.
- Optional OCR dependencies missing.
- Text quality is poor but text-layer extraction was still used.
- Name was recovered from file name because extracted name was invalid.

Warnings should not be used for normal parser uncertainty. Add warnings only when the service changed behavior or skipped an expected fallback.

---

## 18. Testing Guidelines

Parser changes should be covered by targeted tests.

Relevant test files:

```txt
tests/test_file_readers.py
tests/test_document_text_extraction_service.py
tests/test_parse_resume.py
tests/test_resume_parser_layouts.py
tests/test_parser_utilities.py
tests/test_schemas.py
```

Recommended commands:

```bash
python -m pytest
python -m pytest tests/test_resume_parser_layouts.py
python -m pytest tests/test_parse_resume.py
```

Good test assertions:

- `skills` contains `TanStack Query` when source text wraps `TanStack` and `Query` across adjacent lines.
- TOEIC certification enriches `languages` with English proficiency.
- Education GPA is split into `gpa` and `gpa_scale`.
- Project URLs are stored in `projects[].urls`.

Avoid tests that force outdated or undesired response shapes.

---

## 19. Guidance for AI Coding Agents

When modifying this parser, follow these rules:

1. Check the response schema first. Do not add output fields unless there is a clear product need.
2. Do not rewrite parser logic only to satisfy old tests. Re-evaluate whether the test reflects the current contract.
3. Keep extraction deterministic. Avoid non-deterministic LLM calls in this parser path unless architecture changes.
4. Preserve raw text. Sanitization should improve readability, not erase evidence.
5. Prefer small extractor changes. Keep domain rules in the relevant extractor file.
6. Add regression tests for each parser bug. Use realistic raw text snippets from CVs.
7. Use Conventional Commits, for example:
   - `fix(parser): normalize wrapped skill extraction`
   - `feat(parser): infer English proficiency from TOEIC`
   - `test(parser): cover two-column resume layout`
   - `docs(parser): document file text extraction flow`
8. Run the full test suite before merge.

---

## 20. Known Limitations

Current limitations:

- Rule-based parsing cannot perfectly understand every resume layout.
- OCR is environment-sensitive and should be validated on Linux/Docker for scanned PDFs.
- Highly visual resumes may still produce imperfect text order.
- Skill coverage depends on the local skill catalog.
- Candidate profile synchronization, such as copying parsed email/phone into Backend candidate fields, belongs to the Backend service, not this AI service.

---

## 21. Change Checklist

Before merging parser changes:

- [ ] The response schema still matches Backend expectations.
- [ ] New parser behavior has regression tests.
- [ ] Existing tests pass with `python -m pytest`.
- [ ] Warnings are clear and non-fatal.
- [ ] Commit messages follow Conventional Commits.
- [ ] README or docs are updated if behavior changed.
