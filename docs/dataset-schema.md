# CV-JD Dataset Schema

## 1. Purpose

This document defines the first dataset contract for training and evaluating CV-JD matching models in the AI service.

This stage is limited to dataset design and labeling rules. It does not introduce model training, fine-tuning, or inference code yet.

## 2. Dataset Scope

The dataset represents pairs between one parsed or raw resume and one parsed or raw job description.

Each pair should answer one question:

> How suitable is this candidate resume for this job description?

The dataset should support three future use cases:

1. baseline similarity evaluation;
2. supervised scoring or ranking experiments;
3. future fine-tuning when enough labeled pairs exist.

## 3. Parser Output Mapping

Dataset records should be derived from the current AI service parser outputs whenever possible. Raw text should still be kept because parser logic can improve over time and old dataset versions may need to be reprocessed.

The schema uses stable dataset field names. If the parser output uses a different name, transform it during dataset preparation instead of changing the dataset field name casually.

### Resume Parser Mapping

| AI Service Parser Output | Dataset Field | Notes |
| --- | --- | --- |
| `raw_text` | `raw_text` | Cleaned text returned by resume parsing flow |
| `parsed_data.summary` | `summary` | Use `null` if summary is missing |
| `parsed_data.skills[].name` | `skills` | Use display names for human-readable records |
| `parsed_data.skills[].normalized_name` | `normalized_skills` | Recommended for matching and validation |
| `parsed_data.education[]` | `education` | Keep the most relevant education block or a compact list if needed |
| `parsed_data.experience[]` | `experience` | Preserve role/company/date evidence when available |
| `parsed_data.projects[]` | `projects` | Preserve project descriptions and technologies |
| `parsed_data.certifications[]` | `certifications` | Keep certification name and issued year if available |
| `parsed_data.languages[]` | `languages` | Keep language name and proficiency |
| derived from experience/projects | `experience_years` | Use `null` until a reliable derivation rule exists |
| derived from parsed level or manual review | `candidate_level` | `intern`, `junior`, `middle`, `senior`, or `unknown` |

### Job Description Parser Mapping

| AI Service Parser Output | Dataset Field | Notes |
| --- | --- | --- |
| `raw_text` | `raw_text` | Original or cleaned JD text |
| `parsed_data.title` | `title` | Job title |
| `parsed_data.responsibilities[]` | `responsibilities` | Raw responsibility bullet list |
| `parsed_data.requirements[]` | `requirements` | Raw requirement bullet list |
| `parsed_data.nice_to_have[]` | `nice_to_have` | Raw optional requirement bullet list |
| `parsed_data.required_skills[]` | `required_skills` | Normalized required skill names |
| `parsed_data.preferred_skills[]` | `preferred_skills` | Normalized preferred skill names |
| `parsed_data.min_experience_years` | `min_experience_years` | Number or `null` |
| `parsed_data.education_requirement` | `education_requirement` | Text or `null` |
| `parsed_data.domain_keywords[]` | `domain_keywords` | Used for domain relevance scoring |

If a parser field is unavailable, use `null` for scalar fields and `[]` for list fields. Do not invent values unless the record is explicitly marked as `synthetic`.

## 4. Storage Format

Use JSON Lines (`.jsonl`) for versioned dataset files. Each line must be a valid JSON object.

Recommended files:

```txt
datasets/
├── README.md
├── raw/
│   ├── job_descriptions.jsonl
│   └── resumes.jsonl
├── processed/
│   └── cv_jd_pairs.jsonl
└── versions/
    └── v0.1/
        ├── job_descriptions.jsonl
        ├── resumes.jsonl
        └── cv_jd_pairs.jsonl
```

Do not commit private, real, or personally identifiable CV data unless it is anonymized.

## 5. Job Description Record

Each JD record should contain enough raw text and extracted fields for matching.

```json
{
  "id": "jd_001",
  "source": "sample",
  "title": "Backend Developer Intern",
  "level": "intern",
  "employment_type": "internship",
  "location": "Da Nang, Vietnam",
  "remote_allowed": false,
  "posted_date": null,
  "domain": "recruitment_platform",
  "raw_text": "...",
  "responsibilities": ["Build REST APIs", "Integrate PostgreSQL"],
  "requirements": ["Python", "FastAPI", "SQL"],
  "nice_to_have": ["Docker", "AWS"],
  "required_skills": ["Python", "FastAPI", "PostgreSQL"],
  "preferred_skills": ["Docker", "AWS"],
  "min_experience_years": 0,
  "education_requirement": "Information Technology or related field",
  "domain_keywords": ["backend", "api", "database"],
  "language": "en"
}
```

`requirements` and `responsibilities` should preserve readable raw bullet content. `required_skills` and `preferred_skills` should contain normalized skill names extracted from those bullets.

### Required Fields

| Field | Required | Notes |
| --- | --- | --- |
| `id` | yes | Stable identifier, for example `jd_001` |
| `source` | yes | `sample`, `synthetic`, `public`, or `anonymized` |
| `title` | yes | Job title from source JD |
| `level` | yes | `intern`, `junior`, `middle`, `senior`, or `unknown` |
| `raw_text` | yes | Original JD text after privacy cleanup |
| `responsibilities` | yes | Raw or lightly cleaned responsibility list |
| `requirements` | yes | Raw or lightly cleaned requirement list |
| `required_skills` | yes | Normalized skill names required by the job |
| `preferred_skills` | yes | Optional skills that improve fit |
| `min_experience_years` | yes | Number or `null` |
| `language` | yes | `en`, `vi`, or `mixed` |

### Optional but Recommended Fields

| Field | Notes |
| --- | --- |
| `location` | Useful for future filtering, but not required for early scoring |
| `remote_allowed` | Useful when JD includes work mode |
| `posted_date` | Useful for filtering outdated JD samples |
| `employment_type` | Internship, full-time, part-time, contract |
| `domain` | Compact business or technical domain label |
| `education_requirement` | Text requirement or `null` |
| `domain_keywords` | Keywords used for domain relevance scoring |

## 6. Resume Record

Each resume record should contain raw text plus parsed fields that are already available from the resume parser.

```json
{
  "id": "resume_001",
  "source": "synthetic",
  "candidate_level": "intern",
  "raw_text": "...",
  "summary": "Final-year IT student with FastAPI experience.",
  "skills": ["Python", "FastAPI", "PostgreSQL", "React"],
  "normalized_skills": ["python", "fastapi", "postgresql", "react"],
  "experience_years": 0.5,
  "education": {
    "degree": "Bachelor",
    "field_of_study": "Information Technology"
  },
  "projects": [
    {
      "name": "AI Recruiter Mini",
      "description": "Recruitment management platform with CV parsing and scoring.",
      "technologies": ["Python", "FastAPI", "PostgreSQL"]
    }
  ],
  "certifications": [],
  "languages": [
    {
      "name": "English",
      "proficiency": "TOEIC 925/990"
    }
  ],
  "anonymized": true,
  "language": "en"
}
```

### Required Fields

| Field | Required | Notes |
| --- | --- | --- |
| `id` | yes | Stable identifier, for example `resume_001` |
| `source` | yes | `synthetic`, `anonymized`, or `sample` |
| `candidate_level` | yes | `intern`, `junior`, `middle`, `senior`, or `unknown` |
| `raw_text` | yes | Anonymized resume text |
| `skills` | yes | Human-readable skill names |
| `normalized_skills` | yes | Stable skill keys used for matching |
| `experience_years` | yes | Number or `null` if not reliably derived |
| `education` | yes | Object, list, or `null` depending on parser output |
| `projects` | yes | List, empty if unavailable |
| `anonymized` | yes | Must be `true` for real resumes committed to the repo |
| `language` | yes | `en`, `vi`, or `mixed` |

### Privacy Requirements

Before a resume is added to the dataset, remove or replace:

- full name;
- phone number;
- email address;
- home address;
- profile URLs that identify a real person;
- company confidential information.

Use synthetic IDs such as `resume_001` instead of real candidate IDs.

## 7. CV-JD Pair Record

Pair records are the primary labeled dataset for model evaluation and later training.

```json
{
  "id": "pair_001",
  "resume_id": "resume_001",
  "job_description_id": "jd_001",
  "split": null,
  "label": "strong_match",
  "overall_score": 82,
  "criterion_scores": {
    "skill_match": 85,
    "experience_match": 75,
    "education_match": 80,
    "domain_relevance": 70,
    "nice_to_have": 60
  },
  "matched_skills": ["Python", "FastAPI", "PostgreSQL"],
  "missing_required_skills": [],
  "matched_preferred_skills": [],
  "label_notes": "Strong backend match for an intern role. Missing Docker/AWS is acceptable because those are preferred skills.",
  "labeled_by": "manual",
  "label_version": "rubric_v0.1"
}
```

### Required Pair Fields

| Field | Required | Notes |
| --- | --- | --- |
| `id` | yes | Stable pair ID |
| `resume_id` | yes | Must exist in `resumes.jsonl` |
| `job_description_id` | yes | Must exist in `job_descriptions.jsonl` |
| `split` | yes | `train`, `validation`, `test`, or `null` before split assignment |
| `label` | yes | Match class derived from score range |
| `overall_score` | yes | Integer from 0 to 100 |
| `criterion_scores` | yes | Score breakdown by rubric |
| `matched_skills` | yes | Required skills found in the resume |
| `missing_required_skills` | yes | Required skills not found or not evidenced |
| `label_notes` | yes | Short reason for the label |
| `label_version` | yes | Rubric version used to label the pair |

## 8. Label Classes

| Score Range | Label | Meaning |
| --- | --- | --- |
| 90-100 | `excellent_match` | Very strong fit, missing almost nothing important |
| 75-89 | `strong_match` | Good fit, minor gaps acceptable for the level |
| 60-74 | `moderate_match` | Usable fit, clear gaps exist |
| 40-59 | `weak_match` | Some overlap but not enough for a reliable shortlist |
| 0-39 | `poor_match` | Mostly mismatched or lacks core requirements |

## 9. Dataset Split Rule

Before the dataset reaches at least 50 labeled pairs, keep `split` as `null`. The early dataset is still exploratory and should not be treated as a stable evaluation set.

After the dataset reaches at least 50 labeled pairs and the label rubric is stable, assign splits with the recommended ratio:

```txt
train: 70%
validation: 15%
test: 15%
```

Avoid data leakage:

- Split by `resume_id` and `job_description_id`, not only by pair ID.
- Do not place the same resume in both train and test.
- Do not place the same JD in both validation and test.
- Keep near-duplicate resumes or near-duplicate JDs in the same split.
- Keep test labels stable once evaluation starts.

## 10. Validation Requirements

Future validation scripts should check at least:

- every JSONL line is valid JSON;
- all required fields exist;
- `overall_score` is between 0 and 100;
- `label` matches the expected score range;
- `criterion_scores` contains all rubric keys;
- `resume_id` and `job_description_id` references exist;
- `split` is one of `train`, `validation`, `test`, or `null`;
- anonymized resume records do not contain obvious email or phone patterns;
- no duplicate IDs exist within each file.

## 11. Versioning Rule

Use semantic-like dataset versions:

```txt
v0.1  Initial manually labeled dataset
v0.2  More labels, same schema/rubric
v1.0  Stable schema and rubric for baseline comparison
```

Schema or rubric changes must be documented before the dataset version is updated.

If the rubric changes, keep old pairs with their original `label_version`. Re-label only when the new version is intended to replace the previous evaluation baseline.

## 12. Phase Boundary

This design stage may add documentation and dataset folder structure only.

Do not add these items in this branch:

- training scripts;
- model files;
- embedding cache;
- fine-tuning code;
- inference endpoint changes.
