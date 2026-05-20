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

## 3. Storage Format

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

## 4. Job Description Record

Each JD record should contain enough raw text and extracted fields for matching.

```json
{
  "id": "jd_001",
  "source": "sample",
  "title": "Backend Developer Intern",
  "level": "intern",
  "employment_type": "internship",
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

### Required Fields

| Field | Required | Notes |
| --- | --- | --- |
| `id` | yes | Stable identifier, for example `jd_001` |
| `title` | yes | Job title from source JD |
| `level` | yes | `intern`, `junior`, `middle`, `senior`, or `unknown` |
| `raw_text` | yes | Original JD text after privacy cleanup |
| `required_skills` | yes | Normalized skill names required by the job |
| `preferred_skills` | yes | Optional skills that improve fit |
| `min_experience_years` | yes | Number or `null` |
| `language` | yes | `en`, `vi`, or `mixed` |

## 5. Resume Record

Each resume record should contain raw text plus parsed fields that are already available from the resume parser.

```json
{
  "id": "resume_001",
  "source": "synthetic",
  "candidate_level": "intern",
  "raw_text": "...",
  "summary": "Final-year IT student with FastAPI experience.",
  "skills": ["Python", "FastAPI", "PostgreSQL", "React"],
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

### Privacy Requirements

Before a resume is added to the dataset, remove or replace:

- full name;
- phone number;
- email address;
- home address;
- profile URLs that identify a real person;
- company confidential information.

Use synthetic IDs such as `resume_001` instead of real candidate IDs.

## 6. CV-JD Pair Record

Pair records are the primary labeled dataset for model evaluation and later training.

```json
{
  "id": "pair_001",
  "resume_id": "resume_001",
  "job_description_id": "jd_001",
  "split": "train",
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
| `label` | yes | Match class derived from score range |
| `overall_score` | yes | Integer from 0 to 100 |
| `criterion_scores` | yes | Score breakdown by rubric |
| `label_notes` | yes | Short reason for the label |
| `label_version` | yes | Rubric version used to label the pair |

## 7. Label Classes

| Score Range | Label | Meaning |
| --- | --- | --- |
| 90-100 | `excellent_match` | Very strong fit, missing almost nothing important |
| 75-89 | `strong_match` | Good fit, minor gaps acceptable for the level |
| 60-74 | `moderate_match` | Usable fit, clear gaps exist |
| 40-59 | `weak_match` | Some overlap but not enough for a reliable shortlist |
| 0-39 | `poor_match` | Mostly mismatched or lacks core requirements |

## 8. Dataset Split Rule

Use the split field only after the initial dataset is stable.

Recommended ratio:

```txt
train: 70%
validation: 15%
test: 15%
```

Avoid data leakage:

- Do not place near-duplicate resumes across train and test.
- Do not place near-duplicate JDs across validation and test.
- Keep test labels stable once evaluation starts.

## 9. Versioning Rule

Use semantic-like dataset versions:

```txt
v0.1  Initial manually labeled dataset
v0.2  More labels, same schema/rubric
v1.0  Stable schema and rubric for baseline comparison
```

Schema or rubric changes must be documented before the dataset version is updated.

## 10. Phase Boundary

This design stage may add documentation and dataset folder structure only.

Do not add these items in this branch:

- training scripts;
- model files;
- embedding cache;
- fine-tuning code;
- inference endpoint changes.
