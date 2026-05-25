# Training Utilities

This folder contains lightweight scripts used before and during Phase 15 dataset/model training work.

At this stage, the repository only includes dataset validation. No model training or fine-tuning code is added yet.

## Validate dataset

Validate the current working dataset:

```bash
python training/validate_dataset.py
```

This checks:

- `datasets/raw/job_descriptions.jsonl`
- `datasets/raw/resumes.jsonl`
- `datasets/processed/cv_jd_pairs.jsonl`

Validate a versioned snapshot:

```bash
python training/validate_dataset.py --version v0.1
```

This checks:

- `datasets/versions/v0.1/job_descriptions.jsonl`
- `datasets/versions/v0.1/resumes.jsonl`
- `datasets/versions/v0.1/cv_jd_pairs.jsonl`

## What the validator checks

The validator currently checks that:

- every JSONL line is valid JSON;
- every line is a JSON object;
- required fields exist for JD, resume, and CV-JD pair records;
- duplicate IDs are reported;
- `resume_id` and `job_description_id` references exist;
- `overall_score` is an integer from 0 to 100;
- `label` matches the score range from `docs/dataset-schema.md`;
- `criterion_scores` contains the required rubric keys;
- `split` is `train`, `validation`, `test`, or `null`;
- resume records are marked as anonymized;
- obvious email, phone, or personal profile URL patterns are not present in resume text.

## Current scope

This script is intentionally dependency-free and uses only the Python standard library.

It does not train models, split datasets, create embeddings, or export artifacts.
