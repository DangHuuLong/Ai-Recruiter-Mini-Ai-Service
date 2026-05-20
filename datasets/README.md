# Dataset Directory

This directory stores dataset design notes and future versioned data for CV-JD matching experiments.

## Current Status

This stage defines the dataset contract only. It does not include real training data yet.

## Planned Structure

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

## Privacy Rule

Do not commit private resumes, real candidate contact information, or company-confidential job descriptions.

Allowed data types:

- synthetic resumes;
- anonymized resumes;
- public or rewritten job descriptions;
- small fixture-like examples for documentation.

## Related Documentation

- `docs/dataset-schema.md`
- `docs/labeling-guide.md`
- `docs/training-strategy.md`
