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

Additional rules:

- Do not commit real CVs even in folders named `test`, `example`, or `sample`.
- Git history keeps deleted sensitive files. If private data is accidentally committed, remove it from history before continuing.
- Prefer synthetic records for early validation until the anonymization process is reliable.

## How to Contribute a Sample Later

When dataset samples are introduced in a later branch, use this flow:

1. create or anonymize the resume/JD record;
2. confirm the record follows `docs/dataset-schema.md`;
3. add resume and JD records to the proper JSONL files;
4. create the CV-JD pair record;
5. run the dataset validation script when it exists;
6. update the dataset version notes if the sample belongs to a versioned dataset.

## Related Documentation

- `docs/dataset-schema.md`
- `docs/labeling-guide.md`
- `docs/training-strategy.md`
