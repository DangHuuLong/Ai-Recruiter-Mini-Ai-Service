# Dataset Directory

This directory stores dataset files and versioned snapshots for CV-JD matching experiments.

The dataset is used for validation, baseline evaluation, fine-tune readiness checks, and future model training work.

## Current Status

The current fine-tune-ready snapshot is:

```txt
datasets/versions/v0.2/
```

This version is built with `rubric_v0.2`, contains stable train/validation/test split assignments, and is the recommended dataset version for the next fine-tuning experiment branch.

## Directory Structure

```txt
datasets/
  README.md
  raw/
    job_descriptions.jsonl
    resumes.jsonl
  processed/
    cv_jd_pairs.jsonl
  versions/
    v0.1/
      job_descriptions.jsonl
      resumes.jsonl
      cv_jd_pairs.jsonl
    v0.2/
      job_descriptions.jsonl
      resumes.jsonl
      cv_jd_pairs.jsonl
```

## Working Dataset

The working dataset is the editable source used while adding or fixing records:

```txt
datasets/raw/job_descriptions.jsonl
datasets/raw/resumes.jsonl
datasets/processed/cv_jd_pairs.jsonl
```

Use this area when collecting, cleaning, anonymizing, or labeling new records.

Validate the current working dataset with:

```bash
python training/validate_dataset.py --dataset-root datasets
```

## Versioned Snapshots

A versioned snapshot is a stable dataset copy used for reproducible evaluation or training.

Each version folder should contain exactly the three official JSONL files:

```txt
datasets/versions/<version>/job_descriptions.jsonl
datasets/versions/<version>/resumes.jsonl
datasets/versions/<version>/cv_jd_pairs.jsonl
```

For stable training and evaluation versions, `cv_jd_pairs.jsonl` must contain the final `split` assignment directly:

```json
"split": "train"
```

Temporary files such as `cv_jd_pairs_split.jsonl` or `cv_jd_pairs_no_split.jsonl` should not be kept inside the official version folder.

Validate a versioned snapshot with:

```bash
python training/validate_dataset.py --dataset-root datasets --version v0.2
```

## Version Notes

| Version | Notes |
| --- | --- |
| `v0.1` | Initial dataset snapshot kept for historical comparison. |
| `v0.2` | Fine-tune-ready scorer-aligned snapshot using `rubric_v0.2` and stable split assignments. |

## Privacy Rule

Only anonymized or synthetic resume data should be committed.

Allowed data types:

- synthetic resumes;
- anonymized resumes;
- public or rewritten job descriptions;
- small fixture-like examples for documentation.

Additional rules:

- Keep identifying details out of committed resume text.
- Git history keeps deleted files. If sensitive data is accidentally committed, remove it from history before continuing.
- Prefer synthetic records for early validation until the anonymization process is reliable.

## Dataset Update Flow

When adding or changing dataset records, use this flow:

1. add or edit JD and resume records in `datasets/raw/`;
2. add or edit CV-JD pair records in `datasets/processed/cv_jd_pairs.jsonl`;
3. run the dataset validator against the working dataset;
4. create a new versioned snapshot when the working dataset is stable;
5. assign stable split values for the snapshot;
6. validate the versioned snapshot;
7. run the fine-tune readiness check before starting a training branch.

## Related Documentation

- `docs/dataset-schema.md`
- `docs/labeling-guide.md`
- `docs/training-strategy.md`
- `training/README.md`
