# Training Utilities

This folder contains lightweight scripts used before and during Phase 15 dataset/model training work.

At this stage, the repository includes dataset validation, split planning, baseline evaluation, and fine-tune readiness checks. It does not include real fine-tuning code yet.

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
- `criterion_scores` contains the required keys for the record's `label_version`;
- `split` is `train`, `validation`, `test`, or `null`;
- resume records are marked as anonymized;
- obvious email, phone, or personal profile URL patterns are not present in resume text.

## Plan dataset split

Plan train/validation/test split for the current working CV-JD pair dataset:

```bash
python training/split_dataset.py --dry-run
```

Plan split for a versioned snapshot:

```bash
python training/split_dataset.py --version v0.1 --dry-run
```

The split planner follows the dataset rule from `docs/dataset-schema.md`: datasets with fewer than 50 labeled pairs should keep `split = null` and should not be treated as a stable evaluation set.

Because the current sample dataset is still small, the script reports the planned split but skips writing a split file by default.

When the dataset is large enough, write a split file with:

```bash
python training/split_dataset.py --output datasets/versions/v0.2/cv_jd_pairs.jsonl
```

For local experiments only, split generation can be forced below the minimum threshold:

```bash
python training/split_dataset.py --force --output /tmp/cv_jd_pairs.split.jsonl
```

Do not commit forced split outputs as stable dataset versions.

## How split planning works

The split planner:

- reads `datasets/processed/cv_jd_pairs.jsonl` by default;
- supports `--version v0.1` for versioned snapshots;
- uses the default target ratio `70/15/15` for train/validation/test;
- groups pairs that share the same `resume_id` or `job_description_id` into the same split;
- avoids leaking the same resume or JD across train/validation/test;
- uses a deterministic random seed for repeatable planning.

## Evaluate baseline similarity

Run the current pretrained embedding baseline:

```bash
python training/evaluate_baseline_similarity.py
```

Run against a versioned snapshot:

```bash
python training/evaluate_baseline_similarity.py --version v0.1
```

The script writes a local report under `artifacts/reports/`. The `artifacts/` folder is ignored by git.

## Check fine-tune readiness

Check whether the current working dataset is ready for fine-tuning:

```bash
python training/check_fine_tune_readiness.py
```

Use strict mode in CI or before starting a real fine-tuning branch:

```bash
python training/check_fine_tune_readiness.py --strict
```

The readiness check currently requires:

- enough labeled CV-JD pairs;
- current scorer-aligned `rubric_v0.2` labels;
- no `split = null` records;
- train/validation/test splits all present;
- no resume leakage across splits;
- no JD leakage across splits.

The default minimum is 100 labeled pairs:

```bash
python training/check_fine_tune_readiness.py --min-pairs 100
```

With the current small synthetic dataset, this check is expected to report that fine-tuning is not ready yet.

## Current scope

These scripts are intentionally lightweight.

They do not save model checkpoints, fine-tune models, or export inference artifacts yet.
