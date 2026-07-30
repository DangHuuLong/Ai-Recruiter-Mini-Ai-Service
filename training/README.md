# Training Utilities

This folder contains lightweight scripts used before and during Phase 15 dataset/model training work.

At this stage, the repository includes dataset validation, split planning, baseline evaluation, and fine-tune readiness checks. It does not include real fine-tuning code yet.

## Validate dataset

Validate the current working dataset:

```bash
python training/validate_dataset.py --dataset-root datasets
```

This checks:

- `datasets/raw/job_descriptions.jsonl`
- `datasets/raw/resumes.jsonl`
- `datasets/processed/cv_jd_pairs.jsonl`

Validate a versioned snapshot:

```bash
python training/validate_dataset.py --dataset-root datasets --version v0.2
```

This checks:

- `datasets/versions/v0.2/job_descriptions.jsonl`
- `datasets/versions/v0.2/resumes.jsonl`
- `datasets/versions/v0.2/cv_jd_pairs.jsonl`

Use the working dataset command while records are still being edited. Use the versioned snapshot command before baseline evaluation or fine-tuning experiments.

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
python training/split_dataset.py --dataset-root datasets --version v0.2 --dry-run
```

The split planner follows the dataset rule from `docs/dataset-schema.md`: datasets with fewer than 50 labeled pairs should keep `split = null` and should not be treated as a stable evaluation set.

## Write stable split assignments

When the versioned dataset is large enough and the dry run looks correct, write split assignments into the official pair file:

```bash
python training/split_dataset.py --dataset-root datasets --version v0.2 --output datasets/versions/v0.2/cv_jd_pairs.jsonl
```

This command overwrites `datasets/versions/v0.2/cv_jd_pairs.jsonl` with records that include stable `split` values.

Do not keep temporary files such as `cv_jd_pairs_split.jsonl` or `cv_jd_pairs_no_split.jsonl` inside the official version folder. A stable version should expose one official pair file only.

For local experiments only, split generation can be forced below the minimum threshold:

```bash
python training/split_dataset.py --force --output /tmp/cv_jd_pairs.split.jsonl
```

Do not commit forced split outputs as stable dataset versions.

## How split planning works

The split planner:

- reads `datasets/processed/cv_jd_pairs.jsonl` by default;
- supports `--version v0.2` for versioned snapshots;
- uses the default target ratio `70/15/15` for train/validation/test;
- groups pairs that share the same `resume_id` or `job_description_id` into the same split;
- avoids leaking the same resume or JD across train/validation/test;
- uses a deterministic random seed for repeatable planning.

## Check fine-tune readiness

Check whether the current working dataset is ready for fine-tuning:

```bash
python training/check_fine_tune_readiness.py --dataset-root datasets
```

Check a versioned snapshot:

```bash
python training/check_fine_tune_readiness.py --dataset-root datasets --version v0.2
```

Use strict mode in CI or before starting a real fine-tuning branch:

```bash
python training/check_fine_tune_readiness.py --dataset-root datasets --version v0.2 --strict
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

The `v0.2` snapshot is expected to pass this check after split assignments are written into `datasets/versions/v0.2/cv_jd_pairs.jsonl`.

## Evaluate baseline similarity

Run the current pretrained embedding baseline against a versioned snapshot:

```bash
python training/evaluate_baseline_similarity.py --dataset-root datasets --version v0.2 --output artifacts/reports/baseline_similarity_v0.2_report.json
```

The default primary model is configured in `training/baseline_models.json`. At the time of the `v0.2` baseline, the selected model is:

```txt
sentence-transformers/all-MiniLM-L6-v2
```

The first `v0.2` baseline result was:

```txt
pairs evaluated: 2275
MAE: 17.6431
RMSE: 22.4497
label accuracy: 0.2822
```

Future fine-tuning experiments should beat this baseline on the stable `v0.2` split before being treated as an improvement.

## Generated artifacts

The baseline script writes reports under `artifacts/reports/`.

The repository currently ignores the full `artifacts/` folder, so generated reports, model checkpoints, embedding caches, and exported model files are local artifacts by default.

Do not commit large generated artifacts unless the project explicitly changes the artifact policy.

## Current scope

These scripts are intentionally lightweight.

They do not save model checkpoints, fine-tune models, or export inference artifacts yet.
