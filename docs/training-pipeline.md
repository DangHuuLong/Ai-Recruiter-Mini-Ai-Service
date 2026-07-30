# Training Pipeline

## 1. Overview

This document describes the full end-to-end workflow for training, validating,
and evaluating the CV-JD similarity model, from raw labeled data to a
deployable model artifact.

```
dataset prep
    ↓
validate_dataset.py        ← check schema, field types, privacy, referential integrity
    ↓
split_dataset.py           ← assign train/validation/test (leakage-safe)
    ↓
check_fine_tune_readiness.py ← blockers: pair count, split stability, leakage
    ↓
evaluate_baseline_similarity.py ← pretrained baseline on all pairs
    ↓
fine_tune_similarity.py --max-train-samples 20  ← smoke test (CPU, local)
    ↓
fine_tune_similarity.py (full)  ← GPU, Colab / Kaggle
    ↓
evaluate_similarity_pipeline.py  ← benchmark fine-tuned vs base on test split
```

Each stage has a gate. Do not proceed to the next stage if the current stage
reports blockers.

---

## 2. Directory Structure

```
datasets/
  versions/
    v0.2/
      job_descriptions.jsonl
      resumes.jsonl
      cv_jd_pairs.jsonl       ← pairs with split, label, overall_score assigned
training/
  validate_dataset.py
  split_dataset.py
  check_fine_tune_readiness.py
  evaluate_baseline_similarity.py
  evaluate_similarity_pipeline.py
  fine_tune_similarity.py
  baseline_models.json
models/
  fine-tuned-miniLM-v0.2-ep3/ ← not committed to git, download from Google Drive
artifacts/
  reports/                    ← generated reports, not committed to git
```

---

## 3. Stage 1 — Dataset Preparation

Before running any script, the dataset files must exist and follow the schema
defined in [dataset-schema.md](dataset-schema.md).

Required files per version:

| File | Contents |
|------|----------|
| `job_descriptions.jsonl` | One JD per line. Required: `id, source, title, level, raw_text, responsibilities, requirements, required_skills, preferred_skills, min_experience_years, language` |
| `resumes.jsonl` | One resume per line. Required: `id, source, candidate_level, raw_text, skills, normalized_skills, experience_years, education, projects, anonymized, language`. `anonymized` must be `true`. |
| `cv_jd_pairs.jsonl` | One pair per line. Required: `id, resume_id, job_description_id, split, label, overall_score, criterion_scores, matched_skills, missing_required_skills, label_notes, label_version`. Uses `rubric_v0.2`. |

Rubric v0.2 criterion keys: `SKILLS_MATCH`, `EXPERIENCE_RELEVANCE`,
`PROJECT_RELEVANCE`, `EDUCATION_CERTIFICATION`, `KEYWORD_DOMAIN_ALIGNMENT`.

Label mapping:

| `overall_score` range | `label` |
|-----------------------|---------|
| 0 – 39 | `poor_match` |
| 40 – 59 | `weak_match` |
| 60 – 74 | `moderate_match` |
| 75 – 89 | `strong_match` |
| 90 – 100 | `excellent_match` |

See [labeling-guide.md](labeling-guide.md) for rubric details.

---

## 4. Stage 2 — Validate Dataset

Validates field types, allowed values, referential integrity (resume_id /
job_description_id exist), label–score consistency, and privacy patterns
(email, phone, LinkedIn/GitHub URLs in resume text).

```bash
python training/validate_dataset.py --dataset-root datasets --version v0.2
```

**Pass condition**: `Dataset validation passed.` — zero errors.

Warnings (empty lines, extra criterion keys) do not block the pipeline but
should be investigated.

Common errors and fixes:

| Error | Fix |
|-------|-----|
| `missing required fields` | Add the missing field to the JSONL record |
| `label 'strong_match' does not match overall_score 55` | Fix score or label to be consistent |
| `anonymized must be true` | Anonymize the resume before committing |
| `contains an email-like pattern` | Remove the PII from `raw_text` |
| `resume_id does not exist` | Fix the `resume_id` reference in the pair record |

---

## 5. Stage 3 — Assign Splits

Assigns `train` / `validation` / `test` to each pair using a **component-based
strategy**: pairs are grouped into connected components by shared `resume_id`
or `job_description_id`. Entire components are assigned to a single split,
preventing resume/JD leakage across splits.

Default ratio: `70 / 15 / 15`. Default seed: `42`.

**Dry run (preview only, no file written):**

```bash
python training/split_dataset.py \
  --version v0.2 \
  --dry-run
```

**Write split assignments:**

```bash
python training/split_dataset.py \
  --version v0.2 \
  --output datasets/versions/v0.2/cv_jd_pairs.jsonl
```

This overwrites `cv_jd_pairs.jsonl` with `split` fields filled in. Run only
when the dataset is stable — changing splits invalidates any comparisons
already made on the old split.

The script will not write splits if the pair count is below `--min-pairs` (default
50) unless `--force` is passed. `--force` is for local experiments only.

---

## 6. Stage 4 — Fine-Tune Readiness Check

Checks that the dataset meets all requirements before fine-tuning begins.

```bash
python training/check_fine_tune_readiness.py \
  --dataset-root datasets \
  --version v0.2
```

**Blockers** (must all be resolved before training):

| Blocker | Meaning |
|---------|---------|
| `Only N labeled pairs found; at least 100 recommended` | Collect more labeled pairs |
| `Dataset should use only rubric_v0.2` | Migrate all pairs to rubric_v0.2 |
| `N pairs still have split=null` | Run `split_dataset.py` first |
| `Missing required split(s)` | All three splits must be present |
| `Resume leakage detected` | Fix split assignment — resume appears in >1 split |
| `Job description leakage detected` | Fix split assignment — JD appears in >1 split |

**Warnings** (do not block, but investigate):

| Warning | Meaning |
|---------|---------|
| `Only N label class(es) found` | Low diversity — model may overfit to majority class |
| `Split 'test' has low label diversity` | Test split may not reflect real distribution |

---

## 7. Stage 5 — Baseline Evaluation

Evaluates the pretrained base model on all labeled pairs. This establishes the
baseline MAE/RMSE to beat. Run once per dataset version.

```bash
python training/evaluate_baseline_similarity.py \
  --dataset-root datasets \
  --version v0.2
```

v0.2 baseline result (for reference):

| Metric | Value |
|--------|-------|
| Model | `sentence-transformers/all-MiniLM-L6-v2` |
| Pairs | 2,275 |
| MAE | 17.6431 |
| RMSE | 22.4497 |
| Label accuracy | 28.22% |

The baseline report is written to `artifacts/reports/baseline_similarity_report.json`.
Fine-tuning is only considered successful if it beats this on the test split.

---

## 8. Stage 6 — Smoke Test (Local, CPU)

Runs one epoch on a tiny subset to verify the full training loop works before
committing GPU time on Colab.

```bash
python training/fine_tune_similarity.py \
  --dataset-root datasets \
  --version v0.2 \
  --output-dir artifacts/models/debug-miniLM-v0.2 \
  --report-path artifacts/reports/debug_fine_tune_v0.2.json \
  --epochs 1 \
  --batch-size 2 \
  --max-train-samples 20 \
  --max-eval-samples 20
```

**Pass condition**: script completes without error and writes a report JSON.
Metrics at this scale are meaningless — only verify no crash.

---

## 9. Stage 7 — Fine-Tuning (GPU, Colab)

Full training run. Designed for Google Colab or Kaggle with GPU. Run one epoch
at a time and evaluate after each epoch to detect overfitting early.

**Epoch 1:**

```bash
python training/fine_tune_similarity.py \
  --dataset-root datasets \
  --version v0.2 \
  --base-model sentence-transformers/all-MiniLM-L6-v2 \
  --output-dir artifacts/models/fine-tuned-miniLM-v0.2-ep1 \
  --report-path artifacts/reports/fine_tune_v0.2_ep1.json \
  --epochs 1 \
  --batch-size 8
```

Repeat with `--epochs 1` and incrementing `--output-dir` / `--report-path` for
each subsequent epoch. Stop when validation MAE stops improving.

**Why train one epoch at a time:**
The optimal stopping epoch is not known in advance. Starting with a fixed count
like 10 epochs risks spending GPU hours past the optimal point. Checking after
each epoch lets you stop as soon as overfitting is detected (validation MAE
increases relative to the previous epoch).

**v0.2 training history:**

| Epoch | Validation MAE | Decision |
|-------|---------------|----------|
| Baseline | 17.6431 | reference |
| 2 | ~13.x | continue |
| 3 | 10.88 | **best** |
| 4 | > ep3 | stopped — overfitting |

Checkpoint for epoch 3 was kept as the final artifact.

For full training methodology and results see
[similarity-model-v0.2.md](similarity-model-v0.2.md).

---

## 10. Stage 8 — Benchmark on Test Split

After selecting the best epoch checkpoint, run the comparison pipeline on the
held-out test split. The test split must not have been used for any model
selection decisions.

```bash
python -m training.evaluate_similarity_pipeline \
  --models base:sentence-transformers/all-MiniLM-L6-v2 \
           v0.2-ep3:models/fine-tuned-miniLM-v0.2-ep3 \
  --output artifacts/reports/comparison_v0.2.json
```

v0.2 test split benchmark (350 pairs):

| Model | MAE | RMSE | Label Acc | ΔMAE |
|-------|-----|------|-----------|------|
| base | 20.3240 | 25.4998 | 32.3% | baseline |
| v0.2-ep3 | 11.9164 | 17.4966 | 47.7% | **−41.4%** |

**Pass condition**: fine-tuned MAE < baseline MAE on the test split.

---

## 11. Artifacts

Generated outputs are excluded from git via `.gitignore`. Store them locally
or in Google Drive.

| Path | Contents | In git? |
|------|----------|---------|
| `models/fine-tuned-miniLM-v0.2-ep3/` | Final model checkpoint | No |
| `artifacts/reports/baseline_similarity_report.json` | Baseline evaluation | No |
| `artifacts/reports/fine_tune_v0.2_epN.json` | Per-epoch training report | No |
| `artifacts/reports/comparison_v0.2.json` | Final benchmark comparison | No |
| `datasets/versions/v0.2/` | Labeled dataset snapshot | Yes (via git) |

Model artifacts should be downloaded from Google Drive before running inference.
See [model-integration-guide.md](model-integration-guide.md) for download and
config setup.

---

## 12. Starting a New Experiment Version

When beginning a new experiment (e.g., v0.3 with more data or a larger base model):

1. Create a new dataset version under `datasets/versions/v0.3/`.
2. Run stages 2–4 on the new version before touching training code.
3. Create a new experiment branch (`experiment/fine-tune-similarity-colab-v0.3`).
4. Keep artifacts under version-specific paths (`fine-tuned-miniLM-v0.3-epN`).
5. Compare against the v0.2 test split result as the reference baseline.

Do not retrain on an existing test split. If the dataset grows, re-run
`split_dataset.py` to generate a new stable split, then treat all prior
comparisons as invalidated.
