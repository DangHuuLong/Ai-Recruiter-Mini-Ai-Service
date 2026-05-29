# Training Strategy

## 1. Purpose

This document records the training direction for the CV-JD matching model after the baseline scoring flow is stable.

The current repository state prepares the dataset, validation scripts, split planning, baseline evaluation, and fine-tune readiness checks. Actual fine-tuning scripts should be added in a separate experiment branch.

## 2. Recommended Order

Do not fine-tune first. Follow this order:

1. define dataset schema;
2. define labeling rubric;
3. create a clean labeled dataset;
4. validate dataset quality;
5. split train/validation/test without leakage;
6. evaluate pretrained embedding baseline;
7. add feature-based scoring if needed;
8. fine-tune only when the dataset passes readiness checks;
9. export the best model artifact for local/service inference.

## 3. Current Dataset Baseline

The current fine-tune-ready dataset snapshot is:

```txt
datasets/versions/v0.2/
```

This snapshot uses `rubric_v0.2`, has stable train/validation/test split assignments, and should be used as the starting point for the next fine-tuning experiment.

Validate it before using it:

```bash
python training/validate_dataset.py --dataset-root datasets --version v0.2
python training/check_fine_tune_readiness.py --dataset-root datasets --version v0.2
```

## 4. Initial Baseline

The first baseline should be simple and explainable:

```txt
resume_text -> embedding
job_description_text -> embedding
similarity = cosine(resume_embedding, jd_embedding)
```

Suggested lightweight model for first English-first experiment:

```txt
sentence-transformers/all-MiniLM-L6-v2
```

This model is suitable for a local baseline because it is lightweight and easy to run.

Note: `all-MiniLM-L6-v2` is mainly suitable for English text. If the dataset includes Vietnamese or mixed-language resumes/JDs, compare it with a multilingual baseline such as:

```txt
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

## 5. v0.2 Baseline Result

The first `v0.2` pretrained embedding baseline result is:

```txt
model: sentence-transformers/all-MiniLM-L6-v2
pairs evaluated: 2275
MAE: 17.6431
RMSE: 22.4497
label accuracy: 0.2822
```

Future fine-tuning experiments should beat this baseline on the stable `v0.2` split before being treated as an improvement.

## 6. Evaluation Metrics

Use different metrics depending on the target.

| Target | Suggested Metrics |
| --- | --- |
| Score regression | MAE, RMSE |
| Match class classification | Accuracy, macro F1 |
| Candidate ranking | Spearman correlation, Precision@K, Recall@K |

For recruitment screening, ranking metrics are important because the service often needs to order candidates by fit, not only predict one score.

## 7. Data Leakage Prevention

Dataset splitting must avoid leakage before any baseline result is trusted.

Rules:

- Split by `resume_id` and `job_description_id`, not only by pair ID.
- One resume must not appear in both train and test.
- One JD must not appear in both validation and test.
- Keep near-duplicate resumes in the same split.
- Keep near-duplicate JDs in the same split.
- Assign split first, then shuffle within each split.
- Do not tune thresholds or choose models using the test split.

If these rules are hard to satisfy because the dataset is still small, keep `split` as `null` and treat the dataset as exploratory only.

## 8. Feature-Based Model Direction

Before fine-tuning deep models, consider a feature-based model using parser outputs.

Potential features:

- embedding similarity;
- required skill coverage;
- preferred skill coverage;
- missing required skill count;
- experience gap;
- education match;
- domain keyword overlap.

This direction is practical when the dataset is still small or when explainability is more important than raw model capacity.

## 9. Experiment Logging

For a mini project, heavyweight tracking tools are optional. A simple JSON or CSV report is enough for early baselines.

Recommended local report path:

```txt
artifacts/reports/baseline_similarity_v0.2_report.json
```

Suggested report fields:

```txt
run_id, model_name, dataset_version, split, language_scope, MAE, RMSE, F1_macro, Spearman, Precision_at_5, Recall_at_5, notes
```

The repository currently ignores `artifacts/`, so generated reports are local by default.

Do not commit large generated outputs or model files unless the project explicitly decides to version them.

## 10. Fine-Tuning Readiness

Fine-tuning should wait until there are enough high-quality labeled pairs.

Recommended minimum before fine-tuning:

```txt
300-500 labeled pairs for early experiments
1000+ labeled pairs for more meaningful comparison
```

Also check label balance before training. As a rough rule, try to have at least 50 pairs per label class before treating the dataset as trainable. An imbalanced dataset, for example many `strong_match` pairs and very few `poor_match` pairs, can produce misleading metrics.

Do not treat fine-tuning as successful unless it beats the pretrained baseline on a stable test set.

## 11. Artifact Direction

Future model artifacts should be stored outside source code or in a clearly ignored local folder.

Recommended local structure:

```txt
artifacts/
  models/
  reports/
```

Large model files, embedding caches, and experiment outputs should not be committed to the repository unless the project explicitly decides otherwise.

## 12. Phase Boundary

This document is still a planning artifact for later training work.

Do not add in this docs branch:

- training Python scripts;
- model checkpoints;
- generated embeddings;
- runtime inference changes.
