# Training Strategy

## 1. Purpose

This document records the training direction for the CV-JD matching model after the baseline scoring flow is stable.

The current branch only prepares the dataset and labeling design. Actual training scripts should be added later.

## 2. Recommended Order

Do not fine-tune first. Follow this order:

1. define dataset schema;
2. define labeling rubric;
3. create a small clean labeled dataset;
4. validate dataset quality;
5. split train/validation/test;
6. evaluate pretrained embedding baseline;
7. add feature-based scoring if needed;
8. fine-tune only when enough labeled pairs exist;
9. export the best model artifact for local/service inference.

## 3. Initial Baseline

The first baseline should be simple and explainable:

```txt
resume_text -> embedding
job_description_text -> embedding
similarity = cosine(resume_embedding, jd_embedding)
```

Suggested lightweight model for first experiment:

```txt
sentence-transformers/all-MiniLM-L6-v2
```

This model is suitable for a local baseline because it is lightweight and easy to run.

## 4. Evaluation Metrics

Use different metrics depending on the target.

| Target | Suggested Metrics |
| --- | --- |
| Score regression | MAE, RMSE |
| Match class classification | Accuracy, macro F1 |
| Candidate ranking | Spearman correlation, Precision@K, Recall@K |

For recruitment screening, ranking metrics are important because the service often needs to order candidates by fit, not only predict one score.

## 5. Feature-Based Model Direction

Before fine-tuning deep models, consider a feature-based model using parser outputs.

Potential features:

- embedding similarity;
- required skill coverage;
- preferred skill coverage;
- missing required skill count;
- experience gap;
- education match;
- domain keyword overlap.

This direction is practical when the dataset is still small.

## 6. Fine-Tuning Readiness

Fine-tuning should wait until there are enough high-quality labeled pairs.

Recommended minimum before fine-tuning:

```txt
300-500 labeled pairs for early experiments
1000+ labeled pairs for more meaningful comparison
```

Do not treat fine-tuning as successful unless it beats the pretrained baseline on a stable test set.

## 7. Artifact Direction

Future model artifacts should be stored outside source code or in a clearly ignored local folder.

Recommended local structure:

```txt
artifacts/
├── models/
└── reports/
```

Large model files, embedding caches, and experiment outputs should not be committed to the repository unless the project explicitly decides otherwise.

## 8. Phase Boundary

This document is only a planning artifact for later training work.

Do not add in this branch:

- training Python scripts;
- model checkpoints;
- generated embeddings;
- evaluation reports from incomplete datasets;
- runtime inference changes.
