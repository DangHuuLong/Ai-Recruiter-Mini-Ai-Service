# Similarity Model v0.3 — Experiment Report

## 1. Overview

This document records the fine-tuning experiment for the CV-JD similarity model using dataset v0.3.
The goal was to improve on v0.2 by training on a larger, more diverse dataset with automatic best-checkpoint saving.

**Result**: Fine-tuned model (best checkpoint at epoch 5) reduces MAE by 43.1% compared to the base model on the held-out test split.

---

## 2. Model Architecture

| Property | Value |
|----------|-------|
| Base model | `sentence-transformers/all-MiniLM-L6-v2` |
| Architecture | Siamese Network (dual encoder) |
| Embedding dimension | 384 |
| Max sequence length | 256 tokens |
| Loss function | `CosineSimilarityLoss` (MSE on cosine similarity values) |

**How it works**: CV text and JD text are encoded separately by the same encoder. The similarity score is the cosine similarity between the two embeddings, scaled to 0–100.

```
CV text  → encoder → embedding_cv  ─┐
                                      ├─ cosine_similarity → score (0–100)
JD text  → encoder → embedding_jd  ─┘
```

---

## 3. Dataset

| Property | Value |
|----------|-------|
| Version | v0.3 |
| Total pairs | 7,000 |
| Train split | 70% (4,900 pairs) |
| Validation split | 15% (1,050 pairs) |
| Test split | 15% (1,050 pairs) |
| Label rubric | `rubric_v0.2` |
| Split strategy | Component-based (no resume/JD leakage across splits) |

**Label categories:**

| Label | Score range |
|-------|------------|
| poor_match | 0 – 39 |
| weak_match | 40 – 59 |
| moderate_match | 60 – 74 |
| strong_match | 75 – 89 |
| excellent_match | 90 – 100 |

---

## 4. Training

Training ran for up to 20 epochs with automatic best-checkpoint saving. The trainer evaluated on the validation set after every epoch and saved the checkpoint with the highest `spearman_cosine`. At the end of training, the best checkpoint was loaded automatically (`load_best_model_at_end=True`).

**Training arguments:**

| Argument | Value |
|----------|-------|
| `num_train_epochs` | 20 |
| `per_device_train_batch_size` | 32 |
| `eval_strategy` | `epoch` |
| `save_strategy` | `epoch` |
| `metric_for_best_model` | `eval_val_spearman_cosine` |
| `save_total_limit` | 2 |
| `fp16` | `True` |

**Metrics used:**

- **Spearman cosine** (`eval_val_spearman_cosine`): primary metric for checkpoint selection — measures rank correlation between predicted and true scores
- **MAE** (Mean Absolute Error): reported in final evaluation
- **RMSE** (Root Mean Square Error): penalises large errors more than MAE
- **Label accuracy**: percentage of pairs where predicted label matches true label

*Note: F1/precision/recall are not used because this is a regression task (predicting a continuous score 0–100), not a classification task.*

**Per-epoch validation results (spearman_cosine):**

| Epoch | Spearman cosine | Status |
|-------|----------------|--------|
| 1 | 0.8480 | — |
| 2 | 0.8597 | ↑ |
| 3 | 0.8664 | ↑ |
| 4 | 0.8660 | ↓ slight |
| **5** | **0.8678** | **best** |
| 6 | 0.8623 | ↓ overfitting |
| 7 | 0.8645 | — |
| 8 | 0.8587 | ↓ |
| 9–20 | < 0.867 | no recovery |

---

## 5. Best Checkpoint

The trainer automatically selected **epoch 5** (step 770) as the best checkpoint with `spearman_cosine = 0.8678`. After epoch 5, the metric did not recover, confirming overfitting on subsequent epochs despite 15 more epochs of training.

The best model was loaded at end of training and saved to the output directory as the final artifact.

---

## 6. Benchmark on Test Split

Evaluated with `training/evaluate_similarity_pipeline.py` on the held-out test split (1,050 pairs, never seen during training or validation).

| Model | MAE | RMSE | Label Acc | ΔMAE |
|-------|-----|------|-----------|------|
| base (`all-MiniLM-L6-v2`) | 19.6377 | 23.5987 | 26.86% | baseline |
| v0.2-ep3 (fine-tuned) | 12.2380 | 16.3091 | 42.76% | −7.40 (−37.7%) |
| **v0.3 (fine-tuned)** | **11.1637** | **14.5805** | **43.14%** | **−8.47 (−43.1%)** |

**Interpretation:**
- On average, v0.3 predicts the score within ~11 points vs ~20 points for the base model.
- v0.3 improves over v0.2 by ~1.1 MAE points and ~1.7 RMSE points on the larger v0.3 test set.
- Label accuracy is comparable to v0.2 (43.1% vs 42.8%) — the gain from more data is mainly in score precision, not label bucket accuracy.

To reproduce this benchmark:

```bash
python training/evaluate_similarity_pipeline.py \
  --models "baseline:sentence-transformers/all-MiniLM-L6-v2" \
           "v0.2-ep3:models/fine-tuned-miniLM-v0.2-ep3" \
           "v0.3:models/fine-tuned-miniLM-v0.3" \
  --pairs datasets/versions/v0.3/cv_jd_pairs.jsonl \
  --resumes datasets/versions/v0.3/resumes.jsonl \
  --jds datasets/versions/v0.3/job_descriptions.jsonl \
  --output artifacts/reports/model_comparison_v0.3_test.json
```

Report saved at: `artifacts/reports/model_comparison_v0.3_test.json`

---

## 7. Model Artifact

The final model artifact is `fine-tuned-miniLM-v0.3` (best checkpoint at epoch 5).

| File | Purpose |
|------|---------|
| `config.json` | Model architecture config |
| `model.safetensors` | Fine-tuned weights |
| `tokenizer.json` + `tokenizer_config.json` | Tokenizer |
| `modules.json` | SentenceTransformer module definition |
| `1_Pooling/` | Mean pooling layer config |
| `2_Normalize/` | L2 normalization layer config |

To load in the service, set in `.env`:

```env
SIMILARITY_MODEL_PATH=models/fine-tuned-miniLM-v0.3
SIMILARITY_MODEL_VERSION=v0.3
```

See [model-integration-guide.md](model-integration-guide.md) for full setup instructions.

---

## 8. Limitations

- **English-only**: `all-MiniLM-L6-v2` is optimised for English. Vietnamese or mixed-language CVs/JDs may produce less accurate scores.
- **Architecture ceiling**: MiniLM-L6 (6 layers, 22M params) has limited capacity. Increasing dataset size yields diminishing returns beyond this model's learning capacity.
- **Task-model mismatch**: Cosine similarity between resume and JD text measures semantic closeness, not structured skill matching. The model cannot distinguish "has Python" from "requires Python" at the token level.
- **Label accuracy plateau**: Label accuracy stagnated at ~43% across v0.2 and v0.3, suggesting the limit is architectural rather than data-driven.
- **Score calibration**: Cosine similarity (0–1) maps directly to 0–100 without explicit calibration to human-labeled score distributions.

---

## 9. Next Steps

| Action | Priority |
|--------|----------|
| Try larger backbone (`all-mpnet-base-v2`, 12 layers) | High |
| Try cross-encoder architecture for better skill matching | High |
| Calibrate output scores against validation distribution | Medium |
| Add multilingual support | Low |
