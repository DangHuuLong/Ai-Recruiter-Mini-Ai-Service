# Similarity Model v0.2 — Experiment Report

## 1. Overview

This document records the fine-tuning experiment for the CV-JD similarity model using dataset v0.2.
The goal was to improve on the pretrained embedding baseline by fine-tuning on labeled CV-JD pairs.

**Result**: Fine-tuned model (epoch 3) reduces MAE by 41.4% compared to the base model on the held-out test split.

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
| Version | v0.2 |
| Total pairs | 2,275 |
| Train split | 70% (~1,593 pairs) |
| Validation split | 15% (~341 pairs) |
| Test split | 15% (350 pairs) |
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

Fine-tuning was done incrementally — one epoch at a time — evaluating on the validation set after each epoch. This allows early stopping at the optimal point rather than training to a fixed epoch count.

**Metrics used:**

- **MAE** (Mean Absolute Error): primary metric, measures average score prediction error
- **RMSE** (Root Mean Square Error): penalizes large errors more than MAE
- **Label accuracy**: percentage of pairs where predicted label matches true label

*Note: F1/precision/recall are not used because this is a regression task (predicting a continuous score 0–100), not a classification task.*

**Per-epoch validation results:**

| Epoch | MAE | RMSE | Label Acc | Status |
|-------|-----|------|-----------|--------|
| Baseline (no fine-tune) | 17.6431 | 22.4497 | 28.22% | reference |
| Epoch 2 | ~13.x | — | — | improvement |
| Epoch 3 | 10.88 | — | — | **best** |
| Epoch 4 | > ep3 | — | — | overfitting — stopped |

---

## 5. Why Epoch 3

Epoch 3 achieved the lowest validation MAE. Epoch 4 showed a higher MAE than epoch 3, which is the classic sign of overfitting — the model started memorising training data instead of generalising.

Training was stopped at epoch 4 to preserve the epoch 3 checkpoint as the final artifact.

---

## 6. Benchmark on Test Split

Evaluated with `training/evaluate_similarity_pipeline.py` on the held-out test split (350 pairs, never seen during training or validation).

| Model | MAE | RMSE | Label Acc | ΔMAE |
|-------|-----|------|-----------|------|
| base (`all-MiniLM-L6-v2`) | 20.3240 | 25.4998 | 32.29% | baseline |
| v0.2-ep3 (fine-tuned) | 11.9164 | 17.4966 | 47.71% | **-8.41 (−41.4%)** |

**Interpretation:**
- On average, the fine-tuned model's score prediction is off by ~12 points vs ~20 points for the base model.
- Label accuracy improved from 32% to 48% — nearly 1 in 2 pairs now get the correct label bucket.
- The improvement holds on unseen data (test split), confirming the model generalised and did not overfit.

To reproduce this benchmark:

```bash
python -m training.evaluate_similarity_pipeline \
  --models base:sentence-transformers/all-MiniLM-L6-v2 \
           v0.2-ep3:models/fine-tuned-miniLM-v0.2-ep3 \
  --output artifacts/reports/comparison_v0.2.json
```

Report saved at: `artifacts/reports/comparison_v0.2.json`

---

## 7. Model Artifact

The final model artifact is `fine-tuned-miniLM-v0.2-ep3`.

| File | Purpose |
|------|---------|
| `config.json` | Model architecture config |
| `model.safetensors` | Fine-tuned weights |
| `tokenizer.json` + `tokenizer_config.json` | Tokenizer |
| `modules.json` | SentenceTransformer module definition |
| `1_Pooling/` | Mean pooling layer config |
| `2_Normalize/` | L2 normalization layer config |

The artifact is **not committed to git**. Download from Google Drive and place at `models/fine-tuned-miniLM-v0.2-ep3/`.

To load in the service, set in `.env`:

```env
SIMILARITY_MODEL_PATH=models/fine-tuned-miniLM-v0.2-ep3
SIMILARITY_MODEL_VERSION=v0.2
```

See [model-integration-guide.md](model-integration-guide.md) for full setup instructions.

---

## 8. Limitations

- **English-only**: `all-MiniLM-L6-v2` is optimised for English. Vietnamese or mixed-language CVs/JDs may produce less accurate scores.
- **Dataset size**: 2,275 pairs is sufficient for a first experiment but small for a production model. More pairs and better label balance would improve robustness.
- **Label imbalance**: Dataset label distribution is not perfectly balanced across the 5 categories, which may bias predictions toward more frequent labels.
- **Score calibration**: The cosine similarity scale (0–1) maps directly to 0–100, but the model was not explicitly calibrated to match human-labeled score distributions.

---

## 9. Next Steps

| Action | Branch |
|--------|--------|
| Expand dataset and re-train | `experiment/fine-tune-similarity-colab-v0.3` |
| Try larger base model (`all-mpnet-base-v2`) | `experiment/fine-tune-similarity-colab-v0.3` |
| Add multilingual support | future |
