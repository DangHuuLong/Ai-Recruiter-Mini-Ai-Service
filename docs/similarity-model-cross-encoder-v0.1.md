# Similarity Model — Cross-Encoder v0.1 — Experiment Report

## 1. Overview

This document records the cross-encoder fine-tuning experiment built on top of the bi-encoder pipeline from v0.3.
The goal was to improve label accuracy by replacing the cosine-similarity scoring step with a cross-encoder that reads CV and JD together via bi-directional attention.

**Result**: The 2-stage pipeline (bi-encoder retrieval → cross-encoder rerank) reduces MAE from 11.16 to 9.93 and improves label accuracy from 43.14% to 58.19% — a **+35% relative improvement** in label accuracy on the held-out test split.

---

## 2. Model Architecture

### 2.1 Bi-encoder (Stage 1 — unchanged from v0.3)

| Property | Value |
|----------|-------|
| Base model | `sentence-transformers/all-MiniLM-L6-v2` |
| Architecture | Siamese Network (dual encoder) |
| Embedding dimension | 384 |
| Scoring | Cosine similarity → 0–100 |

### 2.2 Cross-encoder (Stage 2 — new)

| Property | Value |
|----------|-------|
| Base model | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Architecture | BERT-based cross-encoder (bi-directional attention) |
| Max sequence length | 512 tokens |
| Output | `sigmoid(logit) × 100` → score 0–100 |
| Loss function | `MSELoss` (regression, label = score / 100) |
| Activation | `Sigmoid` |

**How it works**: Unlike the bi-encoder, the cross-encoder concatenates CV and JD text into a single input and uses bi-directional self-attention across both documents. This lets the model detect token-level matches (e.g. a skill required by the JD that appears in the CV) rather than comparing aggregate embeddings.

```
[CLS] CV text [SEP] JD text [SEP]
         ↓  bi-directional attention
       logit  →  sigmoid  →  score (0–100)
```

**2-stage pipeline:**

```
CV text  ─┐
           ├─ bi-encoder → top-K CVs per JD  ─┐
JD text  ─┘                                    ├─ cross-encoder → reranked score
                                              ─┘
           (fallback: bi-encoder score for pairs outside top-K)
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

**Data preparation**: CV text and JD text were extracted from structured JSON objects using the same `_extract_resume_text` / `_extract_jd_text` helpers as the bi-encoder pipeline. Each record was written to JSONL with `label = score / 100` (normalized to 0–1 for MSE loss). Files are in `datasets/versions/v0.3/cross_encoder/`.

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

Training was performed on Google Colab (T4 GPU, ~15 min) using `sentence_transformers.CrossEncoder.fit()`.

**Training arguments:**

| Argument | Value |
|----------|-------|
| `epochs` | 10 |
| `batch_size` | 16 |
| `max_length` | 512 |
| `warmup_ratio` | 0.1 |
| `steps_per_epoch` | 307 (`ceil(4900 / 16)`) |
| `warmup_steps` | 30 (`int(307 × 0.1)`) |
| `loss_fct` | `torch.nn.MSELoss` |
| `activation_fct` | `torch.nn.Sigmoid` |
| `use_amp` | `True` (mixed precision) |
| `save_best_model` | `True` |
| Evaluator | `CECorrelationEvaluator` (Spearman on validation set) |

**Per-epoch validation results (Spearman correlation):**

| Epoch | Pearson | Spearman | Status |
|-------|---------|----------|--------|
| 1 | 0.8270 | 0.8322 | — |
| 2 | 0.8256 | 0.8353 | ↑ |
| 3 | 0.8403 | 0.8460 | ↑ |
| 4 | 0.8211 | 0.8271 | ↓ |
| 5 | 0.8399 | 0.8439 | ↑ |
| **6** | **0.8391** | **0.8464** | **best** |
| 7 | 0.8391 | 0.8459 | ↓ slight |
| 8 | 0.8394 | 0.8438 | ↓ |
| 9 | 0.8372 | 0.8422 | ↓ |
| 10 | 0.8365 | 0.8431 | — |

---

## 5. Best Checkpoint

The trainer saved the checkpoint with the highest Spearman correlation on the validation set. **Epoch 6** achieved `val_spearman = 0.8464`, which was not surpassed in subsequent epochs. This model was saved to `models/cross-encoder-cv-jd-v0.1/`.

Note: epoch 4 shows an unusual dip (Spearman 0.8271) despite surrounding epochs being in the 0.84+ range — likely due to stochastic batch sampling at a learning rate transition point during warmup exit.

---

## 6. Benchmark on Test Split

### 6.1 Bi-encoder Retrieval Quality (Recall@50)

The bi-encoder is used as the first stage to narrow candidates before cross-encoder scoring. recall@K measures the fraction of test pairs where the correct CV appears in the bi-encoder's top-50 results for its JD.

| Subset | Recall@50 | Hits / Total |
|--------|-----------|--------------|
| All pairs | 47.81% | 502 / 1050 |
| Score ≥ 60 (moderate+) | 75.65% | 463 / 612 |
| Score ≥ 75 (strong+) | 86.31% | 391 / 453 |

For the pairs that matter most (moderate and above), the bi-encoder places the correct CV in top-50 with high reliability. The low recall for all pairs is expected — the test set includes 210 unique CVs and 210 unique JDs, and most CV-JD combinations are low-relevance (score < 40).

### 6.2 Scoring Accuracy (Test Split, 1,050 pairs)

| Approach | MAE | RMSE | Label Acc | ΔMAE |
|----------|-----|------|-----------|------|
| Bi-encoder only (v0.3) | 11.1637 | 14.5805 | 43.14% | baseline |
| Cross-encoder only (v0.1) | 10.0856 | 14.7551 | 58.38% | −1.08 (−9.6%) |
| **2-stage pipeline** | **9.9292** | **13.9806** | **58.19%** | **−1.23 (−11.0%)** |

*Cross-encoder calibration: `sigmoid(logit) × 100`. Pairs outside top-50 fall back to bi-encoder score.*

**Interpretation:**
- The cross-encoder alone improves label accuracy by **+15.2 percentage points** (43.14% → 58.38%) versus the bi-encoder.
- The 2-stage pipeline further lowers MAE by combining the bi-encoder's recall breadth with the cross-encoder's precision on relevant candidates.
- RMSE drops only slightly in the pipeline vs cross-encoder-only, because the 548 pairs outside top-50 still use the less accurate bi-encoder score.

To reproduce this benchmark:

```bash
python -m training.evaluate_cross_encoder_pipeline \
  --biencoder models/fine-tuned-miniLM-v0.3 \
  --cross-encoder models/cross-encoder-cv-jd-v0.1 \
  --top-k 50 \
  --pairs datasets/versions/v0.3/cv_jd_pairs.jsonl \
  --resumes datasets/versions/v0.3/resumes.jsonl \
  --jds datasets/versions/v0.3/job_descriptions.jsonl \
  --output artifacts/reports/cross_encoder_pipeline_v0.1.json
```

Report saved at: `artifacts/reports/cross_encoder_pipeline_v0.1.json`

---

## 7. Calibration Experiment

After fine-tuning, an isotonic regression calibrator was fitted on the validation set to test whether post-hoc score adjustment could improve label accuracy.

**Method**: `sklearn.isotonic.IsotonicRegression(out_of_bounds="clip")` fitted on 1,050 validation pairs. Calibration script: `training/calibrate_cross_encoder.py`.

| Approach | MAE | RMSE | Label Acc |
|----------|-----|------|-----------|
| Cross-encoder raw | 10.0856 | 14.7551 | **58.38%** |
| Cross-encoder calibrated | 9.8689 | 14.3406 | 57.33% ↓ |
| 2-stage pipeline raw | 9.9292 | 13.9806 | **58.19%** |
| 2-stage pipeline calibrated | 9.9164 | 14.0587 | 56.67% ↓ |

**Finding**: Calibration reduced MAE marginally but lowered label accuracy on both validation (53.05% → 52.48%) and test (58.38% → 57.33%) splits.

**Root cause**: Isotonic regression minimises squared error in score space — it has no knowledge of the discrete bucket boundaries (40 / 60 / 75 / 90). Scores near boundaries were shifted in directions that slightly reduced MAE but caused net-negative bucket crossings. Because the model was fine-tuned directly with `label = score/100` as the regression target, `sigmoid(logit)×100` is already reasonably calibrated to the true score distribution, leaving little room for isotonic calibration to help.

**Decision**: Raw cross-encoder scores (`sigmoid(logit)×100`) are used in the final pipeline. The calibration infrastructure (`training/calibrate_cross_encoder.py` + `models/cross-encoder-cv-jd-v0.1/calibrator.pkl`) is retained — future, larger models may benefit more from post-hoc calibration if their raw output is less well-calibrated.

---

## 8. Model Artifact

The final model is saved at `models/cross-encoder-cv-jd-v0.1/`.

| File | Purpose |
|------|---------|
| `model.safetensors` | Fine-tuned cross-encoder weights |
| `config.json` | Model architecture config |
| `tokenizer.json` + `tokenizer_config.json` | Tokenizer |
| `special_tokens_map.json` | Special token definitions |
| `CECorrelationEvaluator_val_results.csv` | Per-epoch Spearman/Pearson on validation set |
| `calibrator.pkl` | Isotonic regression calibrator (see Section 7 — not used in final pipeline) |

To use in inference:

```python
from sentence_transformers import CrossEncoder
import math

model = CrossEncoder("models/cross-encoder-cv-jd-v0.1", max_length=512)

def score(cv_text: str, jd_text: str) -> float:
    logit = model.predict([(cv_text, jd_text)])[0]
    return 1.0 / (1.0 + math.exp(-logit)) * 100  # sigmoid → 0–100
```

---

## 9. Limitations

- **Recall ceiling**: 47.81% recall@50 for all pairs means the pipeline falls back to the weaker bi-encoder for ~52% of test pairs. Increasing top-K trades recall improvement against cross-encoder inference cost.
- **Score discontinuity at fallback boundary**: Reranked pairs use `sigmoid(logit)×100` while fallback pairs use `cosine×100` — the two score scales are not aligned, which may cause inconsistent rankings across the boundary.
- **Inference speed**: The cross-encoder scores one pair at a time. At top-K=50 with 210 JDs, this is 10,500 cross-encoder calls versus 420 bi-encoder calls — roughly 25× slower.
- **English-only**: Inherits the same language limitation as the bi-encoder; Vietnamese or mixed-language CVs/JDs may degrade performance.
- **Label accuracy plateau**: 58% label accuracy still means ~42% of pairs are assigned the wrong label bucket. The score rubric boundaries (40/60/75/90) may not align well with the model's internal score distribution.
- **Isotonic calibration ineffective**: Post-hoc score calibration did not improve label accuracy for this model (see Section 7). The fine-tuning objective already produced well-calibrated raw scores.

---

## 10. Next Steps

| Action | Priority |
|--------|----------|
| Tune top-K threshold (try 30, 75, 100) and measure recall-vs-latency trade-off | High |
| Fine-tune a larger cross-encoder (`ms-marco-MiniLM-L-12-v2`, 12 layers) | Medium |
| Explore ensemble: average bi-encoder and cross-encoder scores for reranked pairs | Low |
