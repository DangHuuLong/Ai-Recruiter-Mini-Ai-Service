# Similarity Model — Cross-Encoder v0.2 — Experiment Report

## 1. Overview

This document records the cross-encoder fine-tuning experiment that upgrades the base model from a 6-layer to a 12-layer MiniLM variant while keeping all training code and data identical to v0.1. The goal was to test whether the larger backbone yields measurably better scoring accuracy.

**Result**: Upgrading to `ms-marco-MiniLM-L-12-v2` (12 layers, ~2× parameters) improves cross-encoder label accuracy by **+2.38 pp** (58.38% → 60.76%) and reduces MAE from 10.09 to 9.72 on the held-out test split. The 2-stage pipeline (bi-encoder retrieval → cross-encoder rerank) achieves 59.24% label accuracy — a **+37% relative improvement** over the bi-encoder alone (43.14%).

**v0.1 → v0.2 comparison:**

| Metric | v0.1 (L-6) | v0.2 (L-12) | Δ |
|--------|-----------|------------|---|
| Val best Spearman | 0.8464 | **0.8514** | +0.0050 |
| CE-only LabelAcc | 58.38% | **60.76%** | +2.38 pp |
| CE-only MAE | 10.09 | **9.72** | −0.37 |
| Pipeline LabelAcc | 58.19% | **59.24%** | +1.05 pp |
| Pipeline MAE | 9.93 | **9.84** | −0.09 |

---

## 2. Model Architecture

### 2.1 Bi-encoder (Stage 1 — unchanged from v0.3)

| Property | Value |
|----------|-------|
| Base model | `sentence-transformers/all-MiniLM-L6-v2` |
| Architecture | Siamese Network (dual encoder) |
| Embedding dimension | 384 |
| Scoring | Cosine similarity → 0–100 |

### 2.2 Cross-encoder (Stage 2 — v0.2)

| Property | Value |
|----------|-------|
| Base model | `cross-encoder/ms-marco-MiniLM-L-12-v2` |
| Architecture | BERT-based cross-encoder, 12 transformer layers |
| Max sequence length | 512 tokens |
| Output | `sigmoid(logit) × 100` → score 0–100 |
| Loss function | `MSELoss` (regression, label = score / 100) |
| Activation | `Sigmoid` |

The only change from v0.1 is the base model: `L-12-v2` has 12 transformer layers versus `L-6-v2`'s 6, giving it richer cross-attention capacity over long CV+JD sequences. All training code, data, and hyperparameters are identical to v0.1.

```
[CLS] CV text [SEP] JD text [SEP]
         ↓  bi-directional attention (12 layers)
       logit  →  sigmoid  →  score (0–100)
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

Training was performed on Google Colab (T4 GPU, ~18 min) using `sentence_transformers.CrossEncoder.fit()`.

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
| 1 | 0.7805 | 0.8097 | — |
| 2 | 0.8061 | 0.8349 | ↑ |
| 3 | 0.8187 | 0.8450 | ↑ |
| **4** | **0.8239** | **0.8514** | **best** |
| 5 | 0.8226 | 0.8486 | ↓ |
| 6 | 0.8216 | 0.8489 | → |
| 7 | 0.8254 | 0.8490 | → |
| 8 | 0.8243 | 0.8481 | ↓ |
| 9 | 0.8274 | 0.8497 | ↑ |
| 10 | 0.8268 | 0.8505 | → |

---

## 5. Best Checkpoint

The trainer saved the checkpoint with the highest Spearman correlation on the validation set. **Epoch 4** achieved `val_spearman = 0.8514`, which was not surpassed in the remaining six epochs. After epoch 4, Spearman oscillates in the 0.848–0.850 range — the model converged early, consistent with the 12-layer model having a higher effective learning capacity than the 6-layer v0.1.

The saved model is at `models/cross-encoder-cv-jd-v0.2/`.

---

## 6. Benchmark on Test Split

### 6.1 Bi-encoder Retrieval Quality (Recall@50)

The bi-encoder stage is unchanged from v0.1. Recall numbers are identical.

| Subset | Recall@50 | Hits / Total |
|--------|-----------|--------------|
| All pairs | 47.81% | 502 / 1050 |
| Score ≥ 60 (moderate+) | 75.65% | 463 / 612 |
| Score ≥ 75 (strong+) | 86.31% | 391 / 453 |

### 6.2 Scoring Accuracy (Test Split, 1,050 pairs)

| Approach | MAE | RMSE | Label Acc | ΔMAE |
|----------|-----|------|-----------|------|
| Bi-encoder only (v0.3) | 11.1637 | 14.5805 | 43.14% | baseline |
| Cross-encoder only (v0.1, L-6) | 10.0856 | 14.7551 | 58.38% | −1.08 |
| **Cross-encoder only (v0.2, L-12)** | **9.7151** | **13.9108** | **60.76%** | **−1.45** |
| 2-stage pipeline (v0.1) | 9.9292 | 13.9806 | 58.19% | −1.23 |
| **2-stage pipeline (v0.2)** | **9.8366** | **13.9596** | **59.24%** | **−1.33** |

*Cross-encoder calibration: `sigmoid(logit) × 100`. Pairs outside top-50 fall back to bi-encoder score (548/1050).*

**Interpretation:**
- The 12-layer model improves cross-encoder LabelAcc by **+2.38 pp** over v0.1 (58.38% → 60.76%) and MAE by **−0.37**.
- The 2-stage pipeline for v0.2 is slightly below the standalone v0.2 cross-encoder (59.24% vs 60.76%) because 548 pairs outside top-50 fall back to the weaker bi-encoder. This is the same pattern as v0.1.
- RMSE improves notably for the standalone cross-encoder (14.75 → 13.91), suggesting the 12-layer model is better calibrated in absolute score space.

To reproduce this benchmark:

```bash
python -m training.evaluate_cross_encoder_pipeline \
  --biencoder models/fine-tuned-miniLM-v0.3 \
  --cross-encoder models/cross-encoder-cv-jd-v0.2 \
  --top-k 50 \
  --pairs datasets/versions/v0.3/cv_jd_pairs.jsonl \
  --resumes datasets/versions/v0.3/resumes.jsonl \
  --jds datasets/versions/v0.3/job_descriptions.jsonl \
  --output artifacts/reports/cross_encoder_pipeline_v0.2.json \
  --calibrator none
```

Report saved at: `artifacts/reports/cross_encoder_pipeline_v0.2.json`

---

## 7. Calibration Experiment

An isotonic regression calibrator was fitted on the validation set to confirm whether post-hoc calibration helps v0.2, given that it did not help v0.1.

**Method**: `sklearn.isotonic.IsotonicRegression(out_of_bounds="clip")` fitted on 1,050 validation pairs. Calibration script: `training/calibrate_cross_encoder.py`.

| Split | Metric | Raw | Calibrated | Δ |
|-------|--------|-----|------------|---|
| Validation | MAE | 10.4764 | 9.7142 | −0.76 |
| Validation | LabelAcc | 52.00% | 51.43% | −0.57 pp |
| Test | MAE | 9.7151 | 9.7720 | +0.06 |
| Test | LabelAcc | **60.76%** | 56.10% | **−4.66 pp** |

**Finding**: Calibration again degrades label accuracy — and more severely on test (−4.66 pp) than on v0.1 (−1.05 pp). The test raw LabelAcc (60.76%) is substantially higher than the validation raw LabelAcc (52.00%), which means the isotonic calibrator fits the wrong score distribution and shifts test predictions into the wrong buckets.

**Root cause**: Two compounding issues:
1. The validation set was used for both model selection (best checkpoint by val_spearman) and calibrator fitting, making it "doubly seen" — the calibrator overfits validation noise.
2. The v0.2 model is already better calibrated in raw score space (RMSE 13.91 vs 14.76 for v0.1), leaving even less room for isotonic regression to improve.

**Decision**: Raw cross-encoder scores (`sigmoid(logit)×100`) are used in the final pipeline. The `calibrator.pkl` is retained alongside the model for reference but is not loaded in production.

---

## 8. Model Artifact

The final model is saved at `models/cross-encoder-cv-jd-v0.2/`.

| File | Purpose |
|------|---------|
| `model.safetensors` | Fine-tuned cross-encoder weights (12-layer MiniLM) |
| `config.json` | Model architecture config |
| `tokenizer.json` + `tokenizer_config.json` | Tokenizer |
| `special_tokens_map.json` + `vocab.txt` | Vocabulary |
| `CECorrelationEvaluator_val_results.csv` | Per-epoch Spearman/Pearson on validation set |
| `calibrator.pkl` | Isotonic regression calibrator (see Section 7 — not used in final pipeline) |

To use in inference:

```python
from sentence_transformers import CrossEncoder
import math

model = CrossEncoder("models/cross-encoder-cv-jd-v0.2", max_length=512)

def score(cv_text: str, jd_text: str) -> float:
    logit = model.predict([(cv_text, jd_text)])[0]
    return 1.0 / (1.0 + math.exp(-logit)) * 100  # sigmoid → 0–100
```

---

## 9. Limitations

- **Recall ceiling unchanged**: 47.81% recall@50 for all pairs is unchanged because the bi-encoder (Stage 1) was not updated. ~52% of pairs still fall back to the weaker bi-encoder score in the pipeline.
- **Pipeline underperforms standalone cross-encoder**: At top-K=50, pipeline (59.24%) is worse than cross-encoder alone (60.76%) because the 548 fallback pairs drag down aggregate accuracy. This can be mitigated by increasing top-K or upgrading the bi-encoder.
- **Inference speed**: The 12-layer model is ~1.6× slower than the 6-layer v0.1 at inference time (CPU: ~3–4 s/batch vs ~2 s/batch observed locally). At top-K=50 with 210 JDs, this is 10,500 cross-encoder calls.
- **Calibration ineffective**: Post-hoc isotonic calibration harms label accuracy for this model (−4.66 pp on test). The fine-tuning objective already produces well-calibrated raw scores.
- **Label accuracy ceiling**: 60.76% label accuracy means ~39% of pairs still receive the wrong label bucket. The rubric boundaries (40/60/75/90) create hard classification constraints that a regression model cannot optimise directly.

---

## 10. Next Steps

| Action | Priority |
|--------|----------|
| Upgrade bi-encoder to improve recall@50 beyond 47.81% (current bottleneck for the pipeline) | High |
| Try higher top-K (75, 100) to reduce fallback pairs and close the pipeline vs CE-only gap | Medium |
| Explore direct classification with a soft-label cross-entropy loss (aligned to the 5-bucket rubric) | Medium |
| Investigate cross-encoder ensembling (v0.1 + v0.2 score averaging) | Low |
