# Similarity Model — Cross-Encoder v0.3 — Experiment Report

## 1. Overview

This document records the cross-encoder fine-tuning experiment that introduces `BoundaryAwareLoss` — a composite loss combining MSE with ordinal binary cross-entropy penalties at the four rubric boundaries (40/60/75/90). The goal was to directly penalise predictions that land on the wrong side of a bucket boundary, targeting label accuracy beyond what MSE can achieve.

**Result**: `BoundaryAwareLoss` does **not** improve label accuracy over the v0.2 MSELoss baseline. Both evaluator configurations (Spearman and LabelAcc) yield test LabelAcc ≤ 60.76% with worse MAE and RMSE than v0.2. The boundary-aware approach is **not effective** for this dataset and model.

**v0.2 → v0.3 comparison:**

| Metric | v0.2 (MSE, Spearman) | v0.3 run-1 (Boundary, Spearman) | v0.3 run-2 (Boundary, LabelAcc) |
|--------|---------------------|--------------------------------|--------------------------------|
| Test LabelAcc | **60.76%** | 60.76% | 59.62% |
| Test MAE | **9.72** | 9.98 | 10.40 |
| Test RMSE | **13.91** | 14.26 | 15.11 |
| Mean predicted score | — | 65.25 | 64.25 |
| Mean target score | 62.85 | 62.85 | 62.85 |

---

## 2. Hypothesis

v0.2 reaches 60.76% label accuracy using a pure MSE regression objective. The MSE loss minimises absolute score distance but does not distinguish between a prediction that is marginally on the wrong side of a boundary (e.g., predicting 59 when the ground truth is 61) from one that is far off. The hypothesis was that adding an explicit ordinal penalty at each boundary would force the model to learn which side of each threshold a pair belongs to.

---

## 3. BoundaryAwareLoss

The loss is a weighted sum of MSE and an ordinal BCE term:

```
L(pred, label) = MSE(pred, label) + α × (1/B) × Σ_b BCE_with_logits(20×(pred−b), 𝟙[label≥b])
```

where:
- `pred`, `label` ∈ [0, 1] (sigmoid applied by `CrossEncoder.fit()` before this loss receives them)
- `B = 4` boundaries: 0.40, 0.60, 0.75, 0.90
- `α = 0.3` (ordinal term weight relative to MSE)
- Logit scale factor `20` creates a soft threshold: `sigmoid(20×(pred−b)) ≈ 𝟙[pred≥b]`

`F.binary_cross_entropy_with_logits` is used instead of `F.binary_cross_entropy` because the latter is unsafe under AMP autocast (GPU training). Passing the raw logit `20×(pred−b)` to `binary_cross_entropy_with_logits` is numerically equivalent and autocast-safe.

---

## 4. Training

Training was performed on Google Colab (T4 GPU) using `sentence_transformers.CrossEncoder.fit()`.

**Training arguments (both runs):**

| Argument | Value |
|----------|-------|
| `base_model` | `cross-encoder/ms-marco-MiniLM-L-12-v2` |
| `epochs` | 10 |
| `batch_size` | 16 |
| `max_length` | 512 |
| `warmup_ratio` | 0.1 |
| `loss_fct` | `BoundaryAwareLoss` (α=0.3) |
| `activation_fct` | `torch.nn.Sigmoid` |
| `use_amp` | `True` |
| `save_best_model` | `True` |

**Run 1** — checkpoint selected by `val_spearman` (via `CECorrelationEvaluator`):

| Argument | Value |
|----------|-------|
| `evaluator` | `CECorrelationEvaluator` (Spearman) |

**Run 2** — checkpoint selected by `val_label_acc` (via `CELabelAccEvaluator`):

| Argument | Value |
|----------|-------|
| `evaluator` | `CELabelAccEvaluator` (label accuracy on val set) |

`CELabelAccEvaluator` was implemented specifically for this experiment. It computes label accuracy on the validation set at each epoch and saves the checkpoint with the highest val LabelAcc.

---

## 5. Results

### 5.1 Run 1 — Spearman checkpoint selection

| Split | MAE | RMSE | LabelAcc | Mean predicted | Mean target |
|-------|-----|------|----------|---------------|------------|
| Validation | 11.0875 | 14.9432 | 55.62% | 64.82 | 60.65 |
| Test | 9.9830 | 14.2613 | **60.76%** | 65.25 | 62.85 |

### 5.2 Run 2 — LabelAcc checkpoint selection

| Split | MAE | RMSE | LabelAcc | Mean predicted | Mean target |
|-------|-----|------|----------|---------------|------------|
| Validation | 11.0196 | 14.9074 | 55.52% | 64.54 | 60.65 |
| Test | 10.4040 | 15.1145 | 59.62% | 64.25 | 62.85 |

---

## 6. Analysis

### 6.1 BoundaryAwareLoss does not improve LabelAcc

Run 1 matches v0.2's test LabelAcc (60.76%) but is strictly worse on MAE (+0.27) and RMSE (+0.35). Run 2 is worse on all three metrics. The boundary-aware term provides no benefit over a pure regression objective.

### 6.2 Systematic overestimation bias

All v0.3 runs show `mean_predicted > mean_target` (+1.4 to +2.4 points). The ordinal BCE term pushes predictions upward to "clear" each threshold rather than aligning them to ground truth. This introduces a systematic upward bias that partially undermines the regression component.

### 6.3 LabelAcc evaluator does not improve test generalisation

Switching checkpoint selection from Spearman to val LabelAcc (run 2) makes test performance worse (-1.14 pp LabelAcc, +0.42 MAE). The validation set (1,050 pairs) is too small for LabelAcc to be a stable signal for checkpoint selection — Spearman on the same set is smoother and more reliable as a proxy metric.

### 6.4 Val LabelAcc improved, test did not

Val LabelAcc increased from v0.2's 52.00% to 55.52–55.62% with BoundaryAwareLoss. This improvement does not transfer to test. The boundary loss learns patterns specific to the validation distribution rather than the underlying task structure.

### 6.5 Root cause: conflicting objectives

Alpha=0.3 creates a gradient conflict: MSE pulls predictions toward ground truth while boundary BCE pulls predictions toward "clear" decisions at threshold values. With 4 boundaries and α=0.3, the net effect is a bias toward scores just above each threshold (40, 60, 75, 90) rather than the correct continuous ground truth values.

---

## 7. Technical Fixes Introduced

Two technical improvements were made alongside the loss experiment:

**AMP compatibility fix**: The original `BoundaryAwareLoss` used `F.binary_cross_entropy(sigmoid(logit), target)`, which raises `RuntimeError: binary_cross_entropy unsafe to autocast` on GPU with `use_amp=True`. Fixed by passing the raw logit to `F.binary_cross_entropy_with_logits`.

**`sentence-transformers` version pin**: Custom `loss_fct` in `CrossEncoder.fit()` is broken in sentence_transformers ≥ 3.3.0 due to a bug in `FitMixinLoss` (the internal compatibility wrapper) — the new `CrossEncoderTrainer` passes `prompt`/`task` kwargs that `FitMixinLoss` does not accept. Pinned to `sentence-transformers>=3.0.0,<3.3.0` in `requirements.txt`.

---

## 8. Model Artifact

Model is saved at `models/cross-encoder-cv-jd-v0.3/`. Not recommended for production — v0.2 (`models/cross-encoder-cv-jd-v0.2/`) outperforms it on all metrics.

| File | Purpose |
|------|---------|
| `model.safetensors` | Fine-tuned cross-encoder weights (12-layer MiniLM, BoundaryAwareLoss) |
| `config.json` | Model architecture config |
| `tokenizer.json` + `tokenizer_config.json` | Tokenizer |

---

## 9. Findings

- `BoundaryAwareLoss` with α=0.3 does not improve label accuracy on this dataset and model combination.
- The boundary-aware ordinal penalty introduces overestimation bias, worsening MAE and RMSE.
- Using val LabelAcc for checkpoint selection is noisier than Spearman on 1,050 pairs and leads to worse test generalisation.
- `CECorrelationEvaluator` (Spearman) remains a better checkpoint selection proxy than `CELabelAccEvaluator` for regression-based models.

---

## 10. Next Steps

| Action | Rationale |
|--------|-----------|
| **v0.4: Direct 5-class classification (CrossEntropy loss)** | Eliminates regression-classification mismatch; loss directly optimises label accuracy |
| Upgrade bi-encoder to improve recall@50 beyond 47.81% | Pipeline ceiling unchanged; bottleneck for production use |
| Investigate larger base models (e.g., L-6-v2 domain-specific variants) | Architecture search rather than loss engineering |
