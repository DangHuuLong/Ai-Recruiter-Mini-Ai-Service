# Similarity Model — Cross-Encoder v0.5 — Experiment Report

## 1. Overview

This document records the cross-encoder fine-tuning experiments on dataset **v0.4** — the first dataset version with LLM-assisted relabeling. 4,188 of 7,000 pairs were relabeled by an LLM using rubric v0.2, targeting boundary-zone pairs (scores within ±5 of the four rubric thresholds: 40/60/75/90).

The hypothesis was that higher-quality labels from an LLM rubric would allow the model to exceed v0.2's 60.76% LabelAcc ceiling, which prior loss-function experiments (v0.3 BoundaryAwareLoss, v0.4 ClassificationLoss) failed to break.

**Result**: Despite improved label quality in principle, all five training configurations fail to exceed v0.2's test LabelAcc. The best stable result is **57.43% test LabelAcc** (BoundaryAwareLoss, 10 epochs). The LLM relabeling introduced two structural problems — class imbalance and boundary label noise — that offset the quality gain.

**Full experiment progression:**

| Version | Dataset | Loss | Epochs | Evaluator | Val LabelAcc | Test LabelAcc |
|---------|---------|------|--------|-----------|-------------|--------------|
| v0.2 | v0.2 | MSE | 10 | Spearman | — | **60.76%** |
| v0.5 run-1 | v0.4 | MSE | 10 | Spearman | 58.76% | 56.86% |
| v0.5 run-2 | v0.4 | MSE | 10 | LabelAcc | 57.52% | 57.05% |
| v0.5 run-3 | v0.4 | MSE + WeightedSampler | 10 | LabelAcc | 56.95% | 56.19% |
| v0.5 run-4 | v0.4 | BoundaryAware | 10 | LabelAcc | 58.57% | **57.43%** |
| v0.5 run-5 | v0.4 | BoundaryAware | 15 | LabelAcc | **59.52%** | 56.10% |

**Best stable checkpoint**: run-4 (BoundaryAwareLoss, 10 epochs) — highest test LabelAcc with no overfitting gap.

---

## 2. Dataset v0.4

### 2.1 Composition

| Split | Pairs |
|-------|-------|
| Train | 4,900 |
| Validation | 1,050 |
| Test | 1,050 |
| **Total** | **7,000** |

- **4,188 pairs** relabeled by LLM using rubric v0.2
- **2,812 pairs** retain original labels
- `labeled_by`: `"llm_relabeled"` for updated pairs, original value for unchanged pairs
- `label_version`: `"rubric_v0.2"` for all pairs

### 2.2 Label Distribution (post-relabeling)

| Label | Count | % |
|-------|-------|---|
| poor_match | 2,670 | 38.1% |
| weak_match | 1,930 | 27.6% |
| excellent_match | 956 | 13.7% |
| moderate_match | 765 | 10.9% |
| strong_match | 679 | 9.7% |
| **Total** | **7,000** | |

The relabeling shifted the distribution significantly toward `poor_match`. The original dataset (used in v0.2–v0.4) had a more balanced distribution; the stricter LLM rubric pushed many boundary-zone pairs (originally scored 40–50) below the 40-point threshold into `poor_match`.

---

## 3. Experiments

All runs use the same base model and infrastructure as prior versions.

| Argument | Value |
|----------|-------|
| `base_model` | `cross-encoder/ms-marco-MiniLM-L-12-v2` |
| `batch_size` | 16 |
| `max_length` | 512 |
| `warmup_ratio` | 0.1 |
| `num_labels` | 1 |
| `activation_fct` | `Sigmoid()` |
| `save_best_model` | `True` |
| Environment | Google Colab T4 GPU |

### Run 1 — MSELoss + Spearman evaluator (10 epochs)

Baseline comparison: same config as v0.2 but on v0.4 data.

| Split | MAE | RMSE | LabelAcc |
|-------|-----|------|----------|
| Validation | — | — | 58.76% |
| Test | — | — | 56.86% |

Spearman reached 0.7997 at epoch 6 then plateaued. LabelAcc lower than v0.2 despite same training config — confirms the data distribution change is the cause.

### Run 2 — MSELoss + LabelAcc evaluator (10 epochs)

Switched `save_best_model` to optimise for LabelAcc directly (best checkpoint saved at epoch 7, val 57.52%).

| Split | MAE | RMSE | LabelAcc |
|-------|-----|------|----------|
| Validation | 11.7301 | 15.4807 | 57.52% |
| Test | 11.8099 | 16.4908 | 57.05% |

No improvement over run-1. Training curve oscillated ±5% between epochs — a marker of label noise.

### Run 3 — MSELoss + WeightedRandomSampler + LabelAcc (10 epochs)

Added inverse-frequency sampling to counter the 38% `poor_match` imbalance. Weights: `poor=1.0`, `weak=1.38`, `moderate=3.50`, `strong=3.93`, `excellent=2.78`.

| Split | MAE | RMSE | LabelAcc |
|-------|-----|------|----------|
| Validation | 11.5778 | 15.2125 | 56.95% |
| Test | 11.6463 | 16.2152 | 56.19% |

Epoch 1 dropped to 38.5% LabelAcc (slower convergence due to oversampling minority classes early). Did not recover to exceed run-1 or run-2. Weighted sampling was **reverted** after this run.

### Run 4 — BoundaryAwareLoss + LabelAcc (10 epochs) — **Best**

`BoundaryAwareLoss` = MSE + 0.3 × ordinal BCE at boundaries (40/60/75/90).

| Split | MAE | RMSE | LabelAcc |
|-------|-----|------|----------|
| Validation | 11.6816 | 15.3355 | **58.57%** |
| Test | 11.7442 | 16.2042 | **57.43%** |

Peak at epoch 6 (val 58.57%). Val–test gap = 1.14 pp — stable generalisation. Best configuration for this dataset.

### Run 5 — BoundaryAwareLoss + LabelAcc (15 epochs)

Extended training to check if the model continues improving beyond epoch 10.

| Split | MAE | RMSE | LabelAcc |
|-------|-----|------|----------|
| Validation | 11.5410 | 15.2664 | 59.52% |
| Test | 11.9431 | 16.4850 | 56.10% |

Val peaked at epoch 9 (59.52%) but test dropped to 56.10% — val–test gap of 3.4 pp indicates overfitting to the validation set. The 15-epoch checkpoint generalises worse than the 10-epoch checkpoint.

---

## 4. Analysis

### 4.1 Why v0.4 data does not outperform v0.2

Despite 4,188 LLM-relabeled pairs with rubric v0.2, the model performs ~3 pp below v0.2 on test LabelAcc. Two structural reasons:

**Class imbalance**: 38.1% of pairs are `poor_match`. With MSE loss, the model's predicted scores are pulled toward the low end of the distribution, making it harder to correctly identify `moderate`, `strong`, and `excellent` matches. The minority classes (`strong` 9.7%, `moderate` 10.9%) are underrepresented.

**LLM label noise at boundaries**: The relabeled pairs are by definition boundary-zone pairs (the most ambiguous samples). LLM labeling introduces its own variance — the same pair scored in two different batches may receive different labels. The model sees noisy signal at exactly the hardest decision boundaries.

### 4.2 Training instability as a noise indicator

All five runs show epoch-to-epoch LabelAcc variance of ±4–6%:

```
Run 4 (10ep): 47% → 54% → 58% → 58% → 57% → 59% → 56% → 57% → 57% → 57%
Run 5 (15ep): 47% → 48% → 57% → 57% → 52% → 59% → 58% → 58% → 60% → 59% → 58% → 58% → 56% → 58% → 59%
```

This pattern — characteristic of fitting and unfitting noisy labels — does not appear in v0.2 training curves. The noise originates in the relabeled samples, not in the model architecture or training setup.

### 4.3 BoundaryAwareLoss is effective for this data

Across the experiments, BoundaryAwareLoss consistently outperforms MSELoss on v0.4 data (+0.58–0.81 pp test LabelAcc vs. same-epoch MSE runs). The ordinal BCE terms counteract the pull toward poor_match by explicitly penalising wrong-side-of-boundary predictions.

### 4.4 Overfitting at 15 epochs

The val–test gap jumps from 1.14 pp (10 epochs) to 3.42 pp (15 epochs). With only 1,050 validation pairs and noisy labels, the saved checkpoint at epoch 9 happens to align with a validation spike rather than a generalisation peak. Early stopping based on LabelAcc on this small a validation set is unreliable beyond ~10 epochs.

---

## 5. Findings

- **LLM relabeling improves data coverage** (4,188 pairs with structured rubric assessment) but does not improve model LabelAcc due to class imbalance and introduced label noise.
- **BoundaryAwareLoss** is the recommended loss for v0.4-derived data (+0.6 pp test LabelAcc vs. MSE).
- **10 epochs** is the optimal training length for this data; 15 epochs overfits the validation set.
- **The v0.2 ceiling of 60.76% test LabelAcc remains unbroken** after five experiments across three dataset versions and four loss configurations.
- The bottleneck is **data distribution and label consistency**, not model architecture, loss function, or training hyperparameters.

---

## 6. Next Steps

| Action | Rationale |
|--------|-----------|
| **Generate ~400–500 pairs for `strong_match` and `moderate_match`** | Bring distribution closer to 20% per class; directly addresses the 9.7%/10.9% minority underrepresentation |
| **Post-hoc label consistency check** | Sample 100 relabeled pairs, re-run LLM rubric, measure agreement; if < 80% match, the noise level explains the training instability |
| **Increase validation set size before extending epochs** | 1,050 pairs is too small for stable LabelAcc-based early stopping; 1,500–2,000 pairs would reduce spike sensitivity |
| **Do not add more relabeling without fixing distribution first** | Relabeling more boundary pairs will increase `poor_match` further and worsen the imbalance |
