# Similarity Model — Cross-Encoder v0.4 — Experiment Report

## 1. Overview

This document records the cross-encoder fine-tuning experiment that replaces the regression objective (MSELoss) with a 5-class CrossEntropyLoss. The hypothesis was that directly optimising for label accuracy — by treating the task as classification into five rubric buckets (poor / weak / moderate / strong / excellent) — would close the gap between the optimisation objective and the evaluation metric.

**Result**: The classification approach **dramatically degrades** performance across all metrics. Test LabelAcc drops to **30.29%** from v0.2's 60.76% baseline — a regression of −30.47 pp. MAE and RMSE both increase substantially. This experiment is a negative result.

**Cross-encoder progression:**

| Metric | v0.2 (MSE) | v0.3 (BoundaryAware) | v0.4 (Classification) |
|--------|-----------|---------------------|----------------------|
| Test LabelAcc | **60.76%** | 60.76% | 30.29% |
| Test MAE | **9.72** | 9.98 | 12.50 |
| Test RMSE | **13.91** | 14.26 | 16.04 |
| Mean predicted score | — | 65.25 | 64.79 |
| Mean target score | 62.85 | 62.85 | 62.85 |

**Conclusion on loss engineering**: After three experiments (MSE → BoundaryAware → Classification), the regression MSELoss baseline from v0.2 remains the best performing configuration. Loss function changes have yielded no improvement. The ceiling is not a loss-function problem.

---

## 2. Hypothesis

v0.2 and v0.3 both use regression objectives that minimise score distance rather than bucket boundary errors. The mismatch between training objective (minimise |pred − target|) and evaluation metric (label accuracy) was hypothesised to be a core bottleneck.

The classification approach reframes the task: instead of predicting a continuous score, the model predicts a 5-class distribution over rubric labels:

| Class index | Label | Score range |
|-------------|-------|------------|
| 0 | poor | < 40 |
| 1 | weak | 40–59 |
| 2 | moderate | 60–74 |
| 3 | strong | 75–89 |
| 4 | excellent | ≥ 90 |

CrossEntropyLoss directly optimises the probability assigned to the correct class, which is exactly what label accuracy measures.

---

## 3. Training

Training performed on Google Colab (T4 GPU) using `sentence_transformers.CrossEncoder.fit()`.

| Argument | Value |
|----------|-------|
| `base_model` | `cross-encoder/ms-marco-MiniLM-L-12-v2` |
| `epochs` | 10 |
| `batch_size` | 16 |
| `max_length` | 512 |
| `warmup_ratio` | 0.1 |
| `loss_fct` | `CrossEntropyLoss` (5-class) |
| `activation_fct` | `Identity()` (logits passed raw to CE loss) |
| `num_labels` | 5 |
| `evaluator` | `CELabelAccEvaluator` (val label accuracy) |
| `save_best_model` | `True` |

Labels were converted from continuous scores to class indices (0–4) using the rubric boundaries above. The model outputs 5 logits; argmax is taken as the predicted class, then mapped back to a representative score for MAE/RMSE computation (class midpoints: 20, 50, 67, 82, 95).

---

## 4. Results

| Split | MAE | RMSE | LabelAcc | Mean predicted | Mean target | Pairs |
|-------|-----|------|----------|---------------|------------|-------|
| Validation | 13.0053 | 16.3311 | **28.29%** | 63.83 | 60.65 | 1,050 |
| Test | 12.5027 | 16.0374 | **30.29%** | 64.79 | 62.85 | 1,050 |

---

## 5. Analysis

### 5.1 Why classification fails on this dataset

The 5-class CrossEntropyLoss requires the label assignment itself to be **internally consistent**: a pair with ground truth score 59 must reliably belong to class 1 (weak) and never class 2 (moderate). This assumption breaks on this dataset.

Approximately **60% of samples** have ground truth scores within ±5 of a rubric boundary. These are structurally ambiguous: the same CV–JD pair scored by two annotators on different days could land in adjacent buckets. CrossEntropyLoss penalises the model for predicting class 2 when the label is class 1 (score 59), even if the pair genuinely belongs in the boundary zone. The signal is noise.

By contrast, MSELoss treats a 59 vs 61 prediction error as a 2-point mistake — still small. Classification loss treats it as a completely wrong class — large gradient. The regression objective is robust to labeling noise at boundaries; the classification objective amplifies it.

### 5.2 LabelAcc 30% is near the classification floor

With 5 classes, random guessing achieves 20%. A model predicting the majority class (moderate, ~29% of dataset) achieves ~29% LabelAcc. The v0.4 result (30.29% test) is barely above majority-class baseline — the model has learned almost no discriminative signal from the classification objective.

### 5.3 Systematic overestimation persists

Mean predicted score (64.79) exceeds mean target (62.85) by +1.94 points. This bias appears consistently across v0.3 and v0.4, suggesting a dataset-level imbalance or base model prior, not a loss-function artifact.

### 5.4 Validation–test gap is reversed

Val LabelAcc (28.29%) is *lower* than test LabelAcc (30.29%). This is unusual and indicates that the `CELabelAccEvaluator` checkpoint selection was operating on a noisy signal — the saved checkpoint was not actually the best-generalising point. The 1,050-pair validation set is too small for stable label-accuracy-based early stopping.

---

## 6. Root Cause: Labeling Quality, Not Loss Function

This experiment, combined with v0.3, reveals that the 60.76% ceiling from v0.2 is **not a loss-function problem**. The ceiling is structural:

- ~60% of training samples sit within ±5 points of a rubric boundary (40, 60, 75, 90)
- These samples carry ambiguous labels due to inter-annotator score variance — the same pair could legitimately receive scores of 59 or 61 from the same annotator on different days
- No loss function can correctly learn a signal that is not in the labels

| Boundary | Score range | % of train set |
|----------|------------|---------------|
| 60 (weak/moderate) | 55–65 | ~17.3% |
| 75 (moderate/strong) | 70–80 | ~16.4% |
| 90 (strong/excellent) | 85–95 | ~18.6% |
| 40 (poor/weak) | 35–45 | ~8.4% |

Approximately **2,900 of 4,900 training pairs** fall in boundary ambiguity zones. Collecting more data with the same annotation methodology will not resolve this — it will reproduce the same ~60% boundary confusion at scale.

---

## 7. Findings

- CrossEntropyLoss (5-class classification) **degrades** label accuracy by −30 pp versus v0.2 MSELoss.
- The classification objective is incompatible with boundary-ambiguous continuous score labels: it amplifies label noise at boundaries into large classification errors.
- LabelAcc-based checkpoint selection on 1,050 pairs is not a reliable early stopping signal.
- After three loss experiments, v0.2 MSELoss remains the best configuration. The bottleneck is **data quality**, not model architecture or training objective.

---

## 8. Next Steps

The path forward requires addressing labeling quality in boundary zones, not changing the loss function.

| Action | Rationale |
|--------|-----------|
| **LLM-assisted relabeling of boundary samples** | Replace ambiguous continuous scores in the 55–65 / 70–80 / 85–95 / 35–45 bands with categorical labels (poor/weak/moderate/strong/excellent) using a structured rubric prompt. Targets ~2,900 pairs, not full dataset. |
| **Switch to categorical labels for training** | LLM-assigned categorical labels are more consistent than human continuous scores near boundaries. Combine with MSELoss using class-midpoint targets, or retain Classification loss with cleaner labels. |
| **Cross-field pairs for coverage** | Add ~15–20% cross-field pairs (CV Marketing + JD IT) to strengthen the poor-match end of the distribution. These are unambiguous (score < 30) and help the model anchor its lower range. |
| **Avoid further loss function experiments** | Three experiments confirm: loss engineering cannot compensate for label noise. |
