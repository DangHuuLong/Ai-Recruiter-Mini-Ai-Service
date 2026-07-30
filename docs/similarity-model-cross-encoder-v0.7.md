# Similarity Model — Cross-Encoder v0.7 — Discontinued

**Version**: v0.7
**Date**: July 2026
**Status**: ⏹️ Discontinued — v0.6 (Ensemble, 65.80% test LabelAcc) remains the production model
**Baseline**: v0.6 = 65.80% test LabelAcc (Ensemble of 5 seeds)

---

## 1. Objectives

| Metric | v0.6 (baseline) | v0.7 Target |
|--------|----------------|-------------|
| Test LabelAcc — single run | 64.95% | ≥65.15% |
| Test LabelAcc — ensemble | 65.80% | ≥67.0% |
| weak_match class accuracy | 52.3% | ≥60.0% |

---

## 2. Model Upgrade Experiments (FAILED)

**Initial hypothesis**: Upgrade base model from 384-dim (MiniLM) to 768-dim to increase representation capacity, particularly for `weak_match` (worst class at 52.3% in v0.6).

### Phase 1a: ms-marco-electra-base

**Model**: `cross-encoder/ms-marco-electra-base` · 768-dim · ELECTRA discriminator
**Dataset**: v0.5 · 13,350 pairs · Epochs=10 · Seed=42 · BS=16

| LR | Val LabelAcc | Test LabelAcc | MAE | vs v0.6 single |
|----|-------------|--------------|-----|----------------|
| 2e-5 | 27.10% | 25.60% | 17.54 | −39.35pp |
| 3e-5 | 27.45% | 26.05% | 17.51 | −38.90pp |
| 5e-5 | 25.70% | 24.70% | 17.63 | −40.25pp |

**Root cause**: ELECTRA is a discriminator model pre-trained to detect replaced tokens, not an encoder for relevance scoring. Incompatible with CrossEncoder Sigmoid + MSELoss fine-tuning paradigm. MAE ~17.5 indicates model outputs constant ~0.5 (50/100) for all pairs — learning signal is zero.

### Phase 1b: BAAI/bge-reranker-base

**Model**: `BAAI/bge-reranker-base` · 768-dim · xlm-roberta-base backbone
**Dataset**: v0.5 · 13,350 pairs · Epochs=10 · Seed=42 · BS=16

| LR | Val LabelAcc | Test LabelAcc | MAE | vs v0.6 single |
|----|-------------|--------------|-----|----------------|
| 2e-5 | 28.30% | 26.00% | 17.83 | −38.95pp |
| 3e-5 | 28.85% | 26.15% | 17.76 | −38.80pp |
| 5e-5 | 27.75% | 25.15% | 17.78 | −40.80pp |

**Root cause**: Identical failure pattern as ELECTRA — MAE ~17.8 = constant ~50 prediction. Both 768-dim models fail in the same way regardless of architecture. The issue is not LR or epochs.

### Model Compatibility Summary

Only `cross-encoder/ms-marco-MiniLM-L-12-v2` is compatible with the current fine-tuning setup. All models that deviate from its pre-training paradigm (pointwise reranker on MS MARCO) fail catastrophically:

| Model | Dim | Pre-training Task | Test LabelAcc |
|-------|-----|------------------|---------------|
| ms-marco-MiniLM-L-12-v2 | 384 | MS MARCO ranking (pointwise) | **64.95%** |
| ms-marco-MiniLM-L-6-v2 | 384 | MS MARCO ranking (pointwise) | 57.65% |
| qnli-distilroberta-base | 768 | QNLI classification | 25.00% |
| ms-marco-electra-base | 768 | MS MARCO + ELECTRA discriminator | 26.05% |
| BAAI/bge-reranker-base | 768 | Cross-lingual reranking | 26.15% |
| cross-encoder/ms-marco-roberta-base | 768 | MS MARCO ranking | — (401 auth) |

**Decision: Keep MiniLM-L-12-v2. Pivot to data quality improvement.**

---

## 3. v0.6 Error Analysis — Problems to Fix

From ensemble model v0.6 on 2,000 test pairs:

| Class | Accuracy | Bias Pattern | Priority |
|-------|----------|-------------|---------|
| excellent_match | 80.4% | Under-predict (MSE regression to mean) | Low |
| strong_match | 72.1% | Under-predict | Medium |
| moderate_match | 63.2% | Balanced | Low |
| **weak_match** | **52.3%** | Over-predict (model assigns higher scores) | **Critical** |
| poor_match | 59.8% | Over-predict | Medium |

`weak_match` is the clear bottleneck. Two compounding causes:
1. MSE loss inherently regresses to mean — over-predicts weak pairs as moderate
2. **Data quality**: weak_match CV texts are too short and generic → model cannot learn discriminative features

---

## 4. Dataset v0.6 Plan

### 4.1 Problems in Dataset v0.5

| Issue | Description | Impact |
|-------|-------------|--------|
| CV raw_text too short | Prompt asks 150–250 words; LLM outputs ~50–80 words in practice | High |
| weak_match low quality | Only based on seniority mismatch; missing explicit skill gap signals | High |
| Text degradation at scale | 5CV×5JD=25 pairs → LLM shortens text to fit all 25 in one response | Medium |

### 4.2 Key Changes: 3×3=9 Pairs per Batch

| | Dataset v0.5 batch | Dataset v0.6 batch |
|--|-------------------|--------------------|
| Documents per batch | 5 CV + 5 JD = 10 docs | 3 CV + 3 JD = 6 docs |
| Pairs per batch | 25 | 9 |
| CV raw_text target | 150–250 words (actual: ~50–80) | **≥280 words (hard-enforced)** |
| JD raw_text target | 100–180 words (actual: ~80–120) | **≥200 words (hard-enforced)** |
| Seniority levels | intern/junior/middle/senior/senior (5) | **junior/middle/senior (3, distinct)** |
| weak_match guidance | Seniority gap only | Seniority gap + explicit skill gap prompt |
| Word count validation | None | Hard error if below minimum |
| Pair quality constraint | Score range + boundary zone | Score range + boundary zone + **criterion score consistency check** |

**Why 3×3=9 helps quality**:
- LLM focuses on 3 documents instead of 5 → longer, more specific, less repetitive text
- Fewer pairs per prompt → more accurate scoring (LLM has bandwidth to reason about each pair)
- Natural seniority spread: junior×senior=weak, middle×middle=strong, senior×senior=excellent
- Easier to hit label distribution targets in 9 pairs than in 25

**Effect on train/val/test split**: None. The 70/15/15 split is computed over total pairs at dataset build time, independent of batch size.

### 4.3 Dataset Versioning

**Decision: Create new dataset v0.6 — do not append to v0.5.**

| | Dataset v0.5 | Dataset v0.6 |
|--|-------------|-------------|
| Raw data path | `datasets/raw/` | `datasets/raw_v2/` |
| Batch format | 5×5=25 | 3×3=9 |
| Split path | `datasets/versions/v0.5/cross_encoder/` | `datasets/versions/v0.6/cross_encoder/` |
| Status | Frozen (13,350 pairs) | In generation |
| CV text length | ~50–80 words (actual) | ≥280 words (enforced) |

v0.5 is kept frozen as the baseline. v0.6 is built fresh with higher quality constraints.

**Target**: ~8,000–10,000 pairs (~900–1,100 batches × 9 pairs), balanced at 20% per class (4 classes: weak/moderate/strong/excellent — no poor_match in synthetic data).

### 4.4 Valid Score Windows After Boundary Avoidance

```
Forbidden boundary zones: (35–45)  (55–65)  (70–80)  (85–95)

Valid windows per class:
  weak_match:      46–54   (9 values)   — use 48, 50, 52
  moderate_match:  66–69   (4 values)   — use 67, 68
  strong_match:    81–84   (4 values)   — use 82, 83
  excellent_match: 96–100  (5 values)   — use 97, 98
```

### 4.5 weak_match Quality Requirements

A valid weak_match pair must exhibit at least one of:
1. **Experience gap**: junior CV (1–2 yrs) vs senior JD (5+ yrs required) — EXPERIENCE_RELEVANCE score ≤40
2. **Skill domain mismatch within domain**: e.g., frontend-heavy CV vs backend-heavy JD — SKILLS_MATCH score ≤55
3. **Project relevance gap**: CV projects don't align with JD's core responsibilities — PROJECT_RELEVANCE ≤40

The CV must have enough specific detail (skills, tools, project descriptions) that the gap is clearly inferable — not just a short generic text.

---

## 6. Dataset v0.6 (raw_v2) Result — 91% Test LabelAcc Was Fake

### 6.1 What happened

`datasets/raw_v2` was generated (3,600 pairs, versioned as `datasets/versions/v0.6`): 3×3 batches, ≥280/200-word enforcement, perfectly balanced 20%/class (poor_match was engineered back in via a seniority-mix + skill-overlap mechanism — see 6.4 for why this deviates from this doc's original plan). Fine-tuning MiniLM-L-12-v2 with the same config as the v0.6 baseline (MSE + Spearman, 10 epochs) produced **91.3% test LabelAcc, MAE 3.70** — far above the 65.80% ceiling.

### 6.2 Why it's fake: cross-dataset evaluation

The model was evaluated on data it was **not** trained on, in both directions:

| Model trained on → tested on | MAE | RMSE | LabelAcc | Prediction range |
|---|---|---|---|---|
| raw_v2/v0.6 model → v0.5 test set (2,000 pairs) | **127.73** | 171.51 | 27.4% | mean predicted = **132** (outside valid 0–100 range) |
| v0.6-ensemble-lineage model (trained on v0.5) → raw_v2/v0.6 test set (540 pairs) | 13.54 | 17.43 | 42.8% | mean predicted = 73 (stays in-range) |

The new model doesn't just perform worse out-of-distribution — its regression head produces values outside the valid score range entirely, meaning it never learned a bounded, generalizable scoring function. It memorized this dataset's specific generation formula. The old v0.5-trained model degrades gracefully by comparison (in-range, moderately worse). **91.3% reflects fitting a narrow synthetic distribution, not CV–JD matching ability.**

### 6.3 Root cause: data homogeneity, not the boundary-zone rule

The forbidden-boundary-zone scoring rule (Section 4.4) was the first suspect but is **ruled out** by re-checking dataset history: the actual source of the 60.76% ceiling is `datasets/versions/v0.3` (7,000 pairs, `manual_ai_assisted` — corrected after initially mis-checking the `v0.2` folder), which has **59.8%** of pairs inside "boundary zones." Meanwhile the `llm_synthetic` subset of v0.5 (6,350 pairs, contributing to the 65.80%-ensemble best model) has **0%** boundary-zone pairs — identical to raw_v2 — yet did not cause catastrophic failure. So boundary-zone avoidance alone does not explain it.

Measured, confirmed differences between raw_v2 and every dataset that generalized reasonably:

| Metric | v0.3 (ceiling source, single generation pass) | v0.5 (best model, 3 mixed sources/eras) | raw_v2 (new, 91% fake) |
|---|---|---|---|
| unique JD `domain` tag values | 743 / 1,400 (53%) | — | **191 / 1,200 (16%)** |
| max repeated 6-word resume opening | 9× / 1,400 | — | **27× / 1,200** |
| unique 6-word resume openings | 83.4% | — | 72.7% |

`raw_v2` is generated entirely by one fixed script/prompt template cycling through only 18 domain strings via `random.choice` (each reused ~22× across ~400 batches). Every historical dataset that generalized reasonably had meaningfully more diversity — including v0.5, whose best-ever result (65.80%) came from **combining 3 different generation eras/sources** (`manual_ai_assisted` + `llm_relabeled` + `llm_synthetic`), not from any single "clean" generator. Heterogeneity of source/style — not a particular scoring-window rule — is what historically drove generalization.

Secondary contributor: the `REQUIRED SKILLS OVERLAP` prompt section forced an exact %-overlap-to-label mapping per pair, making score a near-linear, mechanically learnable function of keyword overlap — an easy shortcut for a small model to exploit instead of learning semantic matching.

### 6.4 Deviation from the "no poor_match" plan (Section 4.2)

This document originally planned 4 classes with **no poor_match** in synthetic data ("balanced at 20% per class (4 classes)" — note this ratio is internally inconsistent, 4 × 20% = 80%, so the target was never fully specified). In practice, poor_match was engineered back in via a same-domain seniority-gap + forced-low-skill-overlap mechanism. Section 7 redesigns poor_match generation around genuine cross-domain mismatches instead (CV domain A × JD domain B), closer to the original "cross-field pairs" idea from the v0.4 report's next-steps, and more realistic than a same-domain contrivance.

---

## 7. Dataset v0.7 (raw_v3) — Plan

Supersedes the v0.6/raw_v2 approach. Keeps everything that measurably worked (word-count enforcement, explicit weak_match criteria, water-filling label balance) and changes the parts implicated in Section 6.3:

| Change | Rationale |
|---|---|
| Batch size 3×3 → **2×2** (4 pairs/round) | Further reduces per-response document load, same direction as the original 5×5→3×3 fix |
| **5 rounds of 2×2 accumulated, scored together** (20 pairs/scoring prompt) | Keeps CV/JD generation at the proven-safe small size while cutting manual round-trips for the scoring step ~5× |
| Domain list expanded + **least-used-domain rotation** (not `random.choice` with replacement) | Directly targets the measured 191-vs-743 domain-diversity gap |
| **Cross-domain round mode** for poor_match (resume domain ≠ JD domain) | Genuine mismatch signal instead of a same-domain contrivance; also raises domain diversity per the table above |
| Removed forbidden-boundary-zone score windows — scores now continuous 0–100 | Ruled out as the cause (6.3); the real historical risk (v0.4/v0.5) was inconsistent *multi-session relabeling*, not continuous scores, which doesn't apply to this single-pass generator |
| Removed fixed `%`-overlap-to-label table — qualitative skill-authoring guidance instead | Removes the mechanical, easily-memorized shortcut identified in 6.3 |
| Seniority-tier description text varied across multiple phrasings | Reduces the measured 27×-repeated opening-sentence pattern |
| New path: `datasets/raw_v3` (raw_v2/v0.6 kept frozen as a documented negative result) | Matches this project's established versioning convention (v0.1–v0.5 kept frozen); enables a clean ablation later (v0.6 alone vs v0.7 alone vs combined) |

---

## 8. Progress Checklist

### Data Pipeline
- [x] v0.6 error analysis completed — weak_match identified as bottleneck
- [x] Model upgrade path explored (ELECTRA, BGE) — both failed, MiniLM confirmed as only viable base
- [x] Update `generate_synthetic_data.py` (v1): 3×3=9 batch, ≥280-word CV enforcement, weak_match guidance, criterion score consistency check
- [x] Create `datasets/raw_v2/` and generate 3,600 pairs (balanced 20%/class)
- [x] Fine-tune on raw_v2/v0.6 → 91.3% test LabelAcc
- [x] Cross-dataset evaluation reveals the result is fake (Section 6.2) — root-caused to data homogeneity (Section 6.3)
- [ ] Update `generate_synthetic_data.py` (v2): 2×2 batch, 5-round accumulation, expanded domains, cross-domain poor_match mode, continuous scores, qualitative skill guidance
- [ ] Create `datasets/raw_v3/` and start generation
- [ ] Build `datasets/versions/v0.7/cross_encoder/` split (70/15/15)
- [ ] Fine-tune on raw_v3 alone, then on raw_v2+raw_v3 combined — compare both against the v0.5/v0.6 baseline AND via cross-dataset evaluation (not just in-domain test LabelAcc)

### Model Training
- [ ] Fine-tune MiniLM-L-12-v2 on raw_v3 (LR=5e-5, 15 epochs, BS=16)
- [ ] Cross-evaluate: raw_v3 model → v0.5 test set, and v0.5-lineage model → raw_v3 test set (repeat the Section 6.2 check — a healthy result should NOT reproduce the MAE-127 / out-of-range collapse)
- [ ] If in-domain and cross-dataset results are both healthy: run ensemble with 5 seeds
- [ ] Update this document with final results

---

## 9. Discontinuation

Stopped generating `raw_v3` at 760 pairs. Before stopping, dataset validation surfaced a real
generator bug: duplicate IDs across rounds (`jd_375`/`jd_376`/`resume_375`/`resume_376` each
appeared twice with unrelated domain content — a same-domain Round 3 batch and a later
Mobile Engineering batch collided on the same numeric suffix). This would have silently
corrupted any `cross_encoder` split built from the raw data (dict keyed by id drops the
first record). Decision made to end the v0.7 experiment here rather than fix the generator
and continue — v0.6 (Ensemble of 5 seeds, 65.80% test LabelAcc) is kept as the final/production
model. `datasets/raw_v3` is left as-is (not versioned into `datasets/versions/v0.7/`, not
trained on).

**Last Updated**: July 2026
**Status**: ⏹️ Discontinued — v0.6 (Ensemble, 65.80% test LabelAcc) is the final model for this project
