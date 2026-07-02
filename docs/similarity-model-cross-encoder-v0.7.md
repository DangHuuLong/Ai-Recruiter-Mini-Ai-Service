# Similarity Model — Cross-Encoder v0.7 — In Progress

**Version**: v0.7
**Date**: July 2026
**Status**: 🔄 In Progress
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

## 5. Progress Checklist

### Data Pipeline
- [x] v0.6 error analysis completed — weak_match identified as bottleneck
- [x] Model upgrade path explored (ELECTRA, BGE) — both failed, MiniLM confirmed as only viable base
- [ ] Update `generate_synthetic_data.py`: 3×3=9 batch, ≥280-word CV enforcement, weak_match guidance
- [ ] Create `datasets/raw_v2/` and start generation
- [ ] Target: ~1,000 batches × 9 pairs = ~9,000 pairs, balanced across 4 classes
- [ ] Build `datasets/versions/v0.6/cross_encoder/` split (70/15/15)
- [ ] Quality verification: word count stats, label distribution, manual spot-check of 50 pairs

### Model Training (after dataset v0.6 is ready)
- [ ] Fine-tune MiniLM-L-12-v2 on v0.6 dataset (LR=5e-5, 15 epochs, BS=16)
- [ ] Compare: v0.7 model (dataset v0.6) vs v0.6 baseline (dataset v0.5)
- [ ] If single run ≥65.15%: run ensemble with 5 seeds
- [ ] Update this document with final results

---

**Last Updated**: July 2026
**Status**: 🔄 In Progress — updating data generation pipeline
