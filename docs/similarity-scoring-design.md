# Similarity Scoring Design

## 1. Overview

The CV-JD scoring system uses two scoring approaches that can operate
independently or in combination:

- **Deterministic scorer** — rule-based, fully explainable, always available
- **Model-assisted scorer** — embedding similarity, requires a loaded model

The key design decision is **blend, not replace**: the ML score is combined with
the rule-based score using a configurable weight, rather than replacing it.
The rule-based scorer remains the authoritative source for all explainability
output (criteria breakdown, skill gaps, interview questions). The ML score only
adjusts the final numeric value.

---

## 2. The Two Approaches

### Deterministic Scorer

Implemented in [app/scorers/cv_jd_scorer.py](../app/scorers/cv_jd_scorer.py).

Computes `overall_score` as a weighted sum of five criteria:

| Criterion | Default weight | What it measures |
|-----------|---------------|-----------------|
| `SKILLS_MATCH` | 35% | Overlap between resume skills and JD required/preferred skills. Core skills weighted 1.25×. Partial matches via token overlap credited at 45%. |
| `EXPERIENCE_RELEVANCE` | 30% | Years of experience vs JD minimum (60%) combined with required-skill evidence in experience entries (40%). Score capped for underqualified profiles. |
| `PROJECT_RELEVANCE` | 15% | Required-skill signal in project descriptions. Uses best-project score (75%) blended with average (25%) to reward depth without ignoring breadth. |
| `EDUCATION_CERTIFICATION` | 10% | Degree field token overlap with JD education requirement. CS/IT degree detection via a fixed term list. |
| `KEYWORD_DOMAIN_ALIGNMENT` | 10% | Resume token coverage of meaningful JD domain keywords (generic words like "team", "build", "system" excluded). |

**Properties:**
- No external dependencies. Runs without GPU or network access.
- Fully deterministic: same input always produces the same output.
- Returns a structured `EvaluationResult` with criteria scores, matched/missing
  skills, evidence strings, skill gap summary, and interview questions.
- Criteria weights are configurable per-request via `config.criteria`.

### Model-Assisted Scorer

Implemented in [app/scorers/similarity_scorer.py](../app/scorers/similarity_scorer.py).

**Updated (v0.6 integration)**: uses a `sentence_transformers.CrossEncoder` —
a pointwise pair scorer that encodes the CV/JD pair *jointly* (not as two
separate embeddings compared by cosine similarity) and outputs a single
relevance score directly:

```
(CV text, JD text) → CrossEncoder.predict() → raw score → clip[0,1] → ×100
```

Trained via `training/fine_tune_cross_encoder_*` (Colab notebooks, see
`docs/similarity-model-cross-encoder-v0.6.md`) on `datasets/versions/v0.5`
(13,350 pairs), reaching **65.80% test LabelAcc** as a 5-seed ensemble
(single-checkpoint runs: 62.55%–64.95%). Only one checkpoint is committed
locally (`models/cross-encoder-cv-jd-v0.6`), so the integrated model runs at
single-checkpoint accuracy, not the full ensemble figure, unless the other
4 seed checkpoints are also placed alongside it and ensembling is added to
`similarity_scorer.py`.

> Prior to this integration, this scorer used a bi-encoder
> (`sentence-transformers/all-MiniLM-L6-v2` fine-tune, `models/fine-tuned-miniLM-v0.2`)
> encoding CV and JD separately and comparing via cosine similarity. That
> line of experimentation (`docs/similarity-model-v0.2.md`, `v0.3.md`) is
> kept as historical record but is no longer what's wired into the live
> scoring path.

Text representation (unchanged from the bi-encoder version):
- **CV text**: summary, skills, experience (role, company, responsibilities,
  technologies), projects (name, description, technologies), education, certifications
- **JD text**: title, seniority, responsibilities, requirements, nice-to-have,
  domain keywords

**Properties:**
- Captures semantic similarity that keyword matching misses (e.g., "Python
  developer" ↔ "backend engineer using Django") — and, being a pointwise
  pair scorer rather than independently-embedded cosine similarity, can
  model cross-text interactions a bi-encoder cannot.
- Not explainable: produces a single score with no breakdown.
- Requires `sentence-transformers` installed and a model loaded.
- Non-deterministic across model versions.
- Slower per-request than a bi-encoder (no precomputable embeddings — every
  CV/JD pair requires a fresh forward pass), which matters only if this
  scorer is ever used for bulk ranking rather than single-application scoring.

---

## 3. Why Blend Instead of Replace

Replacing the rule-based score entirely with ML would lose the structured
explainability output that the API returns — criteria breakdown, per-skill
match/missing status, interview questions. These are consumed by the frontend
and stored in the database per evaluation.

A **blend** preserves all of that while letting the ML score shift the
final numeric value toward a more semantically-informed result.

| Approach | Explainability | Accuracy | ML dependency |
|----------|---------------|----------|--------------|
| Rule-based only | Full | Limited by keyword overlap | None |
| ML only | None | Higher semantic coverage | Required |
| Blend (current) | Full (rule-based output unchanged) | Improved by ML weight | Optional |

The blend approach also lets the two scores act as a sanity check on each other:
a large divergence between rule score and ML score is a signal worth
investigating, not silently discarding.

---

## 4. Blending Formula

```
final_score = (1 − weight) × rule_score + weight × ml_score
```

`weight` is `SIMILARITY_SCORING_WEIGHT` (default `0.0`).

Examples:

| Rule score | ML score | Weight | Final score |
|-----------|---------|--------|------------|
| 70 | 90 | 0.0 | 70.0 (ML disabled) |
| 70 | 90 | 0.3 | 76.0 |
| 70 | 90 | 0.5 | 80.0 |
| 70 | 90 | 1.0 | 90.0 |

Only `overall_score` is blended. Criteria scores, skill lists, evidence strings,
and interview questions are always from the rule-based scorer.

The response field `similarity_score` is populated when blending occurs, so
the caller can observe both the ML score and the blended result.

---

## 5. Backward Compatibility

`SIMILARITY_SCORING_WEIGHT` defaults to `0.0`. At this value the code path
reads:

```python
if cfg.scoring_weight <= 0.0 or cfg.fallback_mode == "rule_only":
    return rule_result
```

The ML scorer is never called and no model is loaded. The response is identical
to what was returned before ML scoring was introduced. Deployments that do not
set the weight variable are unaffected.

This means the feature can be safely deployed before the model artifact is
present, and enabled incrementally per environment by setting the weight in `.env`.

---

## 6. Score Threshold Guard

`SIMILARITY_SCORE_THRESHOLD` (default `0.0`) sets a minimum ML confidence level
before the blend is applied. If the ML score is below the threshold, the
rule-based score is returned unchanged:

```python
if ml_score < cfg.score_threshold:
    logger.info("ML score %.1f below threshold %.1f, skipping blend", ...)
    return rule_result
```

**Motivation**: a very low embedding similarity score (e.g., 10–15) usually
indicates the model is uncertain or the CV/JD text is malformed. Blending such
a score with a rule-based score of 65 would produce a misleadingly low result.
The threshold acts as a confidence gate.

Default `0.0` disables filtering — fully backward compatible. A starting value
of `20.0` is recommended when the fine-tuned model is active.

---

## 7. Fallback Modes

Two modes control what happens when the fine-tuned model is unavailable:

### `base_model` (default)

If `SIMILARITY_MODEL_PATH` is empty or the directory does not contain
`config.json`, the model loader falls back to `SIMILARITY_BASE_MODEL`
(`cross-encoder/ms-marco-MiniLM-L-12-v2` — a pretrained, NOT fine-tuned,
CrossEncoder; scores will be far less accurate than the fine-tuned v0.6
checkpoint). ML scoring still runs using the base model.

Use this in development or when the fine-tuned artifact has not been downloaded.

### `rule_only`

Disables ML entirely. The model is never loaded. Scoring returns the
rule-based result regardless of weight.

```env
SIMILARITY_FALLBACK_MODE=rule_only
```

Use this to turn off ML scoring without changing `scoring_weight` — for example
during an incident, in a CI environment where `sentence-transformers` is not
installed, or in a staging environment where the model file is absent.

---

## 8. Decision Flow

```
score_application(request)
    │
    ├─ rule_result = score_application_mock(request)   ← always runs
    │
    ├─ scoring_weight == 0.0 OR fallback_mode == rule_only?
    │       └─ Yes → return rule_result
    │
    ├─ ml_score = score_cv_jd_similarity(request)
    │       └─ exception → log warning → return rule_result
    │
    ├─ ml_score < score_threshold?
    │       └─ Yes → return rule_result
    │
    └─ blended = (1 − weight) × rule_score + weight × ml_score
            └─ return rule_result with overall_score=blended, similarity_score=ml_score
```

---

## 9. Tradeoffs and Known Limitations

**Deterministic scorer limitations:**

- Keyword matching misses semantic synonyms. "FastAPI" and "REST API framework"
  do not match even if one implies the other.
- Token overlap rewards resumes that copy JD wording, not necessarily real skill
  depth.
- Fixed criteria weights (35/30/15/10/10) are not calibrated to actual hiring
  outcomes — they reflect a reasonable prior, not empirical data.

**ML scorer limitations:**

- Base model (`cross-encoder/ms-marco-MiniLM-L-12-v2`) is English-optimised.
  Mixed Vietnamese/English text produces less reliable scores.
- Learned pairwise relevance measures general semantic/domain closeness, not
  strictly recruitability. A resume about cooking and a JD for a chef
  may score higher than intended.
- The model produces a single holistic score with no breakdown. There is no way
  to tell which part of the resume drove the score.
- Only one of the 5 ensemble seed checkpoints is integrated (see above) —
  running at ~63-65% single-run LabelAcc rather than the 65.80% ensemble figure.
- Trained on synthetic LLM-generated CV/JD pairs (`datasets/versions/v0.5`),
  not real-world hiring outcomes.

**Blend limitation:**

- The blend weight is global. A weight of 0.3 applies to all request types
  equally, regardless of domain, language, or data quality.

---

## 10. Configuration Reference

See [model-integration-guide.md](model-integration-guide.md) for the full
env variable reference and setup instructions.

| Variable | Purpose |
|----------|---------|
| `SIMILARITY_SCORING_WEIGHT` | Blend weight. `0.0` = ML off (default). |
| `SIMILARITY_SCORE_THRESHOLD` | Minimum ML score to allow blending. |
| `SIMILARITY_FALLBACK_MODE` | `base_model` or `rule_only`. |
| `SIMILARITY_MODEL_PATH` | Path to fine-tuned model directory. |
| `SIMILARITY_MODEL_VERSION` | Informational label for logging. |
