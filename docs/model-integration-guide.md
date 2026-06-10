# Model Integration Guide

## 1. Overview

This guide explains how to set up the CV-JD similarity model for local inference:
how to download the fine-tuned artifact, configure environment variables, and
control fallback behavior.

The similarity scoring layer is **off by default** (`scoring_weight = 0.0`).
No model is loaded unless you explicitly enable it.

---

## 2. Prerequisites

Install `sentence-transformers` before using the fine-tuned or base model:

```bash
pip install sentence-transformers
```

If `sentence-transformers` is not installed, calling the similarity scorer raises
`RuntimeError: sentence-transformers is not installed` and the service falls back
to rule-based scoring automatically.

---

## 3. Downloading the Fine-Tuned Model

The model artifact is not committed to git. Download it from Google Drive and
place it inside the `models/` directory at the project root.

Expected path after download:

```
models/
  fine-tuned-miniLM-v0.2-ep3/
    config.json
    model.safetensors
    modules.json
    tokenizer.json
    tokenizer_config.json
    vocab.txt
    1_Pooling/
    2_Normalize/
```

The loader validates the path by checking that `config.json` exists inside the
directory. If the directory is missing or `config.json` is absent, the loader
falls back to the base model automatically and logs a warning.

---

## 4. Environment Variables

All similarity settings are read from `.env` (or environment) via `Settings` in
[app/core/config.py](../app/core/config.py), then surfaced through
[`SimilarityConfig`](../app/ml/similarity_config.py).

| Variable | Default | Description |
|----------|---------|-------------|
| `SIMILARITY_MODEL_PATH` | `""` | Path to fine-tuned model directory. Empty = use base model. |
| `SIMILARITY_BASE_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace model name used when no fine-tuned path is set or the path is invalid. |
| `SIMILARITY_MODEL_VERSION` | `""` | Informational label only. Used for logging and reports. |
| `SIMILARITY_SCORING_WEIGHT` | `0.0` | Blend weight for ML score. `0.0` disables ML scoring entirely. `1.0` uses ML score only. |
| `SIMILARITY_SCORE_THRESHOLD` | `0.0` | Minimum ML score to allow blending. Below this value, rule-based score is returned. `0.0` disables the threshold guard. |
| `SIMILARITY_FALLBACK_MODE` | `base_model` | `base_model` = use base model when fine-tuned path is unavailable. `rule_only` = disable ML entirely regardless of other settings. |

### Minimal config for fine-tuned model with 30% ML blend

```env
SIMILARITY_MODEL_PATH=models/fine-tuned-miniLM-v0.2-ep3
SIMILARITY_MODEL_VERSION=v0.2
SIMILARITY_SCORING_WEIGHT=0.3
```

### Full config example

```env
SIMILARITY_MODEL_PATH=models/fine-tuned-miniLM-v0.2-ep3
SIMILARITY_BASE_MODEL=sentence-transformers/all-MiniLM-L6-v2
SIMILARITY_MODEL_VERSION=v0.2
SIMILARITY_SCORING_WEIGHT=0.3
SIMILARITY_SCORE_THRESHOLD=20.0
SIMILARITY_FALLBACK_MODE=base_model
```

---

## 5. Scoring Weight Behavior

`SIMILARITY_SCORING_WEIGHT` controls how the final score is computed:

```
final_score = (1 - weight) × rule_score + weight × ml_score
```

| Weight | Behavior |
|--------|----------|
| `0.0` | ML scoring disabled. Rule-based score returned as-is. |
| `0.3` | 70% rule + 30% ML blend. |
| `1.0` | ML score only (rule-based score fully replaced). |

The response field `similarity_score` is populated only when blending actually
occurs. If ML is skipped for any reason, `similarity_score` is `null`.

---

## 6. Score Threshold

`SIMILARITY_SCORE_THRESHOLD` sets a minimum ML confidence level for blending to
proceed. If the ML score is below the threshold, the rule-based score is returned
unchanged, and a log line is emitted at `INFO` level:

```
ML score 18.5 below threshold 20.0, skipping blend
```

A threshold of `0.0` (default) means no filtering — blending always proceeds
when `scoring_weight > 0`.

Use this to avoid low-confidence ML scores distorting the final result.
Recommended starting value: `20.0`.

---

## 7. Fallback Mode

`SIMILARITY_FALLBACK_MODE` controls what happens when the fine-tuned model is
unavailable or ML scoring is unwanted.

### `base_model` (default)

If `SIMILARITY_MODEL_PATH` is empty or the directory is missing, the loader
falls back to `SIMILARITY_BASE_MODEL` automatically:

```
SIMILARITY_MODEL_PATH not set, using base model 'sentence-transformers/all-MiniLM-L6-v2'
# or
Fine-tuned model not found at 'models/...', falling back to base model '...'
```

ML scoring still runs using the base model.

### `rule_only`

Disables ML entirely. The model is never loaded. `score_application` skips the
ML path and returns the rule-based result directly, regardless of `scoring_weight`.

```env
SIMILARITY_FALLBACK_MODE=rule_only
```

Use this in environments where `sentence-transformers` is not installed, or
during incidents where ML scoring should be turned off without redeploying.

---

## 8. Model Load Path

The load decision in [`app/ml/similarity_model.py`](../app/ml/similarity_model.py):

```
SIMILARITY_FALLBACK_MODE == rule_only?
  └─ Yes → raise RuntimeError (model never loaded)
  └─ No  → check SIMILARITY_MODEL_PATH
              └─ path set AND config.json exists → load fine-tuned model
              └─ otherwise → log warning/info → load base model
```

The loaded model is cached with `@lru_cache(maxsize=1)` — it is loaded once per
process and reused for all requests.

---

## 9. Verifying the Setup

Run the evaluation pipeline to confirm the model loads and scores correctly:

```bash
python -m training.evaluate_similarity_pipeline \
  --models base:sentence-transformers/all-MiniLM-L6-v2 \
           v0.2-ep3:models/fine-tuned-miniLM-v0.2-ep3 \
  --output artifacts/reports/comparison_v0.2.json
```

Expected output (test split, 350 pairs):

| Model | MAE | Label Acc |
|-------|-----|-----------|
| base | 20.32 | 32.3% |
| v0.2-ep3 | 11.92 | 47.7% |

See [similarity-model-v0.2.md](similarity-model-v0.2.md) for full benchmark details.

---

## 10. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `sentence-transformers is not installed` | Package missing | `pip install sentence-transformers` |
| `Similarity model disabled: fallback_mode=rule_only` | `SIMILARITY_FALLBACK_MODE=rule_only` | Change to `base_model` or remove the variable |
| `Fine-tuned model not found at '...'` | Wrong path or missing `config.json` | Verify the directory exists and contains `config.json` |
| `similarity_score` is always `null` | `SIMILARITY_SCORING_WEIGHT=0.0` | Set weight > 0.0 |
| Blending not happening despite weight > 0 | ML score below `SIMILARITY_SCORE_THRESHOLD` | Lower threshold or check model quality |
