# Resume Section Classifier v0.1 — Distillation from LLM Teacher

**Version**: v0.1
**Date**: July 2026
**Status**: ✅ Complete — merged to `develop`, disabled by default in production
**Scope**: CV/resume section splitting only. JD section splitting (`app/parsers/job_description_parser.py`) is explicitly out of scope for this iteration.

---

## 1. Objective

`app/parsers/section_splitter.py` used a fixed regex/alias-dictionary to detect resume
section headers (`summary`, `skills`, `experience`, `projects`, `education`,
`certifications`, `achievements`, `languages`, plus an `"other"` fallback bucket). It
generalizes poorly to resumes with non-standard or missing headers — exactly the case
where every unmatched line falls into `"other"`.

Goal: replace that blind spot with a small, fast, local model **distilled from a large
LLM teacher**, following the same "LLM generates/labels data → small model learns to
replicate it" pattern already validated on this project's CV-JD cross-encoder scorer —
but far lighter-weight, since this is a per-line classification task, not a semantic
relevance-scoring task.

---

## 2. Why this was simpler than the CV-JD cross-encoder

| | Cross-encoder (CV-JD scoring) | Section classifier (this project) |
|---|---|---|
| What's learned | Semantic relevance between two documents | Which section a single line belongs to |
| Model | Fine-tuned 12-layer transformer | Frozen sentence embeddings + `LogisticRegression` head |
| Training compute | GPU (Colab/Kaggle), hours per run | Local CPU, minutes |
| Dataset size needed | ~7,000–13,350 pairs to reach ceiling | ~300 resumes sufficient for a useful v0.1 |
| Tuning required | LR grid search, loss engineering, ensembling, multiple dataset iterations (v0.1–v0.7) | None — single training pass |

---

## 3. Architecture

**Sentence-embedding line features → lightweight classifier head → Viterbi smoothing.**

1. **Embedding**: each resume line is encoded with `sentence-transformers/all-MiniLM-L6-v2`
   (same base model as the existing similarity scorer — shares the singleton at runtime
   when compatible, see `app/ml/section_classifier_model.py`).
2. **Structural features** (concatenated to the embedding): log word count, bullet-marker
   presence, trailing colon, ALL-CAPS ratio, whether the line matches the existing regex
   splitter's own header alias table, and relative position in the document. See
   `app/ml/section_classifier_features.py`.
3. **Classifier head**: `sklearn.linear_model.LogisticRegression` over embedding + features.
4. **Viterbi smoothing**: a bigram label-transition matrix estimated from training label
   sequences, decoded with standard Viterbi so predicted labels form coherent runs instead
   of flipping line-by-line — section labels are contiguous ordered spans, not independent
   per-line decisions.

No new dependencies: `sentence-transformers` and `scikit-learn` were already project
dependencies.

---

## 4. Data generation

Existing CV-JD datasets (`datasets/raw*/resumes.jsonl`) are LLM-synthetic **prose with no
headers or bullets by design** — the opposite of what a section-splitter needs to learn.

A new generator, `scripts/generate_section_labeling_data.py`, produces resumes that
**deliberately include headers, bullets, and varied layout** (ALL CAPS / Title Case /
Markdown-bold / inline-colon / minimal-header styles, English/Vietnamese/mixed,
plus known-hard-case layouts like stacked project dates and wrapped skill lines), with
the LLM teacher labeling every line with its section in the same pass. Reuses the
existing multi-provider key-rotation client (`scripts/llm_client.py`) and the
accumulate-and-retry-only-missing pattern already proven on the CV-JD dataset generator.

**Dataset produced**: 303 resumes, 5,379 labeled lines
(`datasets/section_splitting/v0.1/labeled_lines.jsonl`), split 70/15/15 by resume id
(206 / 47 / 50 resumes) via `training/prepare_section_classifier_data.py`.

Label distribution is imbalanced in a realistic way — `experience` (35%) and `projects`
(16%) dominate; `achievements` (3.0%) and `languages` (3.2%) are naturally rare, since
real resumes typically give them only 1–2 lines each.

---

## 5. Results (test split, 50 resumes / 888 lines)

| Metric | No smoothing | With Viterbi smoothing |
|---|---|---|
| Overall accuracy | 85% | **86%** |
| Boundary accuracy (section-change detected at the right line) | — | **89.1%** |

Per-label F1 (Viterbi-smoothed): `languages` 0.90, `education` 0.93, `certifications`
0.92, `summary` 0.88, `skills` 0.89, `experience` 0.87, `projects` 0.76,
`achievements` 0.59.

**Known limitations**:
- `projects` and `experience` are the most confused pair — both read as bulleted
  work-history entries, semantically close.
- `achievements` recall *drops* under Viterbi smoothing (0.50 → 0.42): at only 3% of
  training data, the learned transition matrix assigns it a low prior, so isolated true
  `achievements` lines get smoothed into a neighboring more common section. This is a
  direct consequence of class imbalance, not a code defect — more `achievements`-heavy
  training data is the fix, not an algorithm change.
- The model essentially never predicts `"other"` (true label is only 0.3% of training
  data), so genuinely unclassifiable/decorative lines will now be force-assigned to a
  real section rather than left unclassified. Acceptable for real resume content; a
  known edge case for degenerate input.

---

## 6. Real-CV validation

Tested against the author's own CV (`DangHuuLong_FullStackDeveloper_CV.pdf`) via
`scripts/compare_splitters.py`:

- With a naive PDF-text extractor (no column-reading-order handling), the extracted text
  was scrambled by the CV's two-column layout — an extraction-stage problem, out of scope
  for this splitter.
- With the project's proper reading-order-aware extraction
  (`DocumentTextExtractionService`), the text came out clean with standard English
  headers. The regex splitter already classified every line correctly on its own (no
  `"other"` bucket) — the ML path made **zero changes**, confirming it's a safe no-op
  when the heuristic already succeeds, not just when it's disabled.

Across the full synthetic dataset (303 resumes), the ML path changed at least one line
in 116/303 resumes (38%), moving 978 lines total out of `"other"` — the model earns its
keep specifically on non-standard-header resumes, as designed.

---

## 7. Integration

`split_sections()` (`app/parsers/section_splitter.py`) keeps its exact
`dict[str, str]` signature — zero changes needed in `resume_parser.py` or any extractor.
Strategy: **regex-first, ML-fallback for ambiguous lines only** — any line the regex
pass confidently assigns to a real section (a header was matched) is kept as-is; only
lines regex leaves in `"other"` (content before the first recognized header, or an
entire no-header resume) are reclassified by the model, Viterbi-smoothed.

Controlled via `.env`:
```
SECTION_CLASSIFIER_MODEL_PATH=models/section-classifier-v0.1
SECTION_CLASSIFIER_FALLBACK_MODE=regex_only | model_with_regex_fallback
```
Defaults to `regex_only` (fully disabled / no-op) unless explicitly configured —
confirmed via the full existing parser regression suite passing unchanged with the
default config.

---

## 8. Training pipeline (local CPU, no Colab)

Unlike the cross-encoder's Colab-only fine-tuning, this trains entirely locally since
it only fits a linear head on frozen embeddings (one batched embedding pass, not
iterative transformer fine-tuning):

```
python -m training.validate_section_dataset
python -m training.prepare_section_classifier_data
python -m training.fine_tune_section_classifier
python -m training.evaluate_section_classifier
```

Or interactively via `notebooks/fine_tune_section_classifier_v0.1_local.ipynb`.

Artifacts (`classifier.joblib`, `transition_matrix.npy`, `config.json`) are saved to
`models/section-classifier-v0.1/` (gitignored, like other trained model artifacts).

---

## 9. Next steps (not started)

- Generate more `achievements`/`languages`-heavy resumes to fix the class-imbalance
  weakness noted in Section 5.
- Extend the same distillation approach to JD section splitting
  (`job_description_parser.py`) as a separate follow-up.
- Decide whether to flip the default to `model_with_regex_fallback` in production after
  broader validation.
