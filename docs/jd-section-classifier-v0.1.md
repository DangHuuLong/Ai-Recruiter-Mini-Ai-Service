# JD Section Classifier v0.1 — Distillation from LLM Teacher

**Version**: v0.1
**Date**: July 2026
**Status**: ✅ Complete — merged to `develop`, disabled by default in production
**Scope**: Job description section splitting (`app/parsers/job_description_parser.py`).
Mirrors [`docs/section-classifier-v0.1.md`](section-classifier-v0.1.md) (the CV version)
architecturally, adapted for JD's smaller label vocabulary.

---

## 1. Objective

`app/parsers/job_description_parser.py`'s `_split_sections` used a fixed alias
dictionary (`SECTION_ALIASES`: `responsibilities`, `requirements`, `nice_to_have`,
`benefits`) to detect JD section headers. Two problems:

1. Same generalization limit as the CV splitter — non-standard/missing headers
   aren't recognized.
2. **A real bug**: lines before the first recognized header (e.g. the job title,
   an intro/team-description paragraph) were silently **dropped entirely**, not
   just misclassified. This is a harsher failure mode than the CV splitter's
   `"other"` bucket, which at least preserves the content.

This became a real concern once `/parse/job-description` gained file-upload
support (raw PDF/DOCX JDs are far more likely to have non-standard formatting
than the form-typed text this parser was originally built for).

---

## 2. Fix + architecture

**Bug fix**: `_split_sections` now keeps pre-header content in a new `"other"`
key instead of dropping it — purely additive, doesn't change any existing
caller's behavior for the 4 known keys.

**Distillation model**: identical recipe to the CV section-classifier —
sentence-embedding line features → `LogisticRegression` head → Viterbi
smoothing over a bigram transition matrix. See
`app/ml/jd_section_classifier_features.py` (a parallel module to the CV
version rather than a generalization of it, since the CV path is already
live in production and duplicating ~100 lines of generic math is lower-risk
than touching it).

Label vocabulary (5, vs CV's 9): `responsibilities`, `requirements`,
`nice_to_have`, `benefits`, `other`.

---

## 3. Data generation

`scripts/generate_jd_labeling_data.py` — same reasoning as the CV generator:
existing JD content in `datasets/raw*` is prose-only by design (for CV-JD
scoring), so a new corpus with real headers/bullets/varied layout was needed.
Reuses the same LLM-teacher + accumulate-retry-only-missing infrastructure.

**Dataset produced**: 342 JDs, 4,738 labeled lines
(`datasets/jd_section_splitting/v0.1/labeled_lines.jsonl`), split 70/15/15
(233 / 47 / 62 JDs).

Label distribution: `requirements` 32.4%, `responsibilities` 30.4%,
`benefits` 14.8%, `other` 12.7%, `nice_to_have` 9.7% — reasonably balanced,
`nice_to_have` naturally rarest (often omitted entirely in real postings).

---

## 4. Results (test split, 62 JDs / 885 lines)

| Metric | No smoothing | With Viterbi smoothing |
|---|---|---|
| Overall accuracy | 90% | 89% |
| Boundary accuracy | — | **87.5%** |

Per-label F1 (Viterbi-smoothed): `benefits` 0.95, `nice_to_have` 0.88,
`other` 0.90, `responsibilities` 0.89, `requirements` 0.88.

**Notable, opposite of the CV result**: Viterbi smoothing *improved*
`nice_to_have` (the rarest class) — F1 0.78 → 0.88, recall 0.74 → 0.85 —
whereas the analogous rare class in the CV model (`achievements`) got
*worse* under smoothing. Likely explanation: the `requirements` →
`nice_to_have` → `benefits` progression is a strong, consistent structural
pattern in real JD postings, giving the learned transition matrix a
reliable prior to lean on; the CV equivalent (`achievements` appearing
anywhere) has no comparably strong positional pattern.

Overall accuracy dropped slightly under smoothing (90% → 89%) — the usual
per-line-vs-boundary-coherence tradeoff, same as the CV model.

---

## 5. Real-corpus validation

`scripts/compare_jd_splitters.py` run over the full 342-JD dataset:

- **179/342 JDs (52%) changed** at least one line's section.
- **915/1,368 "other" lines (67%) reclassified** into a real section.
- Spot-checked reclassifications look correct across both English and
  Vietnamese JDs, and across the "minimal-to-no-header" style specifically
  (e.g. a 14-line no-header supply-chain JD correctly split into
  responsibilities/requirements/nice_to_have).

This is a substantially higher change rate than the CV model's 38% —
consistent with the bug in Section 2 (previously-dropped content had no
chance to be "already correct" the way CV's `"other"` bucket sometimes was).

---

## 6. Integration

`_split_sections()` keeps its exact `dict[str, list[str]]` return contract —
no changes needed in `parse_job_description()` or any extractor. Same
regex-first, ML-fallback-for-ambiguous-lines strategy as the CV splitter.

```env
JD_SECTION_CLASSIFIER_MODEL_PATH=models/jd-section-classifier-v0.1
JD_SECTION_CLASSIFIER_FALLBACK_MODE=regex_only | model_with_regex_fallback
```

Defaults to `regex_only` (no-op) unless explicitly configured — confirmed via
the full existing test suite (138 tests) passing unchanged with defaults.

---

## 7. Training pipeline (local CPU, no Colab)

Same reasoning as the CV model — only a classifier head on frozen embeddings.

```
python -m training.validate_jd_section_dataset
python -m training.prepare_jd_section_classifier_data
python -m training.fine_tune_jd_section_classifier
python -m training.evaluate_jd_section_classifier
```

Artifacts (`classifier.joblib`, `transition_matrix.npy`, `config.json`) saved
to `models/jd-section-classifier-v0.1/` (gitignored, like all trained model
artifacts).

---

## 8. Known edge case

`ParsingService._sanitize_raw_text` (shared with the resume path) has a
bullet-normalization regex originally tuned for PDF-extraction artifacts
(e.g. `"•Created business logic"` → `"- Created business logic"`). It also
fires on plain indented lines with no real bullet at all (any line starting
with whitespace + a letter), prepending a `"- "` marker. The old regex-only
splitter never used bullet-presence as a signal, so this cosmetic rewrite
was harmless; the new classifier does use it as a structural feature, so
heavily-indented metadata-style lines (e.g. `"Job Title: X"` re-written as
`"- Job Title: X"`) can occasionally get misclassified as a bullet-like
responsibility/requirement instead of `"other"`. Narrow in practice (real
API callers rarely send JD text with deep leading indentation), but worth
knowing if a future bug report mentions a title/metadata line leaking into
`responsibilities`/`requirements`.

## 9. Next steps (not started)

- Grow the dataset past 342 JDs if `nice_to_have`/`other` precision needs
  further improvement.
- Decide whether to flip the default to `model_with_regex_fallback` in
  production after broader validation on real uploaded JD files.
