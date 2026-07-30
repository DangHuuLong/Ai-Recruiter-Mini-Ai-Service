from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from training.evaluate_similarity_pipeline import (
    EvalSample,
    Metrics,
    ModelResult,
    _extract_jd_text,
    _extract_resume_text,
    compute_metrics,
    label_for_score,
    load_test_pairs,
    run_evaluation,
    run_pipeline,
)


# ── label_for_score ───────────────────────────────────────────────────────────

class TestLabelForScore:
    def test_poor_match(self) -> None:
        assert label_for_score(0.0) == "poor_match"
        assert label_for_score(39.9) == "poor_match"

    def test_weak_match(self) -> None:
        assert label_for_score(40.0) == "weak_match"
        assert label_for_score(59.9) == "weak_match"

    def test_moderate_match(self) -> None:
        assert label_for_score(60.0) == "moderate_match"
        assert label_for_score(74.9) == "moderate_match"

    def test_strong_match(self) -> None:
        assert label_for_score(75.0) == "strong_match"
        assert label_for_score(89.9) == "strong_match"

    def test_excellent_match(self) -> None:
        assert label_for_score(90.0) == "excellent_match"
        assert label_for_score(100.0) == "excellent_match"


# ── compute_metrics ───────────────────────────────────────────────────────────

class TestComputeMetrics:
    def test_empty_returns_zeros(self) -> None:
        m = compute_metrics([], [])
        assert m.mae == 0.0
        assert m.rmse == 0.0
        assert m.label_accuracy == 0.0
        assert m.n_samples == 0

    def test_perfect_predictions(self) -> None:
        m = compute_metrics([80.0, 60.0, 40.0], [80.0, 60.0, 40.0])
        assert m.mae == 0.0
        assert m.rmse == 0.0
        assert m.label_accuracy == 1.0

    def test_known_mae_rmse(self) -> None:
        # errors: [10, 20] → MAE=15, RMSE=sqrt((100+400)/2)=sqrt(250)≈15.8114
        m = compute_metrics([90.0, 80.0], [80.0, 60.0])
        assert m.mae == 15.0
        assert abs(m.rmse - 15.8114) < 0.001

    def test_label_accuracy_partial(self) -> None:
        # pred=85 → strong, true=65 → moderate: wrong
        # pred=65 → moderate, true=62 → moderate: correct
        m = compute_metrics([85.0, 65.0], [65.0, 62.0])
        assert m.label_accuracy == 0.5


# ── text extraction ───────────────────────────────────────────────────────────

class TestExtractText:
    def test_resume_extracts_skills_and_summary(self) -> None:
        r = {
            "summary": "Python developer",
            "skills": [{"name": "Python"}, "FastAPI"],
            "experience": [],
            "projects": [],
            "education": [],
        }
        text = _extract_resume_text(r)
        assert "Python developer" in text
        assert "Python" in text
        assert "FastAPI" in text

    def test_jd_extracts_title_and_requirements(self) -> None:
        jd = {
            "title": "Backend Engineer",
            "level": "Senior",
            "requirements": ["5 years Python"],
            "responsibilities": [],
            "nice_to_have": [],
            "domain_keywords": [],
        }
        text = _extract_jd_text(jd)
        assert "Backend Engineer" in text
        assert "Senior" in text
        assert "5 years Python" in text

    def test_empty_dicts_return_empty_string(self) -> None:
        assert _extract_resume_text({}) == ""
        assert _extract_jd_text({}) == ""


# ── load_test_pairs ───────────────────────────────────────────────────────────

def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(r) for r in records),
        encoding="utf-8",
    )


class TestLoadTestPairs:
    def test_loads_only_test_split(self, tmp_path: Path) -> None:
        resumes_path = tmp_path / "resumes.jsonl"
        jds_path = tmp_path / "jds.jsonl"
        pairs_path = tmp_path / "pairs.jsonl"

        _write_jsonl(resumes_path, [{"id": "r1", "skills": [], "experience": [], "projects": [], "education": []}])
        _write_jsonl(jds_path, [{"id": "j1", "title": "Dev", "responsibilities": [], "requirements": [], "nice_to_have": [], "domain_keywords": []}])
        _write_jsonl(pairs_path, [
            {"id": "p1", "resume_id": "r1", "job_description_id": "j1", "split": "train", "overall_score": 70.0, "label": "strong_match"},
            {"id": "p2", "resume_id": "r1", "job_description_id": "j1", "split": "test",  "overall_score": 80.0, "label": "strong_match"},
            {"id": "p3", "resume_id": "r1", "job_description_id": "j1", "split": "validation", "overall_score": 60.0, "label": "moderate_match"},
        ])

        samples = load_test_pairs(pairs_path, resumes_path, jds_path)

        assert len(samples) == 1
        assert samples[0].pair_id == "p2"
        assert samples[0].true_score == 80.0

    def test_skips_pairs_with_missing_resume(self, tmp_path: Path) -> None:
        resumes_path = tmp_path / "resumes.jsonl"
        jds_path = tmp_path / "jds.jsonl"
        pairs_path = tmp_path / "pairs.jsonl"

        _write_jsonl(resumes_path, [])
        _write_jsonl(jds_path, [{"id": "j1", "title": "Dev", "responsibilities": [], "requirements": [], "nice_to_have": [], "domain_keywords": []}])
        _write_jsonl(pairs_path, [
            {"id": "p1", "resume_id": "r_missing", "job_description_id": "j1", "split": "test", "overall_score": 70.0}
        ])

        samples = load_test_pairs(pairs_path, resumes_path, jds_path)
        assert samples == []

    def test_uses_label_from_pair_if_present(self, tmp_path: Path) -> None:
        resumes_path = tmp_path / "resumes.jsonl"
        jds_path = tmp_path / "jds.jsonl"
        pairs_path = tmp_path / "pairs.jsonl"

        _write_jsonl(resumes_path, [{"id": "r1", "skills": [], "experience": [], "projects": [], "education": []}])
        _write_jsonl(jds_path, [{"id": "j1", "title": "Dev", "responsibilities": [], "requirements": [], "nice_to_have": [], "domain_keywords": []}])
        _write_jsonl(pairs_path, [
            {"id": "p1", "resume_id": "r1", "job_description_id": "j1", "split": "test", "overall_score": 80.0, "label": "strong_match"},
        ])

        samples = load_test_pairs(pairs_path, resumes_path, jds_path)
        assert samples[0].true_label == "strong_match"


# ── run_evaluation ────────────────────────────────────────────────────────────

class TestRunEvaluation:
    _PATCH = "training.evaluate_similarity_pipeline.SentenceTransformer"

    def test_returns_correct_predictions(self) -> None:
        samples = [
            EvalSample("p1", 80.0, "strong_match", "cv text", "jd text"),
            EvalSample("p2", 60.0, "moderate_match", "cv2", "jd2"),
        ]
        mock_model = MagicMock()
        # Encode returns [cv1, cv2, jd1, jd2]
        # cv1 · jd1 = 1.0 (same direction) → 100%
        # cv2 · jd2 = 0.5 → 50%
        mock_model.encode.return_value = [
            [1.0, 0.0],   # cv1
            [1.0, 0.0],   # cv2
            [1.0, 0.0],   # jd1 → sim with cv1 = 1.0
            [0.5, 0.866], # jd2 → sim with cv2 ≈ 0.5
        ]

        with patch(self._PATCH, return_value=mock_model):
            result = run_evaluation("test-model", "test-model", samples)

        assert result.name == "test-model"
        assert len(result.predictions) == 2
        assert result.predictions[0]["predicted_score"] == 100.0
        assert result.predictions[0]["predicted_label"] == "excellent_match"

    def test_raises_without_sentence_transformers(self) -> None:
        samples = [EvalSample("p1", 80.0, "strong_match", "cv", "jd")]
        with patch(self._PATCH, side_effect=ImportError("no module")):
            with pytest.raises(RuntimeError, match="sentence-transformers"):
                # Patch at import level too
                pass  # ImportError is caught at import time in the module


# ── run_pipeline ──────────────────────────────────────────────────────────────

class TestRunPipeline:
    def test_report_structure(self, tmp_path: Path) -> None:
        resumes_path = tmp_path / "resumes.jsonl"
        jds_path = tmp_path / "jds.jsonl"
        pairs_path = tmp_path / "pairs.jsonl"

        _write_jsonl(resumes_path, [{"id": "r1", "skills": [], "experience": [], "projects": [], "education": []}])
        _write_jsonl(jds_path, [{"id": "j1", "title": "Dev", "responsibilities": [], "requirements": [], "nice_to_have": [], "domain_keywords": []}])
        _write_jsonl(pairs_path, [
            {"id": "p1", "resume_id": "r1", "job_description_id": "j1", "split": "test", "overall_score": 80.0, "label": "strong_match"},
        ])

        fake_result = ModelResult(
            name="base",
            model_path="base",
            metrics=Metrics(mae=17.64, rmse=22.45, label_accuracy=0.28, n_samples=1),
            predictions=[],
        )

        with patch("training.evaluate_similarity_pipeline.run_evaluation", return_value=fake_result):
            report = run_pipeline(
                model_configs=[{"name": "base", "path": "base"}],
                pairs_path=pairs_path,
                resumes_path=resumes_path,
                jds_path=jds_path,
            )

        assert "generated_at" in report
        assert report["test_set_size"] == 1
        assert len(report["models"]) == 1
        assert report["models"][0]["name"] == "base"
        assert report["models"][0]["metrics"]["mae"] == 17.64
