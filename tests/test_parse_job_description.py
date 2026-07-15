from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.document_text_extraction_service import DocumentTextExtractionResult


client = TestClient(app)


@pytest.fixture(autouse=True)
def _disable_jd_ml_classifier():
    # This file verifies the deterministic (regex-based) JD parsing contract,
    # not the ML-assisted blend (see tests/test_job_description_parser_ml_assisted.py
    # for that). Force the fallback path regardless of the environment's
    # JD_SECTION_CLASSIFIER_FALLBACK_MODE so these tests don't depend on
    # whatever model happens to be configured/trained locally.
    with patch(
        "app.ml.jd_section_classifier_model.get_jd_section_classifier_model",
        side_effect=RuntimeError("disabled for deterministic-parser tests"),
    ):
        yield


def test_parse_job_description_returns_parsed_jd_data():
    response = client.post(
        "/parse/job-description",
        json={
            "raw_text": "We need a Backend Developer with Python, FastAPI, PostgreSQL, Docker, and Redis."
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["success"] is True
    assert body["message"] == "Job description parsed successfully"
    assert body["data"]["title"] == "Backend Developer"
    assert len(body["data"]["required_skills"]) >= 1


def test_parse_job_description_extracts_structured_sections_and_requirements():
    response = client.post(
        "/parse/job-description",
        json={
            "raw_text": """
            Job Title: Senior Backend Developer
            Employment Type: Full-time

            Responsibilities:
            - Build and maintain REST API services.
            - Work with product and frontend teams.

            Requirements:
            - At least 3 years of backend development experience.
            - Strong Python, FastAPI, PostgreSQL, Redis, and Docker experience.
            - Bachelor's degree in Computer Science or related field.

            Nice to have:
            - Experience with AWS and Kubernetes.
            - Knowledge of CI/CD pipelines.
            """
        },
    )

    assert response.status_code == 200
    data = response.json()["data"]

    assert data["title"] == "Senior Backend Developer"
    assert data["seniority"] == "senior"
    assert data["employment_type"] == "full-time"
    assert data["min_experience_years"] == 3
    assert data["education_requirement"] is not None
    assert "Build and maintain REST API services" in data["responsibilities"][0]
    assert any("Python" in item for item in data["requirements"])
    assert any("AWS" in item for item in data["nice_to_have"])

    required_skill_names = {skill["normalized_name"] for skill in data["required_skills"]}
    preferred_skill_names = {skill["normalized_name"] for skill in data["preferred_skills"]}

    assert {"python", "fastapi", "postgresql", "redis", "docker"}.issubset(required_skill_names)
    assert {"aws", "kubernetes", "ci_cd"}.issubset(preferred_skill_names)
    assert all(skill["is_core"] is True for skill in data["required_skills"])
    assert all(skill["is_core"] is False for skill in data["preferred_skills"])
    assert "backend" in data["domain_keywords"]
    assert "rest api" in data["domain_keywords"]


def test_parse_job_description_supports_vietnamese_headings_and_normalized_skills():
    response = client.post(
        "/parse/job-description",
        json={
            "raw_text": """
            Vị trí: Frontend Engineer
            Hình thức: toàn thời gian

            Trách nhiệm:
            - Xây dựng giao diện web với React và TypeScript.

            Yêu cầu:
            - Tối thiểu 2 năm kinh nghiệm frontend.
            - Thành thạo React, TypeScript, HTML, CSS.

            Ưu tiên:
            - Có kinh nghiệm Next.js và Tailwind CSS.
            """
        },
    )

    assert response.status_code == 200
    data = response.json()["data"]

    assert data["title"] == "Frontend Engineer"
    assert data["employment_type"] == "full-time"
    assert data["min_experience_years"] == 2

    required_skill_names = {skill["normalized_name"] for skill in data["required_skills"]}
    preferred_skill_names = {skill["normalized_name"] for skill in data["preferred_skills"]}

    assert {"react", "typescript", "html", "css"}.issubset(required_skill_names)
    assert {"nextjs", "tailwind_css"}.issubset(preferred_skill_names)
    assert "frontend" in data["domain_keywords"]


def test_parse_job_description_matches_hyphenated_skills_and_avoids_generic_data_domain():
    response = client.post(
        "/parse/job-description",
        json={
            "raw_text": """
            Job Title: Senior Backend Developer
            Employment Type: Full-time

            Responsibilities:
            - Collaborate with frontend developers to define API contracts.
            - Store structured recruiting data for candidates and evaluations.

            Requirements:
            - At least 4 years of backend development experience.
            - Strong Node.js, NestJS, TypeScript, REST API, PostgreSQL, and Redis experience.
            - Good understanding of Docker-based local development and CI/CD pipelines.
            - Experience with Prisma ORM.

            Nice to have:
            - Experience with AWS, Kubernetes, Python, FastAPI, Jest, or Pytest.
            """
        },
    )

    assert response.status_code == 200
    data = response.json()["data"]

    required_skill_names = {skill["normalized_name"] for skill in data["required_skills"]}
    preferred_skill_names = {skill["normalized_name"] for skill in data["preferred_skills"]}

    assert "docker" in required_skill_names
    assert "ci_cd" in required_skill_names
    assert "javascript" not in required_skill_names
    assert {"aws", "kubernetes", "python", "fastapi", "jest", "pytest"}.issubset(preferred_skill_names)
    assert "data" not in data["domain_keywords"]
    assert "backend" in data["domain_keywords"]
    assert "rest api" in data["domain_keywords"]


def test_parse_job_description_does_not_match_short_alias_inside_framework_names():
    response = client.post(
        "/parse/job-description",
        json={
            "raw_text": """
            Job Title: Backend Developer

            Requirements:
            - Strong Node.js and NestJS experience.
            - Experience with TypeScript.
            """
        },
    )

    assert response.status_code == 200
    data = response.json()["data"]
    required_skill_names = {skill["normalized_name"] for skill in data["required_skills"]}

    assert "nodejs" in required_skill_names
    assert "nestjs" in required_skill_names
    assert "typescript" in required_skill_names
    assert "javascript" not in required_skill_names


def test_parse_job_description_accepts_file_via_signed_url():
    mock_result = DocumentTextExtractionResult(
        raw_text="We need a Backend Developer with Python, FastAPI, and PostgreSQL.",
        method="PDF_TEXT_PYMUPDF_WORDS",
    )

    with patch(
        "app.services.parsing_service.document_text_extraction_service.extract_from_signed_url",
        return_value=mock_result,
    ) as mock_extract:
        response = client.post(
            "/parse/job-description",
            json={
                "signed_url": "https://example.com/jd.pdf",
                "file_name": "jd.pdf",
                "file_type": "PDF",
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["data"]["title"] == "Backend Developer"
    mock_extract.assert_called_once_with(
        signed_url="https://example.com/jd.pdf",
        file_type="PDF",
        file_name="jd.pdf",
    )


def test_parse_job_description_requires_raw_text_or_signed_url():
    response = client.post("/parse/job-description", json={})

    assert response.status_code == 422


def test_parse_job_description_requires_file_name_and_type_with_signed_url():
    response = client.post(
        "/parse/job-description",
        json={"signed_url": "https://example.com/jd.pdf"},
    )

    assert response.status_code == 422
