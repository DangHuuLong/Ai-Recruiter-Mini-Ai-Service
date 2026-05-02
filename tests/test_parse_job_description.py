from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


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
