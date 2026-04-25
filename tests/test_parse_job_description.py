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