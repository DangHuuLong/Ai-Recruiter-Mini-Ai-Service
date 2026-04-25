from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_parse_resume_returns_parsed_resume_data():
    response = client.post(
        "/parse/resume",
        json={
            "raw_text": "Candidate has experience with Python, FastAPI, and PostgreSQL."
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["success"] is True
    assert body["message"] == "Resume parsed successfully"
    assert body["data"]["summary"] == "Mock parsed resume profile."
    assert len(body["data"]["skills"]) >= 1