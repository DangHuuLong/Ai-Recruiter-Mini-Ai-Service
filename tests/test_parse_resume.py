from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_parse_resume_returns_parsed_resume_data():
    response = client.post(
        "/parse/resume",
        json={
            "raw_text": "Candidate has experience with Python, FastAPI, and PostgreSQL. Contact: test@example.com +84901234567 https://github.com/test",
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["success"] is True
    assert body["message"] == "Resume parsed successfully"
    # summary may be None for short inputs
    assert "data" in body
    assert isinstance(body["data"], dict)
    assert len(body["data"].get("skills", [])) >= 1
    # email and phone should be extracted
    assert body["data"]["personal"]["email"] == "test@example.com"
    assert body["data"]["personal"]["phone"] == "+84901234567"
    assert body["data"]["personal"]["github_url"] == "https://github.com/test"