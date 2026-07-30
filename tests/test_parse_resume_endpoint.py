from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_parse_resume_accepts_raw_text_directly():
    response = client.post(
        "/parse/resume",
        json={
            "resume_id": "resume-1",
            "raw_text": (
                "Summary:\nBackend developer with practical API experience.\n\n"
                "Skills:\nPython, FastAPI, PostgreSQL\n"
            ),
        },
    )

    assert response.status_code == 200
    body = response.json()

    assert body["success"] is True
    assert body["data"]["text_extraction_method"] == "RAW_TEXT_DIRECT"
    assert "Backend developer" in body["data"]["raw_text"]
    normalized_skills = {s["normalized_name"] for s in body["data"]["parsed_data"]["skills"]}
    assert {"python", "fastapi", "postgresql"}.issubset(normalized_skills)


def test_parse_resume_requires_signed_url_or_raw_text():
    response = client.post("/parse/resume", json={"resume_id": "resume-1"})

    assert response.status_code == 422


def test_parse_resume_requires_file_name_and_type_with_signed_url():
    response = client.post(
        "/parse/resume",
        json={
            "resume_id": "resume-1",
            "signed_url": "https://example.com/resume.pdf",
        },
    )

    assert response.status_code == 422
