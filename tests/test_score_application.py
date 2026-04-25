from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_score_application_returns_evaluation_result():
    response = client.post(
        "/score/application",
        json={
            "resume": {
                "personal": {
                    "full_name": "Mock Candidate",
                    "email": None,
                    "phone": None,
                    "location": None,
                    "linkedin_url": None,
                    "github_url": None,
                    "portfolio_url": None,
                },
                "summary": "Mock parsed resume profile.",
                "skills": [
                    {
                        "name": "Python",
                        "normalized_name": "python",
                        "category": "backend",
                        "evidence": "Python mentioned in resume text",
                    }
                ],
                "education": [],
                "experience": [],
                "projects": [],
                "certifications": [],
                "achievements": [],
                "languages": [],
            },
            "job_description": {
                "title": "Backend Developer",
                "seniority": "junior",
                "employment_type": "full-time",
                "responsibilities": [],
                "requirements": [],
                "nice_to_have": [],
                "required_skills": [
                    {
                        "name": "Python",
                        "normalized_name": "python",
                        "is_core": True,
                        "weight_hint": 1.0,
                    },
                    {
                        "name": "PostgreSQL",
                        "normalized_name": "postgresql",
                        "is_core": True,
                        "weight_hint": 1.0,
                    },
                ],
                "preferred_skills": [],
                "min_experience_years": 1,
                "education_requirement": None,
                "domain_keywords": [],
            },
            "config": {
                "criteria": [
                    {
                        "criterion": "SKILLS_MATCH",
                        "weight": 0.35,
                    }
                ]
            },
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["success"] is True
    assert body["message"] == "Application scored successfully"
    assert body["data"]["overall_score"] == 17.5
    assert len(body["data"]["skills"]) == 2