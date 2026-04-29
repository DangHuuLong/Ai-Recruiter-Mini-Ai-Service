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


def test_parse_resume_extracts_structured_sections():
    response = client.post(
        "/parse/resume",
        json={
            "raw_text": """
John Doe
Email: john.doe@example.com
Phone: +84987654321
LinkedIn: linkedin.com/in/johndoe
Portfolio: johndoe.dev

Summary:
Backend developer with practical API and database experience.

Skills:
Python, FastAPI, PostgreSQL, Docker, Redis, REST API

Experience:
Backend Developer at ABC Tech | Jan 2022 - Mar 2024
- Built REST APIs with FastAPI and PostgreSQL
- Integrated Redis caching and Docker-based deployment

Projects:
AI Recruiter - CV screening API using Python, FastAPI and PostgreSQL https://github.com/test/ai-recruiter
- Used Docker and Redis for local development

Education:
University of Technology - Bachelor of Computer Science, 2019 - 2023

Certifications:
AWS Certified Cloud Practitioner - Amazon Web Services, 2024

Achievements:
Improved API response time by 40% in 2024

Languages:
English - Intermediate, Vietnamese - Native
""",
        },
    )

    assert response.status_code == 200
    data = response.json()["data"]

    assert data["personal"]["full_name"] == "John Doe"
    assert data["personal"]["linkedin_url"] == "https://linkedin.com/in/johndoe"
    assert data["personal"]["portfolio_url"] == "https://johndoe.dev"
    assert data["summary"] == "Backend developer with practical API and database experience."

    normalized_skills = {skill["normalized_name"] for skill in data["skills"]}
    assert {"python", "fastapi", "postgresql", "docker", "redis", "rest_api"}.issubset(normalized_skills)

    assert data["experience"][0]["role"] == "Backend Developer"
    assert data["experience"][0]["company"] == "ABC Tech"
    assert data["experience"][0]["start_date"] == "2022-01"
    assert data["experience"][0]["end_date"] == "2024-03"
    assert data["experience"][0]["duration_months"] == 26
    assert "PostgreSQL" in data["experience"][0]["technologies"]

    assert data["projects"][0]["name"] == "AI Recruiter"
    assert data["projects"][0]["url"] == "https://github.com/test/ai-recruiter"
    assert "Docker" in data["projects"][0]["technologies"]

    assert data["education"][0]["institution"] == "University of Technology"
    assert data["education"][0]["degree"] == "Bachelor"
    assert data["education"][0]["field_of_study"] == "Computer Science"
    assert data["education"][0]["start_year"] == 2019
    assert data["education"][0]["end_year"] == 2023

    assert data["certifications"][0]["issuer"] == "Amazon Web Services"
    assert data["certifications"][0]["issued_year"] == 2024
    assert data["achievements"][0]["year"] == 2024
    assert data["languages"] == [
        {"name": "English", "proficiency": "intermediate"},
        {"name": "Vietnamese", "proficiency": "native"},
    ]


def test_parse_resume_handles_resume_without_clear_section_headers():
    response = client.post(
        "/parse/resume",
        json={
            "raw_text": """
Nguyen Van A
nguyen@example.com | +84 912 345 678 | github.com/nguyenvana
Backend Engineer at Beta Labs | 06/2021 - Present
Built Node.js REST APIs with PostgreSQL and Docker.
University of Science - Bachelor of Information Technology, 2017 - 2021
AWS Certified Developer - Amazon Web Services, 2022
English - Advanced
""",
        },
    )

    assert response.status_code == 200
    data = response.json()["data"]

    assert data["personal"]["full_name"] == "Nguyen Van A"
    assert data["personal"]["github_url"] == "https://github.com/nguyenvana"
    assert data["experience"][0]["role"] == "Backend Engineer"
    assert data["experience"][0]["company"] == "Beta Labs"
    assert data["experience"][0]["start_date"] == "2021-06"
    assert data["experience"][0]["end_date"] == "present"
    assert data["education"][0]["field_of_study"] == "Information Technology"
    assert data["certifications"][0]["issued_year"] == 2022
    assert data["languages"][0]["name"] == "English"


def test_parse_resume_handles_vietnamese_section_headers():
    response = client.post(
        "/parse/resume",
        json={
            "raw_text": """
Tran Thi B
Email: tranb@example.com

Kỹ năng: Python, Django, PostgreSQL

Kinh nghiệm làm việc:
Backend Developer - Cong ty ABC - 2020 - 2022
- Xay dung REST API bang Django va PostgreSQL

Học vấn:
Truong Dai hoc Bach Khoa - Cu nhan Cong nghe thong tin, 2016 - 2020

Ngôn ngữ:
Tieng Anh - Advanced
""",
        },
    )

    assert response.status_code == 200
    data = response.json()["data"]

    assert data["personal"]["full_name"] == "Tran Thi B"
    normalized_skills = {skill["normalized_name"] for skill in data["skills"]}
    assert {"python", "django", "postgresql"}.issubset(normalized_skills)
    assert data["experience"][0]["company"] == "Cong ty ABC"
    assert data["education"][0]["degree"] == "Bachelor"
    assert data["education"][0]["field_of_study"] == "Information Technology"
    assert data["languages"][0]["proficiency"] == "advanced"
