from app.parsers.resume_parser import parse_resume


def _skill_names(result):
    return {skill.name for skill in result.skills}


def test_parse_stacked_projects_with_trailing_right_column_dates():
    raw_text = """
Nguyễn Thanh Hiếu
nguyenthanhhieu17022005@gmail.com | 0587381816 | www.linkedin.com/in/thanhhieu1702

OBJECTIVE
I am seeking an internship where I can apply and strengthen my skills in Node.js, NestJS, and React.
I have experience with Tailwind CSS, Bootstrap, shadcn/ui, and basic deployment using AWS,
Docker, and GitHub Actions.

EDUCATION
University of Science and Technology –
Nov 2023 - Jul 2027
The University of Danang
Information Technology – Bachelor Program
Cumulative GPA: 3.96/4.0

WORK EXPERIENCE
S-Group
Dec 2024 - Present
Backend Developer (Trainee)
Built frontend foundations (HTML, CSS) and developed modern UIs using React, Tailwind CSS,
shadcn/ui
Worked on backend with Node.js, Express.js using JavaScript/TypeScript
Hands-on experience with Docker, AWS EC2, Nginx, and Socket.IO for deployment and realtime
features

PROJECTS
Trello Clone
FullStack Developer
Designed database with RBAC.
Built frontend using React, Tailwind CSS, shadcn/ui, Zustand.
Developed REST APIs and authorization with Node.js, Express.js, TypeScript, PostgreSQL.
DUT Meeting
FullStack Developer
Built UI with React, Bootstrap, shadcn/ui.
Integrated Whisper AI for real-time subtitles via background workers.
Developed REST APIs and authorization with Node.js, Express.js.
2025 - 2026
2025 - 2025

CERTIFICATIONS
2023: TOEIC 925/990
"""

    result = parse_resume(raw_text)

    assert result.personal.email == "nguyenthanhhieu17022005@gmail.com"
    assert result.personal.phone == "0587381816"

    assert len(result.education) == 1
    assert "University of Science and Technology" in result.education[0].institution
    assert "The University of Danang" in result.education[0].institution
    assert result.education[0].field_of_study == "Information Technology"
    assert result.education[0].degree == "Bachelor"
    assert result.education[0].start_year == 2023
    assert result.education[0].end_year == 2027

    assert len(result.experience) == 1
    assert result.experience[0].company == "S-Group"
    assert result.experience[0].role == "Backend Developer (Trainee)"
    assert any("Tailwind CSS, shadcn/ui" in item for item in result.experience[0].responsibilities)
    assert any("realtime features" in item for item in result.experience[0].responsibilities)

    assert len(result.projects) == 2
    assert result.projects[0].name == "Trello Clone"
    assert result.projects[0].role == "FullStack Developer"
    assert result.projects[0].start_date == "2025"
    assert result.projects[0].end_date == "2026"
    assert result.projects[1].name == "DUT Meeting"
    assert result.projects[1].role == "FullStack Developer"
    assert result.projects[1].start_date == "2025"
    assert result.projects[1].end_date == "2025"

    skills = _skill_names(result)
    assert "GitHub Actions" in skills
    assert "shadcn/ui" in skills
    assert "Socket.IO" in skills
    assert "Zustand" in skills
    assert "Whisper AI" in skills
    assert "RBAC" in skills


def test_parse_single_column_projects_with_inline_dates():
    raw_text = """
PROJECTS
Trello Clone
FullStack Developer
2025 - 2026
Designed database with RBAC.
Built frontend using React, Tailwind CSS, shadcn/ui, Zustand.

DUT Meeting
FullStack Developer
2025 - 2025
Built UI with React, Bootstrap, shadcn/ui.
Integrated Whisper AI for real-time subtitles via background workers.
"""

    result = parse_resume(raw_text)

    assert len(result.projects) == 2
    assert result.projects[0].name == "Trello Clone"
    assert result.projects[0].role == "FullStack Developer"
    assert result.projects[0].start_date == "2025"
    assert result.projects[0].end_date == "2026"
    assert result.projects[1].name == "DUT Meeting"
    assert result.projects[1].role == "FullStack Developer"
    assert result.projects[1].start_date == "2025"
    assert result.projects[1].end_date == "2025"


def test_parse_two_column_like_project_dates_assigned_by_order():
    raw_text = """
PROJECTS
Inventory System
Backend Developer
Built APIs with NestJS and PostgreSQL.
Chat App
FullStack Developer
Built realtime chat with Socket.IO.
2024 - 2025
2025 - Present
"""

    result = parse_resume(raw_text)

    assert len(result.projects) == 2
    assert result.projects[0].name == "Inventory System"
    assert result.projects[0].start_date == "2024"
    assert result.projects[0].end_date == "2025"
    assert result.projects[1].name == "Chat App"
    assert result.projects[1].start_date == "2025"
    assert result.projects[1].end_date == "present"
