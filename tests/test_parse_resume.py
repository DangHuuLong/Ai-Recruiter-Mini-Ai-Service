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


def test_parse_resume_removes_null_bytes_from_raw_text():
    response = client.post(
        "/parse/resume",
        json={
            "raw_text": "Nguyen\u0000 Van A\nSkills:\nPython\u0000, FastAPI, PostgreSQL",
        },
    )

    assert response.status_code == 200

    data = response.json()["data"]
    normalized_skills = {skill["normalized_name"] for skill in data["skills"]}

    assert "python" in normalized_skills
    assert "fastapi" in normalized_skills
    assert "postgresql" in normalized_skills


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
    assert data["personal"]["portfolio_url"] == "https://johndoe.dev/"
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


def test_parse_resume_handles_vietnamese_projects_and_education_blocks():
    response = client.post(
        "/parse/resume",
        json={
            "raw_text": """
Đặng Hữu Long
Developer
danghuulong394@gmail.com
Sđt: 0942301096
Quê quán: P. Hòa Xuân, TP. Đà Nẵng

Học tập
2023 - nay Đại học Bách Khoa – Đại học Đà Nẵng
Sinh viên năm 3 – Ngành Công nghệ Thông tin (Hợp
tác doanh nghiệp)
GPA: 3.53/4.0

Dự án
Quản lý cửa hàng thực phẩm
Link Github | Dự án nhóm: 3 thành viên | 3/2025 - 6/2025
Công nghệ sử dụng: HTML, CSS, JS, Bootstrap, .NET
Mô tả chức năng:
CRUD sản phẩm, danh mục, khuyến mãi
Quản lý tài khoản
hủy có lý do)
Upload & quản lý ảnh sản phẩm
Thanh toán VNPay
Vai trò: Phụ trách phần giao diện View và Controller

Quản lý nội thất
Link Github | Dự án cá nhân | 10/2025 - nay
Công nghệ sử dụng: ReactJS, Node.js
Mô tả chức năng:
Xây dựng giao diện web giới thiệu nội thất và hệ thống quản trị đơn giản.
Customer: trang chủ, cửa hàng, giỏ hàng, thanh toán
Admin: quản lý khách hàng, sản phẩm, danh mục
Auth: đăng nhập, đăng ký, quên mật khẩu
Trạng thái: Đang phát triển
""",
        },
    )

    assert response.status_code == 200
    data = response.json()["data"]

    assert data["personal"]["location"] == "P. Hòa Xuân, TP. Đà Nẵng"

    assert len(data["projects"]) == 2
    assert data["projects"][0]["name"] == "Quản lý cửa hàng thực phẩm"
    assert data["projects"][1]["name"] == "Quản lý nội thất"

    first_project_description = data["projects"][0]["description"]
    assert "Upload & quản lý ảnh sản phẩm" in first_project_description
    assert "Thanh toán VNPay" in first_project_description
    assert "hủy có lý do" in first_project_description

    second_project_description = data["projects"][1]["description"]
    assert "Customer" in second_project_description
    assert "Admin" in second_project_description
    assert "Auth" in second_project_description
    assert "Trạng thái" in second_project_description

    assert data["education"][0]["institution"] == "Đại học Bách Khoa – Đại học Đà Nẵng"
    assert data["education"][0]["degree"] == "Bachelor"
    assert data["education"][0]["field_of_study"] == "Information Technology"
    assert data["education"][0]["start_year"] == 2023
    assert data["education"][0]["end_year"] is None
    assert "GPA: 3.53/4.0" in data["education"][0]["description"]

    assert data["experience"] == []


def test_parse_resume_normalizes_personal_links_without_stealing_project_urls():
    response = client.post(
        "/parse/resume",
        json={
            "raw_text": """
Ninh Hoang Khai
FRESHER FRONTEND DEVELOPER
nhkkhaii@gmail.com 0945772109 Ho Chi Minh City, Viet Nam nhkkhaii.super.site
https://nhkkhaii.super.site/

PROJECTS
Personal Portfolio
https://nhkkhaii.github.io/portfolio/
Frontend Developer
Technologies: ReactJS, Bootstrap
09/2022
Description:
Worked on a personal portfolio website to showcase my technical skills and some completed projects.

KaiSneaker – E-commerce Website
https://github.com/ThueCode/KaiSneaker
Full Stack Developer
Technologies: ReactJS (TypeScript), PostgreSQL, RESTful API , Java (Spring Boot)
04/2022 – 06/2022
Description:
A modern e-commerce web application for selling sneakers with complete frontend/backend integration.

CinemaNHK – Movie Ticket Booking System
https://github.com/nhkkhaii/CinemaNHK
Full Stack Developer
Technologies: C#, SQL Server, DevExpress, Microsoft Visual Studio
09/2021 – 12/2021
Description:
An desktop application for the booking and management of cinema tickets.

Project Management – Construction Management System
https://github.com/nhkkhaii/QLDA
Technologies: C#, SQL Server, Microsoft Visual Studio
03/2021 – 05/2021
Description:
Desktop construction project management application.
""",
        },
    )

    assert response.status_code == 200
    data = response.json()["data"]

    assert data["personal"]["github_url"] == "https://github.com/nhkkhaii"
    assert data["personal"]["portfolio_url"] == "https://nhkkhaii.super.site/"

    projects_by_name = {project["name"]: project for project in data["projects"]}
    assert set(projects_by_name) == {
        "Personal Portfolio",
        "KaiSneaker – E-commerce Website",
        "CinemaNHK – Movie Ticket Booking System",
        "Project Management – Construction Management System",
    }
    assert projects_by_name["Personal Portfolio"]["url"] == "https://nhkkhaii.github.io/portfolio/"
    assert projects_by_name["KaiSneaker – E-commerce Website"]["url"] == "https://github.com/ThueCode/KaiSneaker"
    assert projects_by_name["CinemaNHK – Movie Ticket Booking System"]["url"] == "https://github.com/nhkkhaii/CinemaNHK"
    assert projects_by_name["Project Management – Construction Management System"]["url"] == "https://github.com/nhkkhaii/QLDA"

    for project in data["projects"]:
        assert not (project["description"] or "").startswith("https://")


def test_parse_resume_handles_frontend_developer_pdf_text_layout():
    response = client.post(
        "/parse/resume",
        json={
            "raw_text": """
Nguyễn Quốc Bình
Frontend Developer
Objective
Become a frontend expert in 2 years.
Full-stack developer can do frontend, backend and server to participate in large
projects
Career History
Projects
SaiGon Web Company
Frontend Developer
2020/10 - Present
Working as a frontend developer responsible for frontend like coding html/css,
animation and resolved the UI issues.
<Key Achievements>
- Highly appreciated by customers for handling frontend animation and mobile
experience.
- Award for best staff of the year 2019
XTech Corporation
Junior Web Developer
2018/10 - 2020/09
- Develop new features both frontend and backend on E-commerce site using
PHP/Laravel, ReactJS
<Key Achievements>
- Deep understanding of Vuejs framework
Fanclub system 2019/02 - 2021/08
Position: Frontend Developer
Role: Frontend Developer
Teamsize: 7
Description: fanclub is a platform with fans and many idols.
Technologies: PHP/Laravel, VueJS, AWS(EC2, RDS, Autoscaling, CloudWatch)
Contact
Phone
0706-663-784
Email
recruit.cv@growupwork.com
Address
Birthday
1970/01/01
Gender
Male
Hobbies
Reading: reading technologies blog tech and research new javascript framework
References
Main duties: responsible for coding frontend
Achievement/Skills and Knowledge Gained:
Always get the job done before the deadline
Achieve 120% productivity compared to plan
E-learning system 2017/03 - 2019/01
Position: Full stack developer
Role: Web developer
Teamsize: 3
Description: e-learning is an online system of training through video.
Technologies: HTML/CSS/ReactJS, PHP/Laravel
Education
HCMUS
2010 - 2014
IT
Good Rank; GPA: 3.76
Language
English: TOEIC 600
Vietnamese: Native
Certicate
2018 Principle of UI/UX
Honours Awards
2019 Best staff of the year 2019, OneTech
""",
        },
    )

    assert response.status_code == 200
    data = response.json()["data"]

    assert data["personal"]["full_name"] == "Nguyễn Quốc Bình"
    assert data["personal"]["email"] == "recruit.cv@growupwork.com"
    assert data["personal"]["phone"] == "0706663784"
    assert data["personal"]["location"] is None

    assert len(data["experience"]) >= 2
    assert data["experience"][0]["company"] == "SaiGon Web Company"
    assert data["experience"][0]["role"] == "Frontend Developer"
    assert data["experience"][0]["start_date"] == "2020-10"
    assert data["experience"][0]["end_date"] == "present"
    assert data["experience"][1]["company"] == "XTech Corporation"
    assert data["experience"][1]["role"] == "Junior Web Developer"

    project_names = {project["name"] for project in data["projects"]}
    assert "Fanclub system" in project_names
    assert "E-learning system" in project_names

    assert data["education"][0]["institution"] == "HCMUS"
    assert data["education"][0]["degree"] == "Bachelor"
    assert data["education"][0]["field_of_study"] == "Information Technology"
    assert data["education"][0]["start_year"] == 2010
    assert data["education"][0]["end_year"] == 2014
    assert "GPA: 3.76" in data["education"][0]["description"]

    assert data["certifications"][0]["issued_year"] == 2018
    assert "Principle of UI/UX" in data["certifications"][0]["name"]
    assert data["achievements"][0]["year"] == 2019
    assert data["languages"] == [
        {"name": "English", "proficiency": "TOEIC 600"},
        {"name": "Vietnamese", "proficiency": "native"},
    ]
