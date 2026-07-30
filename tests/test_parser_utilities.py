from app.parsers.normalizers.duration_normalizer import (
    extract_date_range,
    months_between,
    normalize_date_label,
    parse_explicit_duration_months,
)
from app.parsers.section_splitter import split_sections


def test_duration_normalizer_supports_common_resume_formats():
    assert normalize_date_label("Sept 2020") == "2020-09"
    assert normalize_date_label("06/2021") == "2021-06"
    assert months_between("06/2021", "08/2022") == 14
    assert parse_explicit_duration_months("2 years 3 months") == 27
    assert parse_explicit_duration_months("2 nam 6 thang") == 30

    date_range = extract_date_range("Backend Engineer | 06/2021 - Present")

    assert date_range["start_date"] == "2021-06"
    assert date_range["end_date"] == "present"
    assert isinstance(date_range["duration_months"], int)


def test_section_splitter_supports_vietnamese_inline_headers():
    sections = split_sections(
        """
Kỹ năng: Python, Django, PostgreSQL
Kinh nghiệm làm việc:
Backend Developer - ABC Tech - 2020 - 2022
Dự án:
Recruiter API - FastAPI service
Ngôn ngữ: Tiếng Anh - Advanced
"""
    )

    assert sections["skills"] == "Python, Django, PostgreSQL"
    assert "Backend Developer" in sections["experience"]
    assert "Recruiter API" in sections["projects"]
    assert sections["languages"] == "Tiếng Anh - Advanced"
