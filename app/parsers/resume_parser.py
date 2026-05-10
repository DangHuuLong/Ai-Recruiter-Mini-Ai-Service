import re

from app.parsers.extractors.achievements import extract_achievements
from app.parsers.extractors.education import extract_education
from app.parsers.extractors.email_phone import extract_emails, extract_phones
from app.parsers.extractors.experience import extract_experience
from app.parsers.extractors.languages import extract_languages
from app.parsers.extractors.links import extract_links
from app.parsers.extractors.projects import extract_projects
from app.parsers.extractors.projects_certifications import extract_certifications
from app.parsers.extractors.skills import extract_skills
from app.parsers.normalizer import normalize_text, split_lines, strip_accents
from app.parsers.normalizers.skill_normalizer import normalize_skill
from app.parsers.section_splitter import split_sections
from app.schemas.resume import (
    ParsedResumeData,
    ResumeAchievement,
    ResumeCertification,
    ResumeEducation,
    ResumeExperience,
    ResumeLanguage,
    ResumePersonalInfo,
    ResumeProject,
    ResumeSkill,
)

LOCATION_LABELS = (
    "location",
    "address",
    "current address",
    "dia chi",
    "địa chỉ",
    "que quan",
    "quê quán",
    "noi o",
    "nơi ở",
    "thanh pho",
    "thành phố",
)

EMPTY_LOCATION_VALUES = {
    "",
    "birthday",
    "gender",
    "phone",
    "email",
    "skills",
    "education",
    "hcmus",
}

BAD_TEXT_MARKERS = ("·", "ï", "¿", "ˇ", "�")
RESUME_HEADING_NAMES = {
    "academic",
    "academic background",
    "achievements",
    "awards",
    "certification",
    "certifications",
    "contact",
    "education",
    "experience",
    "github",
    "languages",
    "links",
    "personal information",
    "profile",
    "projects",
    "skills",
    "summary",
    "technical skills",
    "work experience",
}
NAME_SCAN_STOP_HEADINGS = {"profile", "summary", "objective"}
NAME_NOISE_TOKENS = {
    "api",
    "asp.net",
    "backend",
    "bootstrap",
    "cloudinary",
    "css",
    "database",
    "expo",
    "express",
    "fastapi",
    "frontend",
    "git",
    "github",
    "java",
    "javascript",
    "jwt",
    "mobile",
    "mongodb",
    "mongoose",
    "native",
    "nestjs",
    "node",
    "postgre",
    "postgresql",
    "postman",
    "prisma",
    "python",
    "query",
    "react",
    "redux",
    "render",
    "rest",
    "sql",
    "storage",
    "supabase",
    "tailwind",
    "tanstack",
    "typescript",
    "vercel",
    "vite",
}
SUMMARY_STOP_PATTERNS = (
    r"@",
    r"^s\s*t\s*:",
    r"^phone\s*:",
    r"^ngay sinh\s*:",
    r"^date\s*:",
    r"^que quan\s*:",
    r"^address\s*:",
    r"^technical skills$",
    r"^skills$",
    r"^lien k",
    r"^github$",
)

PROJECT_RECOVERY_KEYWORDS = (
    "app",
    "application",
    "booking",
    "chat",
    "cinema",
    "clone",
    "commerce",
    "construction",
    "e-commerce",
    "ecommerce",
    "inventory",
    "learning",
    "management",
    "meeting",
    "mobile",
    "platform",
    "portfolio",
    "portal",
    "project",
    "recruiter",
    "recruitment",
    "shop",
    "site",
    "sneaker",
    "store",
    "system",
    "website",
)

PROJECT_RECOVERY_CONTEXT = (
    "position",
    "role",
    "teamsize",
    "team size",
    "description",
    "technologies",
    "technology",
    "tech stack",
    "key contributions",
    "key responsibilities",
    "built",
    "designed",
    "developed",
    "implemented",
    "integrated",
    "personal project",
    "cong nghe su dung",
    "mo ta chuc nang",
    "vai tro",
    "link github",
)

BAD_PROJECT_NAME_PREFIXES = (
    "admin",
    "auth",
    "cong nghe su dung",
    "crud",
    "customer",
    "link github",
    "mo ta chuc nang",
    "quan ly tai khoan",
    "thanh toan",
    "trang thai",
    "upload",
    "vai tro",
)

BAD_PROJECT_NAMES = {
    "backend developer",
    "developer",
    "frontend developer",
    "full stack developer",
    "fullstack developer",
    "game developer",
    "intern fullstack developer",
    "junior web developer",
    "personal project",
    "web developer",
}

SECTION_STOP_NAMES = {
    "contact",
    "education",
    "language",
    "languages",
    "certicate",
    "certificate",
    "certification",
    "honours awards",
    "honors awards",
    "skills",
    "soft skills",
}

URL_RE = re.compile(
    r"https?://(?:www\.)?(?:github\.com|gitlab\.com|bitbucket\.org)/[^\s)>,;]+",
    re.IGNORECASE,
)


def _extract_location(raw_text: str) -> str | None:
    lines = split_lines(raw_text)

    for index, raw_line in enumerate(lines):
        line = raw_line.strip()
        normalized = strip_accents(line).lower()

        if ":" in line:
            label, value = line.split(":", 1)
            normalized_label = strip_accents(label).lower().strip()

            if any(token in normalized_label for token in LOCATION_LABELS):
                cleaned = value.strip(" -|,")
                normalized_value = strip_accents(cleaned).lower().strip()
                return cleaned if normalized_value not in EMPTY_LOCATION_VALUES else None

        normalized_line = strip_accents(line).lower().strip()
        if normalized_line in {"address", "dia chi", "địa chỉ"}:
            next_value = lines[index + 1].strip() if index + 1 < len(lines) else ""
            normalized_next = strip_accents(next_value).lower().strip()
            if normalized_next in EMPTY_LOCATION_VALUES:
                return None
            if any(token in normalized_next for token in ["phone", "email", "birthday", "gender", "skills"]):
                return None
            return next_value or None

        if any(label in normalized for label in ["da nang", "đà nẵng", "ha noi", "hà nội", "ho chi minh"]):
            if len(line.split()) <= 10 and not any(token in normalized for token in ["email", "@", "sdt", "phone"]):
                return line

    return None


def _looks_like_bad_text(value: str | None) -> bool:
    return bool(value) and any(marker in value for marker in BAD_TEXT_MARKERS)


def _looks_like_resume_heading(value: str) -> bool:
    return _normalize_name(value) in RESUME_HEADING_NAMES


def _looks_like_name_noise(value: str) -> bool:
    normalized = _normalize_name(value)
    if not normalized:
        return True

    if "," in value or "/" in value or "&" in value:
        return True

    if any(token in normalized.split() for token in NAME_NOISE_TOKENS):
        return True

    if any(token in normalized for token in ("frontend", "backend", "database", "programming languages", "tools services")):
        return True

    return False


def _extract_full_name(raw_text: str) -> str | None:
    for line in split_lines(raw_text):
        normalized = _normalize_name(line)
        lowered = line.lower()

        if normalized in NAME_SCAN_STOP_HEADINGS:
            return None

        if _looks_like_bad_text(line) or _looks_like_resume_heading(line) or _looks_like_name_noise(line):
            continue
        if any(token in lowered for token in ["@", "http", "linkedin", "github", "phone", "email"]):
            continue
        if any(token in lowered for token in ["developer", "engineer", "manager", "designer", "analyst", "consultant"]):
            continue
        if any(char.isdigit() for char in line):
            continue
        if ":" in line:
            continue

        words = line.split()
        if 2 <= len(words) <= 5:
            return line

    return None


def _section_or_fallback(sections: dict[str, str], section_name: str) -> str:
    return sections.get(section_name) or sections.get("other", "")


def _combine_sections(*values: str | None) -> str:
    return "\n".join(value for value in values if value).strip()


def _normalize_name(value: str | None) -> str:
    normalized = strip_accents(value or "").lower()
    normalized = re.sub(r"[^a-z0-9+#./ ]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _is_summary_stop_line(line: str) -> bool:
    normalized = _normalize_name(line)
    return any(re.search(pattern, normalized, re.IGNORECASE) for pattern in SUMMARY_STOP_PATTERNS)


def _clean_summary(summary: str | None) -> str | None:
    if not summary:
        return None

    clean_lines: list[str] = []
    for line in split_lines(summary):
        if _is_summary_stop_line(line):
            break
        if _looks_like_bad_text(line) and len(line.split()) <= 5:
            break
        clean_lines.append(line)

    cleaned = "\n".join(clean_lines).strip()
    return cleaned or None


def _is_bad_project_name(name: str | None) -> bool:
    normalized = _normalize_name(name)
    if not normalized or normalized in BAD_PROJECT_NAMES:
        return True
    return any(normalized.startswith(prefix) for prefix in BAD_PROJECT_NAME_PREFIXES)


def _filter_project_items(items: list[dict]) -> list[dict]:
    return [item for item in items if not _is_bad_project_name(item.get("name"))]


def _dedupe_by_name(items: list[dict]) -> list[dict]:
    result = []
    seen_names = set()

    for item in items:
        name = item.get("name")
        normalized_name = name.lower().strip() if isinstance(name, str) else ""
        if not normalized_name or normalized_name in seen_names:
            continue

        seen_names.add(normalized_name)
        result.append(item)

    return result


def _dedupe_strings(values: list[str]) -> list[str]:
    result = []
    seen = set()

    for value in values:
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(value)

    return result


def _is_repo_url(url: str) -> bool:
    match = re.match(r"https?://(?:www\.)?(?:github\.com|gitlab\.com|bitbucket\.org)/([^/?#]+)/([^/?#]+)", url, re.IGNORECASE)
    if not match:
        return False

    repo_name = match.group(2).lower().strip()
    return repo_name not in {"", "repositories", "projects", "stars", "followers", "following"}


def _project_repo_urls_from_text(raw_text: str, personal_github_url: str | None = None) -> list[str]:
    personal_url = (personal_github_url or "").rstrip("/").lower()
    urls = []

    for match in URL_RE.finditer(raw_text or ""):
        url = match.group(0).rstrip(".,;)")
        if url.rstrip("/").lower() == personal_url:
            continue
        if _is_repo_url(url):
            urls.append(url)

    return _dedupe_strings(urls)


def _attach_project_urls(project_items: list[dict], raw_text: str, personal_github_url: str | None = None) -> list[dict]:
    repo_urls = _project_repo_urls_from_text(raw_text, personal_github_url)
    url_index = 0

    for item in project_items:
        item_urls = list(item.get("urls") or [])
        if item.get("url") and item["url"] not in item_urls:
            item_urls.insert(0, item["url"])

        if not item_urls and url_index < len(repo_urls):
            item_urls.append(repo_urls[url_index])
            url_index += 1

        item["urls"] = _dedupe_strings(item_urls)

    return project_items


def parse_resume(raw_text: str) -> ParsedResumeData:
    """Parse resume free text into ParsedResumeData using deterministic heuristics."""
    normalized_text = normalize_text(raw_text, lowercase=False, remove_accents=False, preserve_lines=True)
    sections = split_sections(normalized_text)

    personal = ResumePersonalInfo()
    emails = extract_emails(raw_text)
    phones = extract_phones(raw_text)
    links = extract_links(raw_text)

    if emails:
        personal.email = emails[0]
    if phones:
        personal.phone = phones[0]
    personal.full_name = _extract_full_name(raw_text)
    personal.location = _extract_location(raw_text)
    personal.linkedin_url = links.get("linkedin")
    personal.github_url = links.get("github")
    personal.portfolio_url = links.get("portfolio")

    skills = [
        ResumeSkill(
            name=skill["name"],
            normalized_name=normalize_skill(skill["name"]),
            category=skill.get("category"),
            evidence=skill.get("evidence"),
        )
        for skill in extract_skills(raw_text)
        if skill.get("name")
    ]

    education = [
        ResumeEducation(
            institution=item.get("institution"),
            degree=item.get("degree"),
            field_of_study=item.get("field_of_study"),
            start_year=item.get("start_year"),
            end_year=item.get("end_year"),
            description=item.get("description"),
        )
        for item in extract_education(_section_or_fallback(sections, "education"))
    ]

    experience = [
        ResumeExperience(
            company=item.get("company"),
            role=item.get("role"),
            start_date=item.get("start_date"),
            end_date=item.get("end_date"),
            duration_months=item.get("duration_months"),
            responsibilities=item.get("responsibilities", []),
            technologies=item.get("technologies", []),
        )
        for item in extract_experience(_experience_source(sections, normalized_text))
    ]

    projects_source = _projects_source(sections)
    project_items = _filter_project_items(
        [
            *extract_projects(projects_source),
            *extract_projects(_recover_dated_project_source(normalized_text)),
        ]
    )
    project_items = _dedupe_by_name(project_items)
    project_items = _attach_project_urls(project_items, raw_text, personal.github_url)
    projects = [
        ResumeProject(
            name=item.get("name"),
            role=item.get("role"),
            start_date=item.get("start_date"),
            end_date=item.get("end_date"),
            description=item.get("description"),
            technologies=item.get("technologies", []),
            urls=item.get("urls", []),
        )
        for item in project_items
    ]

    certification_source = _section_or_fallback(sections, "certifications")
    certifications = [
        ResumeCertification(
            name=item.get("name"),
            issuer=item.get("issuer"),
            issued_year=item.get("issued_year"),
            url=item.get("url"),
        )
        for item in extract_certifications(
            certification_source,
            allow_year_only=bool(sections.get("certifications")),
        )
    ]

    achievements_source = _combine_sections(sections.get("achievements"), sections.get("experience"), sections.get("projects"))
    achievements = [
        ResumeAchievement(title=item.get("title"), description=item.get("description"), year=item.get("year"))
        for item in extract_achievements(achievements_source, require_keyword=True)
    ]
    achievements.extend(
        ResumeAchievement(title=item.get("title"), description=item.get("description"), year=item.get("year"))
        for item in extract_achievements(sections.get("other", ""), require_keyword=True)
    )

    languages = [
        ResumeLanguage(name=item["name"], proficiency=item.get("proficiency"))
        for item in extract_languages(_section_or_fallback(sections, "languages"))
        if item.get("name")
    ]

    return ParsedResumeData(
        personal=personal,
        summary=_clean_summary(sections.get("summary")),
        skills=skills,
        education=education,
        experience=experience,
        projects=projects,
        certifications=certifications,
        achievements=achievements,
        languages=languages,
    )


def parse_resume_mock(raw_text: str) -> ParsedResumeData:
    return parse_resume(raw_text)


def _experience_source(sections: dict[str, str], raw_text: str | None = None) -> str:
    if sections.get("experience"):
        return sections["experience"]

    projects_text = sections.get("projects", "")
    if _looks_like_misplaced_experience(split_lines(projects_text)):
        return projects_text

    return _extract_experience_from_unsectioned_text(_combine_sections(sections.get("other", ""), raw_text))


def _projects_source(sections: dict[str, str]) -> str:
    projects_text = sections.get("projects", "")
    if projects_text:
        if _looks_like_misplaced_experience(split_lines(projects_text)):
            return _extract_project_tail_from_text(projects_text)
        return projects_text

    return _combine_sections(
        _extract_project_tail_from_text(sections.get("experience", "")),
        _extract_project_tail_from_text(sections.get("other", "")),
    )


def _looks_like_project_section_start(lines: list[str]) -> bool:
    if not lines:
        return False

    first = _normalize_name(lines[0])
    rest = "\n".join(_normalize_name(line) for line in lines[1:8])

    if any(keyword in first for keyword in PROJECT_RECOVERY_KEYWORDS):
        return True

    return any(token in rest for token in PROJECT_RECOVERY_CONTEXT)


def _looks_like_misplaced_experience(lines: list[str]) -> bool:
    if not lines:
        return False

    first = strip_accents(lines[0]).lower()
    second = strip_accents(lines[1]).lower() if len(lines) > 1 else ""
    third = strip_accents(lines[2]).lower() if len(lines) > 2 else ""

    if any(token in first for token in ["career history", "company", "corporation"]):
        return True

    if _looks_like_project_section_start(lines):
        return False

    if any(token in first for token in ["developer", "engineer", "manager"]):
        return True
    if any(token in second for token in ["developer", "engineer", "manager"]):
        return True
    if re.search(r"(?:19|20)\d{2}[/.-]\d{1,2}\s*(?:-|–|—|to)\s*", third):
        return True

    return False


def _extract_experience_from_unsectioned_text(text: str) -> str:
    lines = split_lines(text)
    candidate_lines = []

    for line in lines:
        lowered = strip_accents(line).lower()

        if not candidate_lines:
            if " at " in lowered:
                candidate_lines.append(line)
                continue
            if re.search(r"(?:19|20)\d{2}|(?:0?[1-9]|1[0-2])[/.-](?:19|20)\d{2}", line):
                if any(role in lowered for role in ["developer", "engineer", "intern", "manager", "designer", "analyst"]):
                    candidate_lines.append(line)
                continue
            continue

        if any(token in lowered for token in ["university", "đại học", "dai hoc", "certified", "certificate", "english -", "language"]):
            break

        candidate_lines.append(line)

    return "\n".join(candidate_lines)


def _looks_like_recoverable_project_start(line: str, next_lines: list[str]) -> bool:
    normalized = strip_accents(line).lower()
    date_match = re.search(
        r"(?:19|20)\d{2}(?:[/.-]\d{1,2})?\s*(?:-|–|—|to|den|đến)\s*(?:nay|present|current|now|(?:19|20)\d{2}(?:[/.-]\d{1,2})?)",
        normalized,
    )
    if not date_match:
        return False

    title = normalized[: date_match.start()].strip(" -|,")
    if not title or _is_bad_project_name(title):
        return False

    if not any(keyword in title for keyword in PROJECT_RECOVERY_KEYWORDS):
        return False

    context = "\n".join(strip_accents(item).lower() for item in next_lines[:6])
    return any(token in context for token in PROJECT_RECOVERY_CONTEXT)


def _recover_dated_project_source(text: str) -> str:
    lines = split_lines(text)
    chunks: list[str] = []
    current: list[str] = []

    for index, line in enumerate(lines):
        if _looks_like_recoverable_project_start(line, lines[index + 1 : index + 7]):
            if current:
                chunks.append("\n".join(current))
            current = [line]
            continue

        if current:
            normalized = _normalize_name(line)
            if normalized in SECTION_STOP_NAMES:
                chunks.append("\n".join(current))
                current = []
                continue
            current.append(line)

    if current:
        chunks.append("\n".join(current))

    return "\n".join(chunks)


def _extract_project_tail_from_text(text: str) -> str:
    lines = split_lines(text)
    tail_lines = []
    collecting = False

    for index, line in enumerate(lines):
        normalized = strip_accents(line).lower()
        next_lines = "\n".join(lines[index + 1 : index + 6]).lower()

        looks_like_project_start = (
            re.search(r"(?:system|platform|app|website|portfolio|project|recruiter|sneaker|cinema|commerce|management|construction|clone|meeting|chat|inventory|learning|mobile|recruitment)\b", normalized)
            and any(
                token in next_lines
                for token in [
                    "position",
                    "role",
                    "teamsize",
                    "description",
                    "technologies",
                    "tech stack",
                    "key contributions",
                    "key responsibilities",
                    "built",
                    "designed",
                    "developed",
                    "implemented",
                    "integrated",
                    "personal project",
                ]
            )
        )

        if looks_like_project_start:
            collecting = True

        if collecting:
            tail_lines.append(line)

    return "\n".join(tail_lines)
