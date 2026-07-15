import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable

from app.schemas.job_description import JobSkill, ParsedJobDescriptionData

logger = logging.getLogger(__name__)

SECTION_ORDER: tuple[str, ...] = ("responsibilities", "requirements", "nice_to_have", "benefits")

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"(?:\+?\d[\s.-]?){8,15}")
_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)


def _looks_like_contact_info(line: str) -> bool:
    return bool(_EMAIL_RE.search(line) or _PHONE_RE.search(line) or _URL_RE.search(line))


SECTION_ALIASES: dict[str, tuple[str, ...]] = {
    "responsibilities": (
        "responsibilities",
        "responsibility",
        "key responsibilities",
        "what you will do",
        "what you'll do",
        "job description",
        "role description",
        "duties",
        "tasks",
        "mo ta cong viec",
        "trach nhiem",
        "nhiem vu",
        "cong viec",
    ),
    "requirements": (
        "requirements",
        "requirement",
        "must have",
        "must-have",
        "qualifications",
        "minimum qualifications",
        "required qualifications",
        "what we need",
        "yeu cau",
        "yeu cau cong viec",
        "bat buoc",
    ),
    "nice_to_have": (
        "nice to have",
        "nice-to-have",
        "preferred qualifications",
        "preferred skills",
        "preferred",
        "bonus points",
        "plus",
        "a plus",
        "good to have",
        "uu tien",
        "diem cong",
        "loi the",
    ),
    "benefits": (
        "benefits",
        "perks",
        "why join us",
        "phuc loi",
        "quyen loi",
    ),
}

TITLE_LABELS = ("job title", "position", "role", "title", "vi tri", "chuc danh")
NORMALIZED_TITLE_LABELS = {"job title", "position", "role", "title", "vi tri", "chuc danh"}

EXPLICIT_SENIORITY_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\b(lead|principal|staff|architect|manager)\b", "lead"),
    (r"\b(senior|sr\.?)\b", "senior"),
    (r"\b(mid[ -]?level|middle)\b", "mid"),
    (r"\b(junior|jr\.?)\b", "junior"),
    (r"\b(fresher|entry[ -]?level|graduate)\b", "fresher"),
    (r"\b(internship|intern|thuc tap|thuc tap sinh)\b", "intern"),
)

EXPERIENCE_SENIORITY_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\b5\+?\s*years?\b", "senior"),
    (r"\b[23]\+?\s*years?\b", "mid"),
    (r"\b(?:0\s*-\s*1|1\+?)\s*years?\b", "junior"),
)

EMPLOYMENT_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\b(full[ -]?time|toan thoi gian)\b", "full-time"),
    (r"\b(part[ -]?time|ban thoi gian)\b", "part-time"),
    (r"\b(contract|contractor|freelance|hop dong)\b", "contract"),
    (r"\b(internship|intern|thuc tap)\b", "internship"),
    (r"\b(remote|hybrid|onsite|on-site)\b", "full-time"),
)

TITLE_PATTERNS: tuple[str, ...] = (
    r"(?:we are looking for|we need|hiring|tuyen dung|can tuyen)\s+(?:an?\s+)?(?P<title>[A-Z][A-Za-z0-9 /.#+-]*(?:Developer|Engineer|Designer|Analyst|Tester|Manager|Specialist|Intern|Lead))",
)

EXPERIENCE_PATTERNS: tuple[str, ...] = (
    r"(?:minimum|min|at least|from|tu|toi thieu)\s*(?P<years>\d+)\+?\s*(?:years?|yrs?|nam)",
    r"(?P<years>\d+)\+?\s*(?:years?|yrs?|nam)\s+(?:of\s+)?(?:experience|kinh nghiem)",
)

EDUCATION_PATTERNS: tuple[str, ...] = (
    r"\b(bachelor'?s? degree|b\.s\.?|bs|university degree|college degree)\b[^.\n;]*",
    r"\b(master'?s? degree|m\.s\.?|ms|mba)\b[^.\n;]*",
    r"\b(phd|ph\.d\.?|doctorate)\b[^.\n;]*",
    r"\b(computer science|software engineering|information technology|cntt|cong nghe thong tin)\b[^.\n;]*",
    r"\b(dai hoc|cao dang|cu nhan|ky su)\b[^.\n;]*",
)

DOMAIN_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("backend", ("backend", "back-end", "server-side", "api", "microservice")),
    ("frontend", ("frontend", "front-end", "ui", "web interface")),
    ("fullstack", ("full stack", "full-stack", "fullstack")),
    ("mobile", ("mobile", "android", "ios", "react native", "flutter")),
    ("data", ("data engineering", "data platform", "data pipeline", "etl", "warehouse", "analytics", "bi")),
    ("ai", ("ai", "machine learning", "ml", "llm", "nlp", "computer vision")),
    ("devops", ("devops", "ci/cd", "kubernetes", "docker", "cloud", "aws", "azure", "gcp")),
    ("qa", ("qa", "quality assurance", "tester", "testing", "automation test")),
    ("ecommerce", ("e-commerce", "ecommerce", "payment", "checkout", "order")),
    ("finance", ("finance", "fintech", "banking", "payment")),
    ("recruitment", ("recruitment", "hiring", "ats", "candidate")),
    ("rest api", ("rest api", "restful", "api")),
)

SHORT_ALIASES = {"go", "js", "ts"}


@dataclass(frozen=True)
class SkillDefinition:
    name: str
    normalized_name: str
    aliases: tuple[str, ...]
    category: str


SKILL_CATALOG: tuple[SkillDefinition, ...] = (
    SkillDefinition("Python", "python", ("python",), "language"),
    SkillDefinition("JavaScript", "javascript", ("javascript", "js"), "language"),
    SkillDefinition("TypeScript", "typescript", ("typescript", "ts"), "language"),
    SkillDefinition("Java", "java", ("java",), "language"),
    SkillDefinition("C#", "csharp", ("c#", "c sharp", ".net"), "language"),
    SkillDefinition("PHP", "php", ("php",), "language"),
    SkillDefinition("Go", "go", ("golang", "go"), "language"),
    SkillDefinition("FastAPI", "fastapi", ("fastapi", "fast api"), "backend"),
    SkillDefinition("Django", "django", ("django",), "backend"),
    SkillDefinition("Flask", "flask", ("flask",), "backend"),
    SkillDefinition("Spring Boot", "spring_boot", ("spring boot", "springboot"), "backend"),
    SkillDefinition("Node.js", "nodejs", ("node.js", "nodejs", "node js"), "backend"),
    SkillDefinition("NestJS", "nestjs", ("nestjs", "nest.js", "nest js"), "backend"),
    SkillDefinition("Express.js", "expressjs", ("express.js", "expressjs", "express js"), "backend"),
    SkillDefinition("React", "react", ("react", "reactjs", "react.js"), "frontend"),
    SkillDefinition("Next.js", "nextjs", ("next.js", "nextjs", "next js"), "frontend"),
    SkillDefinition("Vue.js", "vuejs", ("vue.js", "vuejs", "vue js"), "frontend"),
    SkillDefinition("Angular", "angular", ("angular",), "frontend"),
    SkillDefinition("HTML", "html", ("html", "html5"), "frontend"),
    SkillDefinition("CSS", "css", ("css", "css3"), "frontend"),
    SkillDefinition("Tailwind CSS", "tailwind_css", ("tailwind", "tailwind css"), "frontend"),
    SkillDefinition("PostgreSQL", "postgresql", ("postgresql", "postgres", "postgre"), "database"),
    SkillDefinition("MySQL", "mysql", ("mysql",), "database"),
    SkillDefinition("SQL Server", "sql_server", ("sql server", "mssql"), "database"),
    SkillDefinition("MongoDB", "mongodb", ("mongodb", "mongo"), "database"),
    SkillDefinition("Redis", "redis", ("redis",), "cache"),
    SkillDefinition("REST API", "rest_api", ("rest api", "restful", "restful api"), "api"),
    SkillDefinition("GraphQL", "graphql", ("graphql", "graph ql"), "api"),
    SkillDefinition("Docker", "docker", ("docker",), "devops"),
    SkillDefinition("Kubernetes", "kubernetes", ("kubernetes", "k8s"), "devops"),
    SkillDefinition("AWS", "aws", ("aws", "amazon web services"), "cloud"),
    SkillDefinition("Azure", "azure", ("azure",), "cloud"),
    SkillDefinition("GCP", "gcp", ("gcp", "google cloud"), "cloud"),
    SkillDefinition("Git", "git", ("git",), "tool"),
    SkillDefinition("CI/CD", "ci_cd", ("ci/cd", "cicd", "continuous integration"), "devops"),
    SkillDefinition("Prisma", "prisma", ("prisma",), "orm"),
    SkillDefinition("Supabase", "supabase", ("supabase",), "platform"),
    SkillDefinition("Pytest", "pytest", ("pytest",), "testing"),
    SkillDefinition("Jest", "jest", ("jest",), "testing"),
)

BULLET_RE = re.compile(r"^\s*(?:[-*+•]|\d+[.)])\s*")


def parse_job_description(raw_text: str) -> ParsedJobDescriptionData:
    clean_text = _normalize_text(raw_text)
    sections = _split_sections(clean_text)  # ML-assisted when enabled, see below

    responsibilities = _extract_section_items(sections, "responsibilities")
    requirements = _extract_section_items(sections, "requirements")
    nice_to_have = _extract_section_items(sections, "nice_to_have")

    required_skill_text = "\n".join(requirements) or clean_text
    preferred_skill_text = "\n".join(nice_to_have)

    required_skills = _extract_skills(required_skill_text, is_core=True, weight_hint=1.0)
    preferred_skills = _extract_skills(preferred_skill_text, is_core=False, weight_hint=0.6)

    if not required_skills:
        preferred_names = {skill.normalized_name for skill in preferred_skills}
        required_skills = [
            skill
            for skill in _extract_skills(clean_text, is_core=True, weight_hint=0.9)
            if skill.normalized_name not in preferred_names
        ]

    return ParsedJobDescriptionData(
        title=_detect_title(clean_text),
        seniority=_detect_seniority(clean_text),
        employment_type=_detect_employment_type(clean_text),
        responsibilities=responsibilities,
        requirements=requirements,
        nice_to_have=nice_to_have,
        required_skills=required_skills,
        preferred_skills=preferred_skills,
        min_experience_years=_detect_min_experience_years(clean_text),
        education_requirement=_detect_education_requirement(clean_text),
        domain_keywords=_extract_domain_keywords(clean_text),
    )


def _normalize_text(raw_text: str) -> str:
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u0000", "")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    return "\n".join(line for line in lines if line)


def _regex_split_sections(text: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {key: [] for key in SECTION_ALIASES}
    sections["other"] = []
    current_section: str | None = None

    for line in text.split("\n"):
        section = _match_section_heading(line)
        if section:
            current_section = section
            remainder = _strip_heading(line)
            if remainder:
                sections[current_section].append(remainder)
            continue

        if current_section:
            sections[current_section].append(line)
        else:
            # Previously dropped entirely (lines before the first recognized
            # header, e.g. a title/intro line) — kept in "other" instead, both
            # to avoid silently losing content and as the substrate for
            # ML-assisted reclassification (see _ml_assisted_split below).
            sections["other"].append(line)

    return sections


def _ml_assisted_split(text: str) -> dict[str, list[str]]:
    """Regex-first, ML-fallback-for-ambiguous-lines only — mirrors
    app/parsers/section_splitter.py's design for resumes. A header match is a
    high-precision signal, so any line the regex pass confidently assigns to
    a real section is kept as-is. The regex splitter can only ever leave a
    line in "other" for lines seen BEFORE the first recognized header (or the
    whole document, if no header is ever recognized) — headers always move
    `current_section` to a real section, so "other" is exactly the
    heuristic's blind spot (e.g. a JD's title/intro line, or a JD with
    non-standard headers). Reclassifies just that blind spot with the
    distilled JD section-classifier model (see
    app/ml/jd_section_classifier_model.py), Viterbi-smoothed.

    Raises whatever get_jd_section_classifier_model() raises when the model
    is disabled or its artifacts are missing — callers (_split_sections) must
    catch that and fall back to _regex_split_sections.
    """
    from app.ml.jd_section_classifier_model import get_jd_section_classifier_model

    sections = _regex_split_sections(text)
    other_lines = sections.get("other", [])
    if not other_lines:
        return sections

    reclassifiable_indices = [i for i, line in enumerate(other_lines) if not _looks_like_contact_info(line)]
    if not reclassifiable_indices:
        return sections

    reclassifiable_lines = [other_lines[i] for i in reclassifiable_indices]
    bundle = get_jd_section_classifier_model()
    predicted_labels = bundle.predict_labels(reclassifiable_lines)
    label_by_index = dict(zip(reclassifiable_indices, predicted_labels))

    remaining_other: list[str] = []
    additions: dict[str, list[str]] = {}
    for i, line in enumerate(other_lines):
        label = label_by_index.get(i, "other")
        if label == "other":
            remaining_other.append(line)
        else:
            additions.setdefault(label, []).append(line)

    merged = {key: list(value) for key, value in sections.items()}
    for label, lines in additions.items():
        # These lines occurred earliest in the document, so they're
        # prepended ahead of whatever content the regex pass already
        # assigned to this section.
        merged[label] = lines + merged.get(label, [])
    merged["other"] = remaining_other

    return merged


def _split_sections(text: str) -> dict[str, list[str]]:
    try:
        return _ml_assisted_split(text)
    except Exception:
        logger.debug("ML-assisted JD section split unavailable, falling back to regex.", exc_info=True)
        return _regex_split_sections(text)


def matches_known_jd_header(line: str) -> bool:
    """Whether the regex/alias splitter recognizes `line` as a section header.

    Exposed as a small public signal (mirrors
    app.parsers.section_splitter.matches_known_header) so the JD
    section-classifier's feature extractor
    (app/ml/jd_section_classifier_features.py) can feed the existing
    heuristic's own confidence in as a feature.
    """
    return _match_section_heading(line) is not None


def _match_section_heading(line: str) -> str | None:
    normalized = _normalize_for_match(line).strip(" :.-")
    for section, aliases in SECTION_ALIASES.items():
        for alias in aliases:
            if normalized == alias or normalized.startswith(f"{alias} "):
                return section
    return None


def _strip_heading(line: str) -> str:
    if ":" in line:
        return line.split(":", 1)[1].strip()
    return ""


def _extract_section_items(sections: dict[str, list[str]], section: str) -> list[str]:
    items: list[str] = []
    for raw_line in sections.get(section, []):
        line = BULLET_RE.sub("", raw_line).strip(" -;.")
        if not line or _looks_like_noise(line):
            continue
        items.append(line)
    return _dedupe_preserve_order(items)


def _extract_skills(text: str, *, is_core: bool, weight_hint: float) -> list[JobSkill]:
    text_for_match = _normalize_for_match(text)
    skills: list[JobSkill] = []
    seen: set[str] = set()

    for definition in SKILL_CATALOG:
        if definition.normalized_name in seen:
            continue
        if any(_contains_alias(text_for_match, alias) for alias in definition.aliases):
            skills.append(
                JobSkill(
                    name=definition.name,
                    normalized_name=definition.normalized_name,
                    is_core=is_core,
                    weight_hint=weight_hint,
                )
            )
            seen.add(definition.normalized_name)

    return skills


def _contains_alias(text_for_match: str, alias: str) -> bool:
    normalized_alias = _normalize_for_match(alias)
    if not normalized_alias:
        return False

    if normalized_alias in SHORT_ALIASES:
        return bool(re.search(rf"(?<![a-z0-9.]){re.escape(normalized_alias)}(?![a-z0-9])", text_for_match))

    return bool(re.search(rf"(?<![a-z0-9]){re.escape(normalized_alias)}(?![a-z0-9])", text_for_match))


def _detect_title(text: str) -> str | None:
    for line in text.split("\n")[:12]:
        labeled_title = _extract_labeled_title(line)
        if labeled_title:
            return labeled_title

    for pattern in TITLE_PATTERNS:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return _clean_title(match.group("title"))

    for line in text.split("\n")[:8]:
        if re.search(r"\b(developer|engineer|designer|analyst|tester|manager|intern|lead)\b", line, flags=re.IGNORECASE):
            return _clean_title(line)
    return None


def _extract_labeled_title(line: str) -> str | None:
    if ":" not in line:
        return None
    label, value = line.split(":", 1)
    normalized_label = _normalize_for_match(label)
    if normalized_label in NORMALIZED_TITLE_LABELS and value.strip():
        return _clean_title(value)
    return None


def _clean_title(value: str) -> str:
    title = BULLET_RE.sub("", value).strip(" .,-;:")
    label_pattern = r"^(?:job\s+title|position|role|title|vị\s*trí|vi\s*tri|chức\s*danh|chuc\s*danh)\s*:\s*"
    title = re.sub(label_pattern, "", title, flags=re.IGNORECASE).strip(" .,-;:")
    title = re.split(r"\s+(?:with|for|who|that|to)\s+", title, maxsplit=1, flags=re.IGNORECASE)[0]
    return title[:80].strip()


def _detect_seniority(text: str) -> str | None:
    normalized = _normalize_for_match(text)

    for pattern, seniority in EXPLICIT_SENIORITY_PATTERNS:
        if re.search(pattern, normalized, flags=re.IGNORECASE):
            return seniority

    for pattern, seniority in EXPERIENCE_SENIORITY_PATTERNS:
        if re.search(pattern, normalized, flags=re.IGNORECASE):
            return seniority

    return None


def _detect_employment_type(text: str) -> str | None:
    normalized = _normalize_for_match(text)
    for pattern, employment_type in EMPLOYMENT_PATTERNS:
        if re.search(pattern, normalized, flags=re.IGNORECASE):
            return employment_type
    return None


def _detect_min_experience_years(text: str) -> int | None:
    normalized = _normalize_for_match(text)
    years: list[int] = []
    for pattern in EXPERIENCE_PATTERNS:
        for match in re.finditer(pattern, normalized, flags=re.IGNORECASE):
            years.append(int(match.group("years")))
    return min(years) if years else None


def _detect_education_requirement(text: str) -> str | None:
    for pattern in EDUCATION_PATTERNS:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(0).strip(" .;:-")
    return None


def _extract_domain_keywords(text: str) -> list[str]:
    normalized = _normalize_for_match(text)
    keywords: list[str] = []
    for keyword, aliases in DOMAIN_KEYWORDS:
        if any(_contains_alias(normalized, alias) for alias in aliases):
            keywords.append(keyword)
    return _dedupe_preserve_order(keywords)


def _looks_like_noise(line: str) -> bool:
    normalized = _normalize_for_match(line)
    return normalized in {"and", "or", "the", "job", "candidate", "benefits"}


def _normalize_for_match(value: str) -> str:
    lowered = value.lower().replace("đ", "d")
    normalized = unicodedata.normalize("NFKD", lowered)
    without_accents = "".join(char for char in normalized if not unicodedata.combining(char))
    cleaned = re.sub(r"[^a-z0-9#+./-]+", " ", without_accents)
    return re.sub(r"\s+", " ", cleaned).strip()


def _dedupe_preserve_order(items: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        key = _normalize_for_match(item)
        if key and key not in seen:
            result.append(item)
            seen.add(key)
    return result
