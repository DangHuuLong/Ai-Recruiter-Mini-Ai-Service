import re
from collections.abc import Iterable


SKILL_CATALOG = {
    "Python": {"category": "language", "aliases": ["python"]},
    "JavaScript": {"category": "language", "aliases": ["javascript", "js"]},
    "TypeScript": {"category": "language", "aliases": ["typescript", "ts"]},
    "Java": {"category": "language", "aliases": ["java"]},
    "PHP": {"category": "language", "aliases": ["php"]},
    "Ruby": {"category": "language", "aliases": ["ruby"]},
    "C": {"category": "language", "aliases": ["c"]},
    "C#": {"category": "language", "aliases": ["c#"]},
    "C++": {"category": "language", "aliases": ["c++"]},
    "Go": {"category": "language", "aliases": ["go", "golang"]},
    "SQL": {"category": "database", "aliases": ["sql"]},
    "HTML": {"category": "frontend", "aliases": ["html"]},
    "CSS": {"category": "frontend", "aliases": ["css"]},
    "Bootstrap": {"category": "frontend", "aliases": ["bootstrap"]},
    "React": {"category": "frontend", "aliases": ["react", "reactjs", "react.js"]},
    "React Native": {"category": "mobile", "aliases": ["react native"]},
    "Expo": {"category": "mobile", "aliases": ["expo"]},
    "Vite": {"category": "frontend", "aliases": ["vite"]},
    "Next.js": {"category": "frontend", "aliases": ["next.js", "nextjs"]},
    "Vue": {"category": "frontend", "aliases": ["vue", "vue.js", "vuejs"]},
    "Tailwind CSS": {"category": "frontend", "aliases": ["tailwind css", "tailwind"]},
    "shadcn/ui": {"category": "frontend", "aliases": ["shadcn/ui", "shadcn ui", "shadcn"]},
    "Zustand": {"category": "frontend", "aliases": ["zustand"]},
    "Redux Toolkit": {"category": "frontend", "aliases": ["redux toolkit", "rtk"]},
    "TanStack Query": {"category": "frontend", "aliases": ["tanstack query", "react query"]},
    "React Context": {"category": "frontend", "aliases": ["react context", "context api"]},
    "Flutter": {"category": "mobile", "aliases": ["flutter"]},
    "BLoC/Cubit": {"category": "mobile", "aliases": ["bloc/cubit", "bloc", "cubit"]},
    "Dio": {"category": "mobile", "aliases": ["dio"]},
    "Node.js": {"category": "backend", "aliases": ["node.js", "nodejs", "node"]},
    "ASP.NET MVC": {"category": "backend", "aliases": ["asp.net mvc", "asp net mvc"]},
    "NestJS": {"category": "backend", "aliases": ["nestjs", "nest.js"]},
    "Express": {"category": "backend", "aliases": ["express", "express.js"]},
    "Socket.IO": {"category": "backend", "aliases": ["socket.io", "socket io"]},
    "RBAC": {"category": "backend", "aliases": ["rbac", "role based access control", "role-based access control"]},
    "JWT": {"category": "backend", "aliases": ["jwt", "json web token", "json web tokens"]},
    "FastAPI": {"category": "backend", "aliases": ["fastapi", "fast api"]},
    "Django": {"category": "backend", "aliases": ["django"]},
    "Flask": {"category": "backend", "aliases": ["flask"]},
    "Laravel": {"category": "backend", "aliases": ["laravel"]},
    "Spring Boot": {"category": "backend", "aliases": ["spring boot", "springboot"]},
    ".NET": {"category": "backend", "aliases": [".net", "dotnet"]},
    "REST API": {"category": "backend", "aliases": ["rest api", "rest apis", "restful api", "rest"]},
    "GraphQL": {"category": "backend", "aliases": ["graphql", "graph ql"]},
    "PostgreSQL": {"category": "database", "aliases": ["postgresql", "postgres", "postgre sql"]},
    "MySQL": {"category": "database", "aliases": ["mysql"]},
    "MongoDB": {"category": "database", "aliases": ["mongodb", "mongo db"]},
    "Mongoose": {"category": "orm", "aliases": ["mongoose"]},
    "Redis": {"category": "database", "aliases": ["redis"]},
    "Docker": {"category": "devops", "aliases": ["docker"]},
    "Kubernetes": {"category": "devops", "aliases": ["kubernetes", "k8s"]},
    "CI/CD": {"category": "devops", "aliases": ["ci/cd", "cicd", "continuous integration"]},
    "GitHub Actions": {"category": "devops", "aliases": ["github actions", "gh actions"]},
    "Nginx": {"category": "devops", "aliases": ["nginx"]},
    "AWS": {"category": "cloud", "aliases": ["aws", "amazon web services"]},
    "AWS EC2": {"category": "cloud", "aliases": ["aws ec2", "ec2"]},
    "Azure": {"category": "cloud", "aliases": ["azure", "microsoft azure"]},
    "GCP": {"category": "cloud", "aliases": ["gcp", "google cloud"]},
    "Firebase": {"category": "cloud", "aliases": ["firebase"]},
    "Supabase": {"category": "cloud", "aliases": ["supabase"]},
    "Supabase Storage": {"category": "cloud", "aliases": ["supabase storage"]},
    "Cloudinary": {"category": "cloud", "aliases": ["cloudinary"]},
    "Vercel": {"category": "cloud", "aliases": ["vercel"]},
    "Render": {"category": "cloud", "aliases": ["render"]},
    "Git": {"category": "tooling", "aliases": ["git"]},
    "GitHub": {"category": "tooling", "aliases": ["github"]},
    "Postman": {"category": "tooling", "aliases": ["postman"]},
    "Prisma": {"category": "orm", "aliases": ["prisma"]},
    "Whisper AI": {"category": "ai", "aliases": ["whisper ai", "openai whisper", "whisper"]},
    "Gemini AI": {"category": "ai", "aliases": ["gemini ai", "gemini"]},
    "Jest": {"category": "testing", "aliases": ["jest"]},
    "Pytest": {"category": "testing", "aliases": ["pytest"]},
}


def _alias_pattern(alias: str) -> re.Pattern:
    escaped = re.escape(alias)

    if alias == ".net":
        return re.compile(rf"(?<![\w+]){escaped}(?![\w+])", re.IGNORECASE)

    # Keep short aliases strict so they do not fire on framework suffixes
    # such as Node.js, Express.js, Next.js, or file extensions.
    if alias in {"js", "ts"}:
        return re.compile(rf"(?<![.\w]){escaped}(?![\w])", re.IGNORECASE)

    # Plain C should only match standalone C, C/C++, or C, C++ style language lists.
    # This avoids false positives in words like CSS, Cloudinary, React, or academic.
    if alias == "c":
        return re.compile(r"(?<![A-Za-z0-9+#.])c(?![A-Za-z0-9#.])", re.IGNORECASE)

    # Do not extract plain React from React Native; React Native has its own
    # catalog entry and should remain the more specific match.
    if alias == "react":
        return re.compile(rf"\b{escaped}\b(?!\s+native)", re.IGNORECASE)

    # Do not extract plain CSS from Tailwind CSS; Tailwind CSS is the explicit
    # technology in that phrase.
    if alias == "css":
        return re.compile(rf"(?<!tailwind\s)\b{escaped}\b", re.IGNORECASE)

    if alias in {"c#", "c++", "ci/cd", "shadcn/ui", "socket.io", "bloc/cubit"}:
        return re.compile(rf"(?<!\w){escaped}(?!\w)", re.IGNORECASE)

    return re.compile(rf"\b{escaped}\b", re.IGNORECASE)


def _iter_evidence_units(text: str) -> Iterable[str]:
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    if lines:
        yield from lines

        # PDF extraction can wrap one logical skill line across multiple physical lines,
        # e.g. "TanStack" on one line and "Query" on the next. Search small windows too.
        for index in range(len(lines) - 1):
            yield f"{lines[index]} {lines[index + 1]}"
        for index in range(len(lines) - 2):
            yield f"{lines[index]} {lines[index + 1]} {lines[index + 2]}"
        return

    yield from (part.strip() for part in re.split(r"(?<=[.!?])\s+", text or "") if part.strip())


def extract_skills(text: str) -> list[dict[str, str | None]]:
    """Extract known technical skills from resume text."""
    found: dict[str, dict[str, str | None]] = {}

    for evidence in _iter_evidence_units(text):
        for skill, metadata in SKILL_CATALOG.items():
            if skill in found:
                continue

            if any(_alias_pattern(alias).search(evidence) for alias in metadata["aliases"]):
                found[skill] = {
                    "name": skill,
                    "category": metadata["category"],
                    "evidence": evidence,
                }

    return list(found.values())
