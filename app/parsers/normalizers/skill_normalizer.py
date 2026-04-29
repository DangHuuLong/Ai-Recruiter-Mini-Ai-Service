import json
import re
from pathlib import Path

from app.parsers.normalizer import strip_accents


_MAP_PATH = Path(__file__).parent.parent / "data" / "skill_map.json"

DEFAULT_SKILL_MAP = {
    ".net": "dotnet",
    "amazon web services": "aws",
    "aws": "aws",
    "azure": "azure",
    "c#": "csharp",
    "c++": "cpp",
    "css": "css",
    "ci/cd": "ci_cd",
    "cicd": "ci_cd",
    "continuous integration": "ci_cd",
    "django": "django",
    "docker": "docker",
    "dotnet": "dotnet",
    "express": "express",
    "express.js": "express",
    "fast api": "fastapi",
    "fastapi": "fastapi",
    "flask": "flask",
    "git": "git",
    "github": "github",
    "go": "go",
    "golang": "go",
    "graph ql": "graphql",
    "graphql": "graphql",
    "html": "html",
    "java": "java",
    "javascript": "javascript",
    "jest": "jest",
    "js": "javascript",
    "k8s": "kubernetes",
    "kubernetes": "kubernetes",
    "laravel": "laravel",
    "mongo db": "mongodb",
    "mongodb": "mongodb",
    "mysql": "mysql",
    "nest.js": "nestjs",
    "nestjs": "nestjs",
    "next.js": "nextjs",
    "nextjs": "nextjs",
    "node": "nodejs",
    "node.js": "nodejs",
    "nodejs": "nodejs",
    "postgre sql": "postgresql",
    "postgres": "postgresql",
    "postgresql": "postgresql",
    "python": "python",
    "php": "php",
    "prisma": "prisma",
    "pytest": "pytest",
    "react": "react",
    "react.js": "react",
    "reactjs": "react",
    "redis": "redis",
    "ruby": "ruby",
    "spring boot": "spring_boot",
    "springboot": "spring_boot",
    "rest": "rest_api",
    "rest api": "rest_api",
    "restful api": "rest_api",
    "sql": "sql",
    "tailwind": "tailwind_css",
    "tailwind css": "tailwind_css",
    "ts": "typescript",
    "typescript": "typescript",
    "vue": "vue",
    "vue.js": "vue",
    "vuejs": "vue",
}


def _normalize_key(name: str) -> str:
    text = strip_accents(name or "").strip().lower()
    text = re.sub(r"[^a-z0-9+#.]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _load_map() -> dict[str, str]:
    skill_map = DEFAULT_SKILL_MAP.copy()
    try:
        with open(_MAP_PATH, "r", encoding="utf-8") as file:
            external_map = json.load(file)
            skill_map.update({_normalize_key(key): value for key, value in external_map.items()})
    except Exception:
        pass
    return skill_map


_SKILL_MAP = _load_map()


def normalize_skill(name: str) -> str:
    if not name:
        return ""

    key = _normalize_key(name)
    return _SKILL_MAP.get(key, key.replace(".", "").replace(" ", "_"))
