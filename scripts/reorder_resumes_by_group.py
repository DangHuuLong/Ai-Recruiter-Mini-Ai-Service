import json
import shutil
import argparse
from pathlib import Path
from collections import defaultdict


ROLE_PRIORITY = [
    "frontend_web",
    "fullstack_js",
    "backend_nodejs",
    "backend_java_spring",
    "php_laravel_web",
    "mobile",
    "devops_cloud",
    "data_engineer",
    "ai_ml_data_science",
    "it_helpdesk_support",
    "qa_tester",
    "security_operation",
    "it_ba_pm_manager",
    "general_software",
    "low_information",
    "unknown_role",
]


ROLE_KEYWORDS = {
    "frontend_web": [
        "frontend", "front-end", "react", "reactjs", "vue", "vuejs",
        "angular", "angularjs", "html", "css", "javascript",
        "typescript", "tailwind", "responsive", "ui/ux",
        "cross-browser", "psd to html", "figma"
    ],
    "fullstack_js": [
        "fullstack", "full stack", "react", "node.js", "nodejs",
        "express", "express.js", "mongodb", "mysql", "typescript",
        "restful api", "jwt", "next.js"
    ],
    "backend_nodejs": [
        "backend", "back-end", "node.js", "nodejs", "express",
        "express.js", "nestjs", "mongodb", "postgresql", "mysql",
        "redis", "jwt", "oauth2", "restful api", "graphql",
        "payment integration"
    ],
    "backend_java_spring": [
        "java", "spring boot", "spring framework", "hibernate",
        "jpa", "maven", "junit", "microservices", "servlet",
        "jsp"
    ],
    "php_laravel_web": [
        "php", "laravel", "wordpress", "jquery", "mysql",
        "vuejs", "slim framework", "zend", "php oop"
    ],
    "mobile": [
        "android", "flutter", "react native", "kotlin", "java",
        "android studio", "app store", "google play", "mobile app",
        "ionic capacitor"
    ],
    "devops_cloud": [
        "devops", "docker", "kubernetes", "aws", "gcp", "azure",
        "terraform", "ansible", "jenkins", "gitlab ci",
        "github actions", "prometheus", "grafana", "helm",
        "ci/cd", "infrastructure as code"
    ],
    "data_engineer": [
        "data engineer", "etl", "elt", "airflow", "spark",
        "pyspark", "flink", "kafka", "dbt", "bigquery",
        "redshift", "snowflake", "data lake", "lakehouse",
        "apache iceberg", "great expectations"
    ],
    "ai_ml_data_science": [
        "ai engineer", "machine learning", "deep learning",
        "data science", "tensorflow", "pytorch", "scikit-learn",
        "computer vision", "reinforcement learning", "cnn",
        "rnn", "transformer", "predictive modeling",
        "regression analysis"
    ],
    "it_helpdesk_support": [
        "helpdesk", "technical support", "user support",
        "hardware troubleshooting", "software troubleshooting",
        "network troubleshooting", "lan", "wan", "office 365",
        "printer setup", "domain controller", "file server"
    ],
    "qa_tester": [
        "tester", "testing", "test case", "test report",
        "manual testing", "automation testing", "selenium",
        "functional testing", "gui testing", "quality assurance"
    ],
    "security_operation": [
        "security operation", "security operations", "siem",
        "incident investigation", "digital forensics", "fireeye",
        "mcafee", "risk analysis", "information security"
    ],
    "it_ba_pm_manager": [
        "business analyst", "project manager", "product manager",
        "it manager", "stakeholder", "backlog", "user stories",
        "agile", "scrum", "budget", "vendor management",
        "project management"
    ],
    "general_software": [
        "software development", "software engineering", "programming",
        "c++", "c#", ".net", "ruby", "python", "java",
        "software architecture", "debugging", "database"
    ],
}


SENIORITY_PRIORITY = {
    "intern": 0,
    "junior": 1,
    "middle": 2,
    "senior": 3,
    "unknown": 4,
    None: 4,
}


def load_jsonl(path: Path):
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)
            item["_line_no"] = line_no
            rows.append(item)

    return rows


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            row.pop("_line_no", None)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def resume_number(resume_id: str):
    try:
        return int(resume_id.replace("resume_", ""))
    except Exception:
        return None


def build_search_text(resume: dict) -> str:
    parts = []

    for field in ["summary", "raw_text", "candidate_level"]:
        value = resume.get(field)
        if value:
            parts.append(str(value))

    for field in ["skills", "normalized_skills", "certifications", "languages"]:
        value = resume.get(field)
        if isinstance(value, list):
            parts.extend(str(x) for x in value)

    projects = resume.get("projects") or []
    for project in projects:
        if isinstance(project, dict):
            parts.append(str(project.get("name") or ""))
            parts.append(str(project.get("description") or ""))

            technologies = project.get("technologies") or []
            parts.extend(str(x) for x in technologies)

    return " ".join(parts).lower()


def count_matches(text: str, keywords: list[str]) -> int:
    return sum(1 for keyword in keywords if keyword.lower() in text)


def classify_role(resume: dict):
    text = build_search_text(resume)

    if is_low_information_resume(resume):
        return "low_information", 0

    scores = {}

    for role, keywords in ROLE_KEYWORDS.items():
        scores[role] = count_matches(text, keywords)

    best_role = max(scores, key=scores.get)
    best_score = scores[best_role]

    if best_score == 0:
        return "unknown_role", 0

    return best_role, best_score


def is_low_information_resume(resume: dict) -> bool:
    skills = resume.get("normalized_skills") or resume.get("skills") or []
    projects = resume.get("projects") or []
    summary = resume.get("summary") or ""

    if len(skills) <= 8 and len(projects) == 0:
        return True

    if len(summary.strip()) < 80 and len(skills) <= 10:
        return True

    return False


def normalize_seniority(resume: dict) -> str:
    level = resume.get("candidate_level")

    if level in {"intern", "junior", "middle", "senior"}:
        return level

    years = resume.get("experience_years")

    if isinstance(years, (int, float)):
        if years < 1:
            return "intern"
        if years < 3:
            return "junior"
        if years < 6:
            return "middle"
        return "senior"

    return "unknown"


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default="datasets/raw/resumes.jsonl",
        help="Input resumes.jsonl path",
    )

    parser.add_argument(
        "--output",
        default="datasets/raw/resumes_reordered.jsonl",
        help="Output path. Use same as input if you want overwrite.",
    )

    parser.add_argument(
        "--start-id",
        type=int,
        default=187,
        help="Start resume number to reorder and re-id",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite input file safely after creating backup",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    rows = load_jsonl(input_path)

    fixed_rows = []
    target_rows = []

    for row in rows:
        number = resume_number(row.get("id", ""))

        if number is not None and number >= args.start_id:
            target_rows.append(row)
        else:
            fixed_rows.append(row)

    enriched_targets = []

    for row in target_rows:
        role_group, role_score = classify_role(row)
        seniority_group = normalize_seniority(row)

        row["role_group"] = role_group
        row["seniority_group"] = seniority_group
        row["grouping_score"] = role_score

        enriched_targets.append(row)

    role_rank = {role: index for index, role in enumerate(ROLE_PRIORITY)}

    enriched_targets.sort(
        key=lambda r: (
            role_rank.get(r["role_group"], 999),
            SENIORITY_PRIORITY.get(r["seniority_group"], 999),
            -int(r.get("grouping_score") or 0),
            -(len(r.get("normalized_skills") or r.get("skills") or [])),
            -(r.get("experience_years") or 0),
            r.get("id", ""),
        )
    )

    id_mapping = {}

    next_id = args.start_id

    for row in enriched_targets:
        old_id = row["id"]
        new_id = f"resume_{next_id:03d}"

        id_mapping[old_id] = new_id
        row["old_id"] = old_id
        row["id"] = new_id

        next_id += 1

    output_rows = fixed_rows + enriched_targets

    if args.overwrite:
        backup_path = input_path.with_suffix(".jsonl.bak")
        shutil.copy2(input_path, backup_path)
        write_jsonl(input_path, output_rows)

        mapping_path = input_path.parent / "resume_id_mapping_from_187.json"
        with mapping_path.open("w", encoding="utf-8") as f:
            json.dump(id_mapping, f, ensure_ascii=False, indent=2)

        final_output = input_path
        print(f"Backup written: {backup_path}")
        print(f"ID mapping written: {mapping_path}")
    else:
        write_jsonl(output_path, output_rows)

        mapping_path = output_path.parent / "resume_id_mapping_from_187.json"
        with mapping_path.open("w", encoding="utf-8") as f:
            json.dump(id_mapping, f, ensure_ascii=False, indent=2)

        final_output = output_path
        print(f"ID mapping written: {mapping_path}")

    print(f"Input resumes: {len(rows)}")
    print(f"Fixed resumes before resume_{args.start_id:03d}: {len(fixed_rows)}")
    print(f"Reordered resumes from resume_{args.start_id:03d}: {len(enriched_targets)}")
    print(f"Output written: {final_output}")

    print()
    print("Group summary:")

    summary = defaultdict(int)

    for row in enriched_targets:
        summary[(row["role_group"], row["seniority_group"])] += 1

    for (role_group, seniority_group), count in sorted(
        summary.items(),
        key=lambda x: (
            role_rank.get(x[0][0], 999),
            SENIORITY_PRIORITY.get(x[0][1], 999),
            x[0][0],
        ),
    ):
        print(f"  {role_group:24s} {seniority_group:8s} {count}")


if __name__ == "__main__":
    main()