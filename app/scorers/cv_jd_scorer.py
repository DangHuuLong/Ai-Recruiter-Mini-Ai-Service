import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from app.schemas.evaluation import (
    EvaluationCriterionScore,
    EvaluationInterviewQuestion,
    EvaluationResult,
    EvaluationSkillResult,
    ScoreApplicationRequest,
)

WORD_RE = re.compile(r"[a-z0-9+#.]+")
DEFAULT_CRITERIA = [
    ("SKILLS_MATCH", 0.35),
    ("EXPERIENCE_RELEVANCE", 0.30),
    ("PROJECT_RELEVANCE", 0.15),
    ("EDUCATION_CERTIFICATION", 0.10),
    ("KEYWORD_DOMAIN_ALIGNMENT", 0.10),
]
DEFAULT_EVIDENCE = object()


@dataclass(frozen=True)
class SkillScoringResult:
    score: float
    matched: list[EvaluationSkillResult]
    missing: list[EvaluationSkillResult]
    related: list[EvaluationSkillResult]
    evidence: list[str]


def score_application_mock(request: ScoreApplicationRequest) -> EvaluationResult:
    """Deterministic MVP scorer used by /score/application.

    The backend persists the returned EvaluationResult into Evaluation,
    EvaluationCriterionScore, EvaluationSkill, evidence, and interview question
    tables. Keep this function stable, schema-first, and explainable.
    """

    skill_result = score_skill_overlap(request)
    keyword_result = score_keyword_overlap(request)
    project_result = score_project_relevance(request)
    experience_result = score_experience_relevance(request)
    education_result = score_education_match(request)

    scorer_by_criterion = {
        "SKILLS_MATCH": (skill_result.score, "Skill overlap score based on required/preferred JD skills.", skill_result.evidence),
        "EXPERIENCE_RELEVANCE": experience_result,
        "PROJECT_RELEVANCE": project_result,
        "EDUCATION_CERTIFICATION": education_result,
        "KEYWORD_DOMAIN_ALIGNMENT": keyword_result,
    }

    criteria_scores: list[EvaluationCriterionScore] = []
    for criterion, weight in configured_criteria(request):
        scorer = scorer_by_criterion.get(criterion)
        if scorer is None:
            score = 0.0
            reason = "Unsupported criterion for deterministic scoring baseline."
            evidence: list[str] = []
        else:
            score, reason, evidence = scorer

        criteria_scores.append(
            EvaluationCriterionScore(
                criterion=criterion,
                weight=clamp(weight),
                score_normalized=clamp(score),
                reason=reason,
                evidence=evidence[:8],
            )
        )

    overall_score = round(
        sum(item.score_normalized * item.weight * 100 for item in criteria_scores),
        2,
    )

    skills = [*skill_result.matched, *skill_result.related, *skill_result.missing]
    matched_skill_names = [skill.skill_name for skill in skill_result.matched]
    missing_skill_names = [skill.skill_name for skill in skill_result.missing]
    related_skill_names = [skill.skill_name for skill in skill_result.related]
    strong_points = build_strong_points(
        matched_skill_names,
        experience_result,
        project_result,
        education_result,
    )
    weak_points = build_weak_points(
        missing_skill_names,
        keyword_result,
        experience_result,
        project_result,
        education_result,
    )

    return EvaluationResult(
        overall_score=overall_score,
        summary=build_summary(overall_score, matched_skill_names, missing_skill_names),
        criteria=criteria_scores,
        skills=skills,
        explanation=build_explanation(criteria_scores, strong_points, weak_points),
        skill_gap_summary=build_skill_gap_summary(missing_skill_names, related_skill_names),
        interview_questions=build_interview_questions(
            missing_skill_names=missing_skill_names,
            matched_skill_names=matched_skill_names,
            weak_points=weak_points,
        ),
        evidence_map={
            "matched_skills": matched_skill_names,
            "missing_skills": missing_skill_names,
            "related_skills": related_skill_names,
            "strong_points": strong_points,
            "weak_points": weak_points,
            "criteria": {
                item.criterion: {
                    "score_normalized": item.score_normalized,
                    "weight": item.weight,
                    "evidence": item.evidence,
                }
                for item in criteria_scores
            },
            "interview_focus": missing_skill_names[:3] or weak_points[:3],
        },
    )


def score_skill_overlap(request: ScoreApplicationRequest) -> SkillScoringResult:
    resume_skills = {
        normalize_text(skill.normalized_name or skill.name): skill for skill in request.resume.skills
    }
    resume_tokens = set(tokens_from_text(resume_text(request.resume)))
    matched: list[EvaluationSkillResult] = []
    missing: list[EvaluationSkillResult] = []
    related: list[EvaluationSkillResult] = []
    evidence: list[str] = []

    required = list(request.job_description.required_skills)
    preferred = list(request.job_description.preferred_skills)

    earned = 0.0
    possible = required_skill_weight(required) + 0.5 * required_skill_weight(preferred)

    for skill in required:
        skill_weight = skill.weight_hint or (1.25 if skill.is_core else 1.0)
        result = classify_skill(skill.name, skill.normalized_name, resume_skills, resume_tokens)
        if result == "MATCHED":
            earned += skill_weight
            item = make_skill_result(skill.name, skill.normalized_name, "MATCHED", "HIGH")
            matched.append(item)
            evidence.append(item.evidence or "")
        elif result == "RELATED":
            earned += skill_weight * 0.45
            related.append(
                make_skill_result(
                    skill.name,
                    skill.normalized_name,
                    "PARTIAL",
                    "HIGH",
                    note="Related evidence found, but exact required skill is not explicit.",
                )
            )
        else:
            missing.append(
                make_skill_result(
                    skill.name,
                    skill.normalized_name,
                    "MISSING",
                    "HIGH",
                    evidence=None,
                    note="Required by the job description but not found in resume evidence.",
                )
            )

    for skill in preferred:
        skill_weight = 0.5 * (skill.weight_hint or (1.0 if skill.is_core else 0.75))
        result = classify_skill(skill.name, skill.normalized_name, resume_skills, resume_tokens)
        if result == "MATCHED":
            earned += skill_weight
            item = make_skill_result(
                skill.name,
                skill.normalized_name,
                "MATCHED",
                "MEDIUM",
                note="Preferred skill matched.",
            )
            matched.append(item)
            evidence.append(item.evidence or "")
        elif result == "RELATED":
            earned += skill_weight * 0.4
            related.append(
                make_skill_result(
                    skill.name,
                    skill.normalized_name,
                    "PARTIAL",
                    "MEDIUM",
                    note="Related preferred-skill evidence found.",
                )
            )
        else:
            missing.append(
                make_skill_result(
                    skill.name,
                    skill.normalized_name,
                    "MISSING",
                    "MEDIUM",
                    evidence=None,
                    note="Preferred by the job description but not found in resume evidence.",
                )
            )

    score = 0.0 if possible == 0 else earned / possible
    return SkillScoringResult(
        score=clamp(score),
        matched=matched,
        missing=missing,
        related=related,
        evidence=[item for item in evidence if item],
    )


def score_keyword_overlap(request: ScoreApplicationRequest) -> tuple[float, str, list[str]]:
    jd_keywords = set(normalize_text(item) for item in request.job_description.domain_keywords)
    jd_keywords.update(tokens_from_text(" ".join(request.job_description.requirements)))
    jd_keywords.update(tokens_from_text(" ".join(request.job_description.responsibilities)))
    resume_tokens = set(tokens_from_text(resume_text(request.resume)))

    stop_words = {"and", "or", "the", "with", "for", "to", "in", "of", "a", "an", "is", "are"}
    jd_keywords = {item for item in jd_keywords if len(item) > 2 and item not in stop_words}
    if not jd_keywords:
        return 0.0, "No domain keywords were available in the job description.", []

    matched = sorted(jd_keywords & resume_tokens)
    score = len(matched) / len(jd_keywords)
    return (
        clamp(score),
        "Keyword/domain alignment based on overlap between JD domain terms and resume text.",
        [f"Matched domain keyword: {item}" for item in matched[:8]],
    )


def score_project_relevance(request: ScoreApplicationRequest) -> tuple[float, str, list[str]]:
    jd_skill_names = job_skill_names(request)
    jd_tokens = set(tokens_from_text(job_text(request.job_description))) | jd_skill_names
    if not request.resume.projects:
        return 0.0, "No projects found in parsed resume.", []

    project_scores: list[float] = []
    evidence: list[str] = []
    for project in request.resume.projects:
        project_tokens = set(tokens_from_text(project.description or ""))
        project_tokens.update(normalize_text(item) for item in project.technologies)
        overlap = project_tokens & jd_tokens
        score = len(overlap) / max(1, min(len(jd_tokens), 12))
        project_scores.append(clamp(score))
        if overlap:
            evidence.append(
                f"Project {project.name or 'Unnamed project'} matches: {', '.join(sorted(overlap)[:5])}"
            )

    best_score = max(project_scores) if project_scores else 0.0
    avg_score = sum(project_scores) / len(project_scores) if project_scores else 0.0
    return (
        clamp(best_score * 0.7 + avg_score * 0.3),
        "Project relevance combines best project fit and average project overlap with JD skills/domain.",
        evidence[:8],
    )


def score_experience_relevance(request: ScoreApplicationRequest) -> tuple[float, str, list[str]]:
    total_months = sum(item.duration_months or 0 for item in request.resume.experience)
    min_years = request.job_description.min_experience_years
    years_score = 0.5
    if min_years is not None:
        required_months = max(1, min_years * 12)
        years_score = clamp(total_months / required_months)
    elif total_months > 0:
        years_score = 0.8

    jd_tokens = set(tokens_from_text(job_text(request.job_description))) | job_skill_names(request)
    evidence: list[str] = []
    overlap_scores: list[float] = []
    for exp in request.resume.experience:
        exp_tokens = set(tokens_from_text(" ".join(exp.responsibilities)))
        exp_tokens.update(normalize_text(item) for item in exp.technologies)
        role_tokens = set(tokens_from_text(exp.role or ""))
        overlap = (exp_tokens | role_tokens) & jd_tokens
        overlap_scores.append(len(overlap) / max(1, min(len(jd_tokens), 12)))
        if overlap:
            evidence.append(
                f"Experience {exp.role or 'role'} at {exp.company or 'company'} matches: {', '.join(sorted(overlap)[:5])}"
            )

    relevance_score = max(overlap_scores) if overlap_scores else 0.0
    score = years_score * 0.45 + clamp(relevance_score) * 0.55
    if total_months:
        evidence.insert(0, f"Parsed experience duration: {round(total_months / 12, 1)} years")
    return (
        clamp(score),
        "Experience relevance combines years of experience and overlap with JD responsibilities/skills.",
        evidence[:8],
    )


def score_education_match(request: ScoreApplicationRequest) -> tuple[float, str, list[str]]:
    requirement = request.job_description.education_requirement
    educations = request.resume.education
    certifications = request.resume.certifications
    evidence: list[str] = []

    if not requirement:
        score = 0.7 if (educations or certifications) else 0.5
        if educations:
            evidence.append("Education information is present and no strict JD requirement was specified.")
        if certifications:
            evidence.append("Certifications are present and may support role fit.")
        return score, "Education fit uses available education/certification evidence.", evidence

    requirement_tokens = set(tokens_from_text(requirement))
    education_text = " ".join(
        " ".join(filter(None, [edu.degree, edu.field_of_study, edu.institution, edu.description]))
        for edu in educations
    )
    certification_text = " ".join(
        " ".join(filter(None, [cert.name, cert.issuer])) for cert in certifications
    )
    candidate_tokens = set(tokens_from_text(f"{education_text} {certification_text}"))
    matched = sorted(requirement_tokens & candidate_tokens)
    score = len(matched) / max(1, len(requirement_tokens))
    if matched:
        evidence.append(f"Education/certification matched requirement terms: {', '.join(matched[:6])}")
    return (
        clamp(score),
        "Education/certification score is based on overlap with the JD education requirement.",
        evidence,
    )


def configured_criteria(request: ScoreApplicationRequest) -> list[tuple[str, float]]:
    if request.config.criteria:
        return [(item.criterion, item.weight) for item in request.config.criteria]
    return DEFAULT_CRITERIA


def classify_skill(name: str, normalized_name: str | None, resume_skills: dict, resume_tokens: set[str]) -> str:
    normalized = normalize_text(normalized_name or name)
    if normalized in resume_skills or normalized in resume_tokens:
        return "MATCHED"

    parts = set(tokens_from_text(normalized))
    if parts and parts <= resume_tokens:
        return "MATCHED"

    if parts and parts & resume_tokens:
        return "RELATED"

    return "MISSING"


def make_skill_result(
    name: str,
    normalized_name: str | None,
    match_type: str,
    importance: str,
    evidence: str | None | object = DEFAULT_EVIDENCE,
    note: str | None = None,
) -> EvaluationSkillResult:
    normalized = normalize_text(normalized_name or name)
    resolved_evidence = None if evidence is DEFAULT_EVIDENCE else evidence
    if resolved_evidence is None and match_type in {"MATCHED", "PARTIAL"}:
        resolved_evidence = f"{name} found in resume evidence"
    return EvaluationSkillResult(
        skill_name=name,
        normalized_skill_name=normalized,
        type=match_type,
        importance=importance,
        evidence=resolved_evidence if isinstance(resolved_evidence, str) else None,
        note=note,
    )


def build_summary(score: float, matched: list[str], missing: list[str]) -> str:
    if score >= 80:
        verdict = "Strong fit"
    elif score >= 60:
        verdict = "Potential fit"
    elif score >= 40:
        verdict = "Partial fit"
    else:
        verdict = "Weak fit"
    return (
        f"{verdict}: matched {len(matched)} skills and missing {len(missing)} skills. "
        f"Overall score: {score}."
    )


def build_explanation(
    criteria: list[EvaluationCriterionScore],
    strong_points: list[str],
    weak_points: list[str],
) -> str:
    criterion_text = "; ".join(
        f"{item.criterion}={round(item.score_normalized, 2)}" for item in criteria
    )
    return (
        f"Deterministic baseline scoring completed with criterion scores: {criterion_text}. "
        f"Strong points: {', '.join(strong_points) if strong_points else 'none detected'}. "
        f"Weak points: {', '.join(weak_points) if weak_points else 'none detected'}."
    )


def build_skill_gap_summary(missing: list[str], related: list[str]) -> str:
    if not missing and not related:
        return "No major skill gaps detected."
    parts = []
    if missing:
        parts.append(f"Missing skills: {', '.join(missing[:8])}")
    if related:
        parts.append(f"Related but not exact: {', '.join(related[:5])}")
    return "; ".join(parts)


def build_strong_points(
    matched: list[str],
    experience_result: tuple[float, str, list[str]],
    project_result: tuple[float, str, list[str]],
    education_result: tuple[float, str, list[str]],
) -> list[str]:
    points: list[str] = []
    if matched:
        points.append(f"Matches key skills: {', '.join(matched[:5])}")
    if experience_result[0] >= 0.7:
        points.append("Relevant work experience aligns with the job description")
    if project_result[0] >= 0.7:
        points.append("Projects show strong alignment with required technologies/domain")
    if education_result[0] >= 0.7:
        points.append("Education/certification evidence supports the role")
    return points[:6]


def build_weak_points(
    missing: list[str],
    keyword_result: tuple[float, str, list[str]],
    experience_result: tuple[float, str, list[str]],
    project_result: tuple[float, str, list[str]],
    education_result: tuple[float, str, list[str]],
) -> list[str]:
    points: list[str] = []
    if missing:
        points.append(f"Missing explicit evidence for: {', '.join(missing[:5])}")
    if keyword_result[0] < 0.45:
        points.append("Resume has limited keyword/domain overlap with the job description")
    if experience_result[0] < 0.45:
        points.append("Experience evidence is limited for the target role")
    if project_result[0] < 0.45:
        points.append("Project evidence is limited or not aligned with JD requirements")
    if education_result[0] < 0.45:
        points.append("Education/certification evidence does not clearly match the JD requirement")
    return points[:6]


def build_interview_questions(
    missing_skill_names: list[str],
    matched_skill_names: list[str],
    weak_points: list[str],
) -> list[EvaluationInterviewQuestion]:
    questions: list[EvaluationInterviewQuestion] = []
    for skill in missing_skill_names[:3]:
        questions.append(
            EvaluationInterviewQuestion(
                question=f"The resume does not show clear evidence of {skill}. Can you describe any hands-on experience with it?",
                category="skill_gap",
                linked_skill=skill,
                difficulty="MEDIUM",
                rationale="Validate whether a missing required/preferred skill exists but was not captured in the resume.",
                display_order=len(questions) + 1,
            )
        )
    for skill in matched_skill_names[:2]:
        questions.append(
            EvaluationInterviewQuestion(
                question=f"Can you walk through a real project or task where you used {skill}?",
                category="skill_depth",
                linked_skill=skill,
                difficulty="MEDIUM",
                rationale="Confirm depth behind a matched skill.",
                display_order=len(questions) + 1,
            )
        )
    for point in weak_points[:2]:
        questions.append(
            EvaluationInterviewQuestion(
                question=f"Can you provide more detail about this concern: {point}?",
                category="risk_probe",
                linked_skill=None,
                difficulty="HARD",
                rationale="Probe an evaluation weak point before making a hiring decision.",
                display_order=len(questions) + 1,
            )
        )
    if not questions:
        questions.append(
            EvaluationInterviewQuestion(
                question="Can you describe the most relevant project you have built for this role?",
                category="general_fit",
                linked_skill=None,
                difficulty="MEDIUM",
                rationale="Gather evidence for role relevance when no specific gap was detected.",
                display_order=1,
            )
        )
    return questions[:6]


def required_skill_weight(skills: Iterable[Any]) -> float:
    return sum((skill.weight_hint or (1.25 if skill.is_core else 1.0)) for skill in skills)


def job_skill_names(request: ScoreApplicationRequest) -> set[str]:
    return {
        normalize_text(skill.normalized_name or skill.name)
        for skill in [*request.job_description.required_skills, *request.job_description.preferred_skills]
    }


def job_text(job_description: Any) -> str:
    return " ".join(
        [
            job_description.title or "",
            job_description.seniority or "",
            " ".join(job_description.responsibilities),
            " ".join(job_description.requirements),
            " ".join(job_description.nice_to_have),
            " ".join(job_description.domain_keywords),
        ]
    )


def resume_text(resume: Any) -> str:
    sections = [resume.summary or ""]
    sections.extend(skill.name for skill in resume.skills)
    for exp in resume.experience:
        sections.extend([exp.company or "", exp.role or "", " ".join(exp.responsibilities), " ".join(exp.technologies)])
    for project in resume.projects:
        sections.extend([project.name or "", project.description or "", " ".join(project.technologies)])
    for edu in resume.education:
        sections.extend([edu.degree or "", edu.field_of_study or "", edu.description or ""])
    for cert in resume.certifications:
        sections.extend([cert.name or "", cert.issuer or ""])
    return " ".join(sections)


def tokens_from_text(value: str) -> list[str]:
    return [normalize_text(match.group(0)) for match in WORD_RE.finditer(value.lower())]


def normalize_text(value: str | None) -> str:
    return " ".join(WORD_RE.findall((value or "").lower())).strip()


def clamp(value: float) -> float:
    return min(max(value, 0.0), 1.0)
