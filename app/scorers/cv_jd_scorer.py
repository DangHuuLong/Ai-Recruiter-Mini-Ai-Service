from app.schemas.evaluation import (
    EvaluationCriterionScore,
    EvaluationInterviewQuestion,
    EvaluationResult,
    EvaluationSkillResult,
    ScoreApplicationRequest,
)


def score_application_mock(request: ScoreApplicationRequest) -> EvaluationResult:
    resume_skill_names = {
        skill.normalized_name or skill.name.lower()
        for skill in request.resume.skills
    }

    required_skills = request.job_description.required_skills
    preferred_skills = request.job_description.preferred_skills

    matched_skills: list[EvaluationSkillResult] = []
    missing_skills: list[EvaluationSkillResult] = []

    total_required = len(required_skills)

    for skill in required_skills:
        normalized_name = skill.normalized_name or skill.name.lower()

        if normalized_name in resume_skill_names:
            matched_skills.append(
                EvaluationSkillResult(
                    skill_name=skill.name,
                    normalized_skill_name=normalized_name,
                    type="MATCHED",
                    importance="HIGH",
                    evidence=f"{skill.name} found in resume skills",
                    note=None,
                )
            )
        else:
            missing_skills.append(
                EvaluationSkillResult(
                    skill_name=skill.name,
                    normalized_skill_name=normalized_name,
                    type="MISSING",
                    importance="HIGH",
                    evidence=None,
                    note="Required by the job description but not found in resume skills",
                )
            )

    for skill in preferred_skills:
        normalized_name = skill.normalized_name or skill.name.lower()

        if normalized_name in resume_skill_names:
            matched_skills.append(
                EvaluationSkillResult(
                    skill_name=skill.name,
                    normalized_skill_name=normalized_name,
                    type="MATCHED",
                    importance="MEDIUM",
                    evidence=f"{skill.name} found in resume skills",
                    note="Preferred skill matched",
                )
            )
        else:
            missing_skills.append(
                EvaluationSkillResult(
                    skill_name=skill.name,
                    normalized_skill_name=normalized_name,
                    type="MISSING",
                    importance="MEDIUM",
                    evidence=None,
                    note="Preferred by the job description but not found in resume skills",
                )
            )

    if total_required == 0:
        skills_score = 0.0
    else:
        required_matched_count = sum(
            1
            for skill in required_skills
            if (skill.normalized_name or skill.name.lower()) in resume_skill_names
        )
        skills_score = required_matched_count / total_required

    criteria_scores: list[EvaluationCriterionScore] = []

    for criterion in request.config.criteria:
        if criterion.criterion == "SKILLS_MATCH":
            score_normalized = skills_score
            reason = "Mock score based on required skill overlap."
            evidence = [skill.evidence for skill in matched_skills if skill.evidence]
        else:
            score_normalized = 0.6
            reason = "Mock default score for this criterion."
            evidence = []

        criteria_scores.append(
            EvaluationCriterionScore(
                criterion=criterion.criterion,
                weight=criterion.weight,
                score_normalized=score_normalized,
                reason=reason,
                evidence=evidence,
            )
        )

    overall_score = sum(
        item.score_normalized * item.weight * 100
        for item in criteria_scores
    )

    missing_skill_names = [skill.skill_name for skill in missing_skills]

    return EvaluationResult(
        overall_score=round(overall_score, 2),
        summary="Mock evaluation result for the application.",
        criteria=criteria_scores,
        skills=[*matched_skills, *missing_skills],
        explanation="This is a mock explanation based on simple skill matching.",
        skill_gap_summary=(
            f"Missing skills: {', '.join(missing_skill_names)}"
            if missing_skill_names
            else "No major skill gaps detected in mock scoring."
        ),
        interview_questions=[
            EvaluationInterviewQuestion(
                question="Can you describe a backend API project you have worked on?",
                category="backend",
                linked_skill="REST API",
                difficulty="MEDIUM",
                rationale="Backend API experience is relevant to this role.",
                display_order=1,
            ),
            EvaluationInterviewQuestion(
                question="How do you usually design database interactions in a backend service?",
                category="database",
                linked_skill="PostgreSQL",
                difficulty="MEDIUM",
                rationale="Database experience is important for this role.",
                display_order=2,
            ),
        ],
        evidence_map={
            "matched_skills": [
                skill.skill_name for skill in matched_skills
            ],
            "missing_skills": missing_skill_names,
        },
    )