# CV-JD Labeling Guide

## 1. Purpose

This guide defines how CV-JD pairs should be labeled for the AI Recruiter Mini dataset.

The goal is to create consistent labels before model training or model comparison starts. A small clean dataset is preferred over a large noisy dataset.

## 2. Labeling Input

Each labeling task should include:

1. one resume record;
2. one job description record;
3. parsed resume fields when available;
4. parsed JD fields when available;
5. raw text for both documents.

Labelers should read the parsed fields first, then check raw text when the parsed data looks incomplete or suspicious.

## 3. Scoring Scale

Use a 0-100 score for each criterion and for the final overall score.

| Range | Meaning |
| --- | --- |
| 90-100 | Excellent fit |
| 75-89 | Strong fit |
| 60-74 | Moderate fit |
| 40-59 | Weak fit |
| 0-39 | Poor fit |

## 4. Rubric Versions

### `rubric_v0.1` legacy dataset rubric

The initial dataset docs used these exploratory criteria:

| Criterion | Purpose |
| --- | --- |
| `skill_match` | Required/preferred skill coverage |
| `experience_match` | Years, role level, and practical evidence |
| `education_match` | Degree and education requirement fit |
| `domain_relevance` | Similar domain or project context |
| `nice_to_have` | Optional skills or bonus qualifications |

This version is kept only for historical compatibility with `datasets/versions/v0.1`.

### `rubric_v0.2` current scorer-aligned rubric

The current working dataset should use `rubric_v0.2`.

This version is aligned with the deterministic scoring baseline in `app/scorers/cv_jd_scorer.py`.

| Criterion | Weight | What to check |
| --- | ---: | --- |
| `SKILLS_MATCH` | 35% | Required/preferred JD skill coverage and evidence |
| `EXPERIENCE_RELEVANCE` | 30% | Experience depth, years, role level, and responsibility fit |
| `PROJECT_RELEVANCE` | 15% | Project evidence related to JD responsibilities and stack |
| `EDUCATION_CERTIFICATION` | 10% | Degree, field of study, certifications, or equivalent background |
| `KEYWORD_DOMAIN_ALIGNMENT` | 10% | Domain keywords, responsibilities, and contextual alignment |

Overall score formula:

```txt
overall_score =
  SKILLS_MATCH * 0.35 +
  EXPERIENCE_RELEVANCE * 0.30 +
  PROJECT_RELEVANCE * 0.15 +
  EDUCATION_CERTIFICATION * 0.10 +
  KEYWORD_DOMAIN_ALIGNMENT * 0.10
```

Round the final score to the nearest integer.

## 5. Level-Aware Scoring Rules

Adjust expectations based on the JD level. Do not use the same experience standard for intern and senior roles.

| Job Level | Experience Expectation | Education Expectation |
| --- | --- | --- |
| `intern` | Academic projects, personal projects, and small freelance work are acceptable evidence | Current student, fresh graduate, or related self-study is acceptable |
| `junior` | 0-2 years or several real projects with clear ownership | Related degree preferred but not always blocking |
| `middle` | 2-4 years or strong equivalent project/work evidence | Related degree or strong equivalent background expected |
| `senior` | 4+ years, leadership, architecture, mentoring, or system design evidence | Degree less important than proven senior-level delivery |
| `unknown` | Use the JD wording and responsibilities to infer expectation conservatively | Use explicit JD requirement only |

A senior candidate applying for an intern role should not be penalized in skill or experience match only because of overqualification. If overqualification may affect hiring fit, mention it in `label_notes` instead of silently lowering the score.

## 6. Criterion Scoring Guidance

### `SKILLS_MATCH`

Evaluate required and preferred skill coverage.

| Score | Rule |
| --- | --- |
| 90-100 | Covers almost all required skills with clear evidence |
| 75-89 | Covers most required skills, only minor gaps |
| 60-74 | Covers some required skills, but misses important items |
| 40-59 | Has broad overlap but misses core stack |
| 0-39 | Little or no overlap with required skills |

Do not give high skill scores for keyword mentions without evidence. Prefer skills that appear in projects, work experience, or skill sections.

### `EXPERIENCE_RELEVANCE`

Evaluate whether the candidate's experience fits the JD level and expected responsibility depth.

For intern roles, academic projects and personal projects can count as relevant experience.

| Score | Rule |
| --- | --- |
| 90-100 | Experience clearly exceeds role expectations |
| 75-89 | Experience fits the role well |
| 60-74 | Experience is related but shallow or incomplete |
| 40-59 | Limited practical evidence |
| 0-39 | No relevant experience or projects |

### `PROJECT_RELEVANCE`

Evaluate whether projects demonstrate the same stack, problem type, or responsibilities as the JD.

Examples:

- backend API JD and a FastAPI/PostgreSQL project: high;
- frontend SaaS dashboard JD and a Next.js dashboard project: high;
- DevOps JD and only frontend UI projects: low;
- AI data JD and dataset/evaluation project: high.

### `EDUCATION_CERTIFICATION`

Evaluate degree, field, certificates, or equivalent background only when it is relevant to the JD.

| Score | Rule |
| --- | --- |
| 90-100 | Directly matches required degree and field |
| 75-89 | Related degree or strong equivalent background |
| 60-74 | Partially related education |
| 40-59 | Education is unclear or weakly related |
| 0-39 | Does not meet explicit education requirement |

If the JD does not mention education, use a neutral score around 70 unless education is clearly relevant or clearly mismatched.

### `KEYWORD_DOMAIN_ALIGNMENT`

Evaluate overlap in responsibilities, domain keywords, and business/technical context.

Examples:

- recruitment platform JD and recruitment/CV parsing project: high;
- backend API JD and generic CRUD API project: moderate to high;
- AI scoring JD and no data/AI project evidence: lower.

## 7. Label Class Mapping

After computing `overall_score`, map it to a label:

| Score Range | Label |
| --- | --- |
| 90-100 | `excellent_match` |
| 75-89 | `strong_match` |
| 60-74 | `moderate_match` |
| 40-59 | `weak_match` |
| 0-39 | `poor_match` |

Do not manually override the label unless the rubric has a clear issue. If an override is needed, document the reason in `label_notes`.

## 8. Required Label Notes

Each pair must include a short explanation.

Good notes:

```txt
Strong backend match for an intern role. Candidate covers Python, FastAPI, and PostgreSQL through projects, but does not show Docker or AWS evidence.
```

Bad notes:

```txt
Good candidate.
```

A good note should mention:

- strongest match reason;
- most important missing requirement;
- why the final label makes sense.

## 9. Match Decision Rules

### Strong Match

Use `strong_match` when the candidate covers most core requirements and gaps are acceptable for the job level.

### Moderate Match

Use `moderate_match` when the candidate has useful overlap but would need training or screening before shortlisting.

### Weak Match

Use `weak_match` when there is some relationship but core requirements are missing.

### Poor Match

Use `poor_match` when the resume and JD mostly target different roles, stacks, or seniority levels.

## 10. Borderline Examples

### Case A: Overqualified Candidate

A senior backend engineer applies for a backend intern role.

Suggested approach:

- `SKILLS_MATCH`: high, if required skills are clearly covered.
- `EXPERIENCE_RELEVANCE`: high, because experience exceeds the JD expectation.
- `PROJECT_RELEVANCE`: depends on project/domain overlap.
- `KEYWORD_DOMAIN_ALIGNMENT`: depends on responsibility/domain overlap.
- `label_notes`: mention overqualification as a hiring consideration, but do not lower technical fit only because the candidate is senior.

### Case B: Keyword Stuffing

A resume lists many technologies in one skill line but has no project or experience evidence.

Suggested approach:

- `SKILLS_MATCH`: moderate at most, unless raw text provides evidence elsewhere.
- `EXPERIENCE_RELEVANCE`: low to moderate depending on actual projects/work history.
- `PROJECT_RELEVANCE`: low if projects do not support listed skills.
- `label_notes`: mention that skill evidence is weak.

### Case C: Career Switcher

A frontend candidate applies for a backend role and has one small backend project.

Suggested approach:

- `SKILLS_MATCH`: based on required backend skills actually demonstrated.
- `EXPERIENCE_RELEVANCE`: moderate or weak depending on project depth.
- `PROJECT_RELEVANCE`: can be moderate if the project solves similar problems.
- `label_notes`: mention frontend strength and backend gap separately.

## 11. Consistency Rules

- Use the same rubric for all pairs in the same working dataset version.
- Keep legacy pairs with their original `label_version`.
- Do not punish intern candidates for not having senior-level production experience.
- Do not reward unrelated skills only because they are technical.
- Prefer evidence from projects or experience over plain skill lists.
- Keep labels stable once a dataset version is used for evaluation.
- If two labelers assign overall scores that differ by more than 15 points on the same pair, escalate to a third reviewer before accepting either label.

## 12. Quality Review Checklist

Before accepting a label:

- [ ] `overall_score` matches the criterion scores and rubric weights.
- [ ] `label` matches the score range.
- [ ] `label_version` matches the criterion key set.
- [ ] Required skills were checked against evidence.
- [ ] Missing required skills are listed.
- [ ] Job level was considered when scoring experience.
- [ ] Borderline cases are explained in `label_notes`.
- [ ] Label notes explain the decision.
- [ ] No private candidate information is included in notes.

## 13. Phase Boundary

This guide only defines labeling rules. Fine-tuning and model export should be implemented in later branches after the dataset contract is stable.
