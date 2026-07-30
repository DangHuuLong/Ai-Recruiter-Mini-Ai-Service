# Job Description Parser

This document describes the current Job Description parsing implementation in the AI Service.

## Purpose

The Job Description parser converts recruiter-provided raw job description text into a structured `ParsedJobDescriptionData` object.

The parser is used by the AI Service endpoint:

```text
POST /parse/job-description
```

The Backend calls this endpoint when parsing a job description through:

```text
POST /api/job-descriptions/:id/parse
```

The Backend then stores the returned structured data in `JobDescription.parsedData` and syncs parsed skills into the `JobSkill` table.

## Output Schema

The parser returns data that follows the `ParsedJobDescriptionData` schema.

Main fields:

```text
title
seniority
employment_type
responsibilities
requirements
nice_to_have
required_skills
preferred_skills
min_experience_years
education_requirement
domain_keywords
```

Skill fields follow the `JobSkill` schema:

```text
name
normalized_name
is_core
weight_hint
```

## Current Implementation

The parser is implemented as a deterministic rule-based parser in:

```text
app/parsers/job_description_parser.py
```

The main entrypoint is:

```python
def parse_job_description(raw_text: str) -> ParsedJobDescriptionData:
```

The parser does not call an LLM. It uses text normalization, section detection, regex patterns, and a predefined skill catalog.

This makes the output predictable and easy to test during the MVP stage.

## Parsing Pipeline

The parser follows this flow:

```text
raw_text
↓
normalize text
↓
split text into known sections
↓
extract responsibilities, requirements, and nice-to-have items
↓
extract required skills from requirements
↓
extract preferred skills from nice-to-have
↓
detect title, seniority, employment type, minimum experience, education, and domain keywords
↓
return ParsedJobDescriptionData
```

## Text Normalization

The parser first cleans the input text by:

- converting Windows line endings to `\n`
- removing null bytes
- trimming repeated spaces and tabs
- removing empty lines

For matching, the parser also normalizes text by:

- lowercasing text
- converting Vietnamese `đ` to `d`
- removing accents with `unicodedata`
- preserving useful technical characters such as `#`, `+`, `.`, `/`, and `-`
- collapsing repeated whitespace

This allows the parser to match both English and Vietnamese headings more reliably.

## Section Detection

The parser recognizes common section headings and maps them to canonical section names.

Supported canonical sections:

```text
responsibilities
requirements
nice_to_have
benefits
```

Examples of recognized headings:

```text
Responsibilities
Key Responsibilities
What you will do
Requirements
Must have
Qualifications
Nice to have
Preferred skills
Benefits
Trách nhiệm
Yêu cầu
Ưu tiên
Phúc lợi
```

The parser stores lines under the current section until another recognized section heading appears.

## Responsibilities, Requirements, and Nice-to-have

After sections are detected, the parser extracts list items from each section.

It removes common bullet prefixes such as:

```text
-
*
+
•
1.
1)
```

It also removes duplicate items while preserving the original order.

Example input:

```text
Responsibilities:
- Build REST API services.
- Collaborate with frontend developers.
```

Example output:

```json
{
  "responsibilities": [
    "Build REST API services",
    "Collaborate with frontend developers"
  ]
}
```

## Skill Extraction

Skills are extracted from a predefined catalog.

The catalog is defined in:

```text
SKILL_CATALOG
```

Each skill has:

```text
name
normalized_name
aliases
category
```

Example:

```python
SkillDefinition("NestJS", "nestjs", ("nestjs", "nest.js", "nest js"), "backend")
```

The parser currently supports common MVP skills, including:

```text
Python
JavaScript
TypeScript
Java
C#
PHP
Go
FastAPI
Django
Flask
Spring Boot
Node.js
NestJS
Express.js
React
Next.js
Vue.js
Angular
HTML
CSS
Tailwind CSS
PostgreSQL
MySQL
SQL Server
MongoDB
Redis
REST API
GraphQL
Docker
Kubernetes
AWS
Azure
GCP
Git
CI/CD
Prisma
Supabase
Pytest
Jest
```

## Required and Preferred Skills

Required skills are extracted primarily from the `requirements` section.

Preferred skills are extracted from the `nice_to_have` section.

If no required skills are found from `requirements`, the parser falls back to scanning the full raw text.

Required skills are returned with:

```json
{
  "is_core": true,
  "weight_hint": 1.0
}
```

Preferred skills are returned with:

```json
{
  "is_core": false,
  "weight_hint": 0.6
}
```

## Skill Name Normalization

Each extracted skill includes a normalized name.

Examples:

```text
Node.js     -> nodejs
NestJS      -> nestjs
REST API    -> rest_api
CI/CD       -> ci_cd
PostgreSQL  -> postgresql
Tailwind CSS -> tailwind_css
```

The normalized name is used by the Backend for `JobSkill.normalizedName` and later matching/scoring workflows.

## Short Alias Handling

Some aliases are short and easy to match incorrectly.

Examples:

```text
js
ts
go
```

The parser treats these as short aliases and applies stricter regex boundaries.

This prevents false positives such as:

```text
Node.js  -> should not create JavaScript just because it contains js
NestJS   -> should not create JavaScript just because it ends with JS
```

This behavior is covered by tests.

## Hyphenated Skill Handling

The skill matcher supports skills followed by punctuation or hyphens.

Example:

```text
Docker-based local development
```

Expected output:

```json
{
  "name": "Docker",
  "normalized_name": "docker"
}
```

This prevents the parser from missing skills embedded in phrases like `Docker-based`.

## Title Detection

The parser detects job title using three strategies:

1. Labeled title lines
2. Natural-language hiring phrases
3. Fallback scan of the first few lines

Examples:

```text
Job Title: Senior Backend Developer
Position: Frontend Engineer
Vị trí: Backend Developer
We are looking for a Backend Developer with NestJS...
```

Output example:

```json
{
  "title": "Senior Backend Developer"
}
```

## Seniority Detection

Seniority detection prefers explicit keywords before using years of experience.

Explicit keyword examples:

```text
Intern
Fresher
Junior
Mid-level
Senior
Lead
Principal
Staff
Architect
Manager
```

Fallback by years:

```text
5+ years -> senior
2-3 years -> mid
0-1 year -> junior
```

This prevents a JD such as `Senior Backend Developer` with `3 years experience` from being incorrectly classified as `mid`.

## Employment Type Detection

The parser detects employment type from English and Vietnamese terms.

Examples:

```text
Full-time -> full-time
Part-time -> part-time
Contract -> contract
Internship -> internship
toàn thời gian -> full-time
bán thời gian -> part-time
hợp đồng -> contract
thực tập -> internship
```

## Minimum Experience Detection

The parser detects minimum experience years from patterns such as:

```text
At least 4 years of experience
Minimum 3 years
3+ years experience
Tối thiểu 2 năm kinh nghiệm
```

If multiple year values are found, the parser returns the smallest value as the minimum requirement.

Example:

```text
At least 4 years experience. 5+ years preferred.
```

Output:

```json
{
  "min_experience_years": 4
}
```

## Education Requirement Detection

The parser detects common education requirements, such as:

```text
Bachelor's degree
Master's degree
PhD
Computer Science
Software Engineering
Information Technology
CNTT
Đại học
Cao đẳng
Cử nhân
Kỹ sư
```

Example output:

```json
{
  "education_requirement": "Bachelor's degree in Computer Science, Software Engineering, Information Technology, or a related field"
}
```

## Domain Keyword Extraction

The parser extracts domain keywords from known aliases.

Supported domain keywords include:

```text
backend
frontend
fullstack
mobile
data
ai
devops
qa
ecommerce
finance
recruitment
rest api
```

The `data` domain is intentionally strict. It is only detected from stronger signals such as:

```text
data engineering
data platform
data pipeline
ETL
warehouse
analytics
BI
```

This avoids false positives from generic phrases such as `structured recruiting data`.

## Backend Integration

The Backend calls the AI Service endpoint:

```text
POST /parse/job-description
```

through its AI integration layer.

After a successful parse, the Backend:

1. stores the response in `JobDescription.parsedData`
2. sets `parseStatus = SUCCESS`
3. sets `parserVersion = ai-job-description-parser-v1`
4. clears `parsingError`
5. syncs `required_skills` and `preferred_skills` into the `JobSkill` table

The `JobSkill` table is the source of truth for matching and scoring.

## Example Input

```text
Job Title: Senior Backend Developer
Employment Type: Full-time
Location: Ho Chi Minh City

Responsibilities:
- Design, build, and maintain scalable backend services for an AI recruiting platform.
- Develop REST API endpoints for candidates, resumes, job descriptions, applications, and evaluations.

Requirements:
- At least 4 years of backend development experience.
- Strong experience with Node.js, NestJS, TypeScript, REST API, PostgreSQL, and Redis.
- Good understanding of Docker-based local development and CI/CD pipelines.
- Bachelor's degree in Computer Science, Software Engineering, Information Technology, or a related field.

Nice to have:
- Experience with AWS, Kubernetes, or cloud deployment.
- Experience with Python or FastAPI is a plus.
- Experience writing automated tests with Jest or Pytest.
```

## Example Output

```json
{
  "title": "Senior Backend Developer",
  "seniority": "senior",
  "employment_type": "full-time",
  "responsibilities": [
    "Design, build, and maintain scalable backend services for an AI recruiting platform",
    "Develop REST API endpoints for candidates, resumes, job descriptions, applications, and evaluations"
  ],
  "requirements": [
    "At least 4 years of backend development experience",
    "Strong experience with Node.js, NestJS, TypeScript, REST API, PostgreSQL, and Redis",
    "Good understanding of Docker-based local development and CI/CD pipelines",
    "Bachelor's degree in Computer Science, Software Engineering, Information Technology, or a related field"
  ],
  "nice_to_have": [
    "Experience with AWS, Kubernetes, or cloud deployment",
    "Experience with Python or FastAPI is a plus",
    "Experience writing automated tests with Jest or Pytest"
  ],
  "required_skills": [
    { "name": "TypeScript", "normalized_name": "typescript", "is_core": true, "weight_hint": 1.0 },
    { "name": "Node.js", "normalized_name": "nodejs", "is_core": true, "weight_hint": 1.0 },
    { "name": "NestJS", "normalized_name": "nestjs", "is_core": true, "weight_hint": 1.0 },
    { "name": "PostgreSQL", "normalized_name": "postgresql", "is_core": true, "weight_hint": 1.0 },
    { "name": "Redis", "normalized_name": "redis", "is_core": true, "weight_hint": 1.0 },
    { "name": "REST API", "normalized_name": "rest_api", "is_core": true, "weight_hint": 1.0 },
    { "name": "Docker", "normalized_name": "docker", "is_core": true, "weight_hint": 1.0 },
    { "name": "CI/CD", "normalized_name": "ci_cd", "is_core": true, "weight_hint": 1.0 }
  ],
  "preferred_skills": [
    { "name": "Python", "normalized_name": "python", "is_core": false, "weight_hint": 0.6 },
    { "name": "FastAPI", "normalized_name": "fastapi", "is_core": false, "weight_hint": 0.6 },
    { "name": "Kubernetes", "normalized_name": "kubernetes", "is_core": false, "weight_hint": 0.6 },
    { "name": "AWS", "normalized_name": "aws", "is_core": false, "weight_hint": 0.6 },
    { "name": "Pytest", "normalized_name": "pytest", "is_core": false, "weight_hint": 0.6 },
    { "name": "Jest", "normalized_name": "jest", "is_core": false, "weight_hint": 0.6 }
  ],
  "min_experience_years": 4,
  "education_requirement": "Bachelor's degree in Computer Science, Software Engineering, Information Technology, or a related field",
  "domain_keywords": ["backend", "ai", "devops", "recruitment", "rest api"]
}
```

## Tests

Current tests are in:

```text
tests/test_parse_job_description.py
```

Coverage includes:

- endpoint response shape
- section extraction
- responsibilities extraction
- requirements extraction
- nice-to-have extraction
- required skill extraction
- preferred skill extraction
- skill name normalization
- Vietnamese heading support
- seniority detection
- employment type detection
- minimum experience year detection
- education requirement detection
- domain keyword extraction
- hyphenated skill matching such as `Docker-based`
- short alias false-positive prevention such as `Node.js` and `NestJS` not creating `JavaScript`
- generic `structured data` not creating the `data` domain keyword

Run tests:

```bash
python -m pytest
```

## Current Limitations

The parser is rule-based, so it only detects skills and patterns currently defined in the code.

Known limitations:

- New skills must be added to `SKILL_CATALOG` before they can be extracted.
- Very unusual JD formats may not split sections correctly.
- Domain keyword extraction is heuristic and may include contextual domains.
- The parser does not currently distinguish primary domain keywords from contextual keywords.
- It does not perform semantic reasoning like an LLM.

## Future Improvements

Possible improvements:

- Add more skills and aliases to the skill catalog.
- Split domain keywords into `primary_domain_keywords` and `context_keywords` if the schema is expanded.
- Add confidence scores for extracted fields.
- Improve Vietnamese JD pattern coverage.
- Add support for more seniority terms and employment types.
- Consider a hybrid parser where deterministic rules handle stable fields and an LLM handles ambiguous text.
