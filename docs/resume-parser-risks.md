# Resume Parser Weaknesses and Risks

This document summarizes the main weaknesses and risks in the current raw-text resume parsing approach.

## Context

The current parser is a rule-based MVP parser. It extracts structured resume data from `raw_text` by normalizing text, splitting sections, and applying dedicated extractors for personal information, skills, education, experience, projects, certifications, achievements, and languages.

This approach is reasonable for an MVP because it is deterministic, easy to test, and easier to debug than an AI-first parser. However, it also has clear limitations that should be tracked before the parser is treated as production-grade.

## 1. Heuristic logic is growing in complexity

The parser relies heavily on keyword lists, regular expressions, and helper functions such as `looks_like_*`, `is_*`, and section-specific detection rules.

### Risk

As more CV formats are added, the rule set can grow quickly and become difficult to maintain. A fix for one CV layout may accidentally break another layout.

### Impact

- Higher chance of false positives and false negatives.
- Harder debugging when multiple heuristics overlap.
- Increasing maintenance cost as more CV formats are supported.
- More regression tests are required for every parser change.

### Recommendation

Group rules by responsibility, keep heuristic rules documented, and expand regression tests using real CV samples before adding more special-case logic.

## 2. Strong dependency on raw text quality

The parser assumes that `raw_text` is already readable and reasonably ordered. This works well for simple text-based PDF/DOCX resumes, but it is fragile when the extracted text has poor layout quality.

### Risk

If the file reader extracts text in the wrong order, loses columns, drops headings, or merges unrelated lines, the parser may classify fields incorrectly.

### Impact

- Multi-column CVs may be parsed in the wrong order.
- Section boundaries may be missed.
- Project, experience, and education blocks may be mixed together.
- Scanned resumes are not supported unless OCR is added.

### Recommendation

Improve observability around raw text extraction, keep original raw text for debugging, and consider adding layout-aware parsing or OCR support later.

## 3. Skill extraction depends on a fixed catalog

Technical skill extraction currently depends on a predefined catalog of known skills and aliases.

### Risk

The parser only extracts skills that exist in the catalog. New frameworks, uncommon tools, spelling variations, or project-specific technologies can be missed.

### Impact

- Candidate skills may be under-reported.
- CV-JD scoring may be lower than expected when missing skills are not detected.
- The skill catalog requires regular maintenance.

### Recommendation

Keep the catalog easy to update, add aliases from real CV samples, and later consider hybrid extraction using rules plus AI-assisted candidate skill detection.

## 4. Projects and work experience can be confused

Projects and work experience share similar signals: role, technologies, dates, descriptions, responsibilities, and links.

### Risk

When a CV does not have clear section headings, the parser may classify a project as work experience or classify work experience as a project.

### Impact

- Experience entries may contain project data.
- Project entries may contain company/job data.
- Downstream scoring may overvalue or undervalue the candidate profile.

### Recommendation

Continue improving section boundary detection, keep project and experience test cases separate, and add tests for CVs without clear headings.

## 5. Full name and location extraction remain fragile

Full name and location extraction are still based on simple heuristics such as early-line detection, labels, known city names, and formatting assumptions.

### Risk

The parser may select the wrong line as the candidate name or miss the location when the resume uses a non-standard layout.

### Impact

- Candidate profile may show incorrect personal information.
- Vietnamese names with initials, short names, or uncommon formatting may be missed.
- Contact/profile sections with unusual ordering may reduce accuracy.

### Recommendation

Add more tests for Vietnamese names, short names, names with initials, and CVs where contact details appear before or beside the name.

## 6. Section splitting can fail on non-standard headings

The parser depends on section headings and known aliases to split resume content into meaningful blocks.

### Risk

If a resume uses uncommon section names or no section headings at all, the parser may fall back to weaker detection logic.

### Impact

- Some sections may be parsed from incomplete text.
- Data can be placed into the wrong extractor.
- Fallback logic may create inconsistent results across CV formats.

### Recommendation

Maintain a documented list of supported section aliases and add fallback tests for resumes with missing or uncommon headings.

## 7. Rule-based parsing has limited semantic understanding

The parser can detect patterns and keywords, but it does not deeply understand meaning, context, or intent.

### Risk

A line can contain matching keywords but still have a different meaning from what the parser expects.

### Impact

- A keyword may trigger the wrong classification.
- Responsibilities, achievements, and summaries may be difficult to separate cleanly.
- The parser may struggle with natural language descriptions that do not follow resume conventions.

### Recommendation

Keep the rule-based parser as the deterministic baseline, then consider adding optional AI-assisted validation or enrichment for ambiguous fields.

## 8. Confidence and debugging signals are limited

The current output focuses on parsed values, but it does not consistently expose why a field was selected or how confident the parser is.

### Risk

When a parsed result is wrong, it can be difficult to understand which rule caused the issue.

### Impact

- Debugging parser bugs takes longer.
- It is harder to identify weak fields.
- Frontend/backend consumers cannot distinguish high-confidence data from uncertain data.

### Recommendation

Add lightweight debug metadata internally, such as matched rule, source line, section source, and confidence level. This does not need to be exposed publicly at first.

## 9. Increasing regression risk as parser coverage expands

Each new CV format may require new rules. Without broad test coverage, parser behavior can regress silently.

### Risk

A small regex or keyword change can improve one case while breaking another existing case.

### Impact

- Parser quality may become unstable over time.
- Manual testing becomes more expensive.
- Confidence in parser changes decreases.

### Recommendation

Build a larger suite of anonymized real CV fixtures and track expected output for key fields such as name, email, phone, skills, education, experience, and projects.

## Summary

The current raw-text parser is a solid MVP foundation. Its biggest strength is that it is deterministic, modular, and testable. Its biggest weakness is that it depends on handcrafted heuristics and the quality of extracted raw text.

The next improvement should not be adding unlimited regex rules. The better direction is to improve test coverage, add parser confidence/debug metadata, document supported CV formats, and later introduce hybrid rule-based plus AI-assisted parsing for ambiguous cases.
