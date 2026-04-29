import re


URL_RE = re.compile(r"https?://[\w\-./?=#%&]+", re.IGNORECASE)


def extract_links(text: str) -> dict:
    urls = list({m.group(0).rstrip('.,;') for m in URL_RE.finditer(text)})
    result = {"linkedin": None, "github": None, "portfolio": None, "other": []}
    for u in urls:
        lu = u.lower()
        if "linkedin.com" in lu:
            result["linkedin"] = u
        elif "github.com" in lu:
            result["github"] = u
        else:
            # heuristics for portfolio
            if any(x in lu for x in ["portfolio", "behance", "dribbble"]) or u.endswith(".dev"):
                result["portfolio"] = u
            else:
                result["other"].append(u)

    return result
