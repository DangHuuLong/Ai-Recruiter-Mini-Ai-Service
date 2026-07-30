import logging
import platform
import re
from statistics import median

from app.services.document_text_extraction_service import (
    BAD_GLYPH_MARKERS,
    DocumentTextExtractionService,
    PdfExtractionQuality,
)

logger = logging.getLogger(__name__)

_original_maybe_merge_local_ocr_text = DocumentTextExtractionService._maybe_merge_local_ocr_text
_original_merge_supplemental_pdf_text = DocumentTextExtractionService._merge_supplemental_pdf_text

NON_NAME_FALLBACK_LINES = {
    "academic",
    "achievements",
    "address",
    "backend",
    "certification",
    "certifications",
    "contact",
    "date",
    "education",
    "experience",
    "frontend",
    "full stack developer",
    "github",
    "languages",
    "links",
    "mobile app developer intern",
    "personal project",
    "phone",
    "profile",
    "projects",
    "skills",
    "technical skills",
    "tools",
    "tools services",
    "work experience",
}


def _maybe_merge_local_ocr_text(
    self: DocumentTextExtractionService,
    document_bytes: bytes,
    primary_text: str,
    quality: PdfExtractionQuality,
    warnings: list[str],
) -> str:
    if platform.system().lower() == "windows":
        message = (
            "PaddleOCR fallback skipped on Windows local runtime because PaddleOCR/PaddlePaddle "
            "CPU inference is not stable in this environment. Use Linux/Docker for OCR, or keep "
            "OCR disabled and rely on PDF text-layer parsing plus filename/candidate fallback."
        )
        logger.warning(message)
        warnings.append(message)
        return self._normalize_extracted_text(primary_text)

    try:
        return _original_maybe_merge_local_ocr_text(self, document_bytes, primary_text, quality, warnings)
    except Exception as exc:
        message = f"PaddleOCR fallback failed and was skipped: {exc}"
        logger.warning(message)
        warnings.append(message)
        return self._normalize_extracted_text(primary_text)


def _merge_supplemental_pdf_text(
    self: DocumentTextExtractionService,
    primary_text: str,
    fallback_text: str,
    warnings: list[str],
) -> str:
    merged_text = _original_merge_supplemental_pdf_text(self, primary_text, fallback_text, warnings)
    merged_text = _recover_corrupted_header_name_from_fallback(merged_text, fallback_text, warnings)
    merged_text = _normalize_common_corrupted_glyphs(merged_text)
    return self._normalize_extracted_text(merged_text)


def _recover_corrupted_header_name_from_fallback(primary_text: str, fallback_text: str, warnings: list[str]) -> str:
    if not primary_text or not fallback_text:
        return primary_text

    primary_lines = primary_text.splitlines()
    corrupted_line_index = None
    for index, line in enumerate(primary_lines[:20]):
        if _has_bad_glyph(line) and _looks_like_corrupted_name_line(line):
            corrupted_line_index = index
            break

    if corrupted_line_index is None:
        return primary_text

    clean_name = _find_clean_name_candidate(fallback_text)
    if not clean_name:
        return primary_text

    primary_lines[corrupted_line_index] = clean_name
    warnings.append("Recovered corrupted header name from alternate PDF text extractor")
    return "\n".join(primary_lines)


def _normalize_common_corrupted_glyphs(text: str) -> str:
    replacements = {
        "Bachelor\u02c7s s Degree": "Bachelor's Degree",
        "Bachelor\u02c7s Degree": "Bachelor's Degree",
        "Bachelor's s Degree": "Bachelor's Degree",
    }
    result = text or ""
    for source, replacement in replacements.items():
        result = result.replace(source, replacement)
    return result


def _has_bad_glyph(text: str) -> bool:
    return any(marker in (text or "") for marker in BAD_GLYPH_MARKERS)


def _has_non_ascii_letter(text: str) -> bool:
    return any(char.isalpha() and ord(char) > 127 for char in (text or ""))


def _letters_only(value: str) -> str:
    return "".join(char for char in value if char.isalpha())


def _looks_like_corrupted_name_line(text: str) -> bool:
    words = str(text or "").strip().split()
    if not 2 <= len(words) <= 5:
        return False
    if any(token in str(text).lower() for token in ["@", "phone", "date", "address", "http"]):
        return False
    return True


def _find_clean_name_candidate(text: str) -> str | None:
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    search_lines = [*lines[:35], *lines[-35:]]
    for line in search_lines:
        if _is_clean_person_name(line):
            return line
    return None


def _is_clean_person_name(line: str) -> bool:
    cleaned = re.sub(r"\s+", " ", str(line or "").strip())
    normalized = re.sub(r"[^a-z0-9 ]+", " ", cleaned.lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()

    if not cleaned or _has_bad_glyph(cleaned):
        return False
    if normalized in NON_NAME_FALLBACK_LINES:
        return False
    if any(token in normalized for token in ["@", "http", "phone", "date", "address", "developer", "engineer"]):
        return False
    if not _has_non_ascii_letter(cleaned):
        return False

    words = cleaned.split()
    if not 2 <= len(words) <= 5:
        return False

    return all(_letters_only(word) for word in words)


def _segments_to_reading_order_text(
    self: DocumentTextExtractionService,
    segments: list[dict],
    page_width: float,
) -> str:
    if not segments:
        return ""

    layout = _detect_two_column_layout(segments, page_width)

    if layout is None:
        ordered_segments = sorted(segments, key=lambda segment: (float(segment["y0"]), float(segment["x0"])))
    else:
        split_x = float(layout["split_x"])
        full_width_segments = [segment for segment in segments if _is_full_width_segment(segment, page_width)]
        column_segments = [segment for segment in segments if segment not in full_width_segments]
        left_column = [segment for segment in column_segments if float(segment["x0"]) < split_x]
        right_column = [segment for segment in column_segments if float(segment["x0"]) >= split_x]

        ordered_segments = [
            *sorted(full_width_segments, key=lambda segment: (float(segment["y0"]), float(segment["x0"]))),
            *sorted(left_column, key=lambda segment: (float(segment["y0"]), float(segment["x0"]))),
            *sorted(right_column, key=lambda segment: (float(segment["y0"]), float(segment["x0"]))),
        ]
        logger.info(
            "Detected two-column PDF layout: split_x=%.1f, left_segments=%s, right_segments=%s",
            split_x,
            len(left_column),
            len(right_column),
        )

    output_lines: list[str] = []
    for segment in ordered_segments:
        output_lines.extend(self._segment_output_lines(segment))
    return "\n".join(output_lines)


def _detect_two_column_layout(segments: list[dict], page_width: float) -> dict[str, float] | None:
    candidates = [_layout_candidate_segment(segment) for segment in segments]
    candidates = [segment for segment in candidates if segment is not None]

    if len(candidates) < 16:
        return None

    split_x = page_width * 0.38
    left = [segment for segment in candidates if float(segment["x0"]) < split_x]
    right = [segment for segment in candidates if float(segment["x0"]) >= split_x]

    if len(left) < 8 or len(right) < 8:
        return None

    left_long = [segment for segment in left if _text_weight(segment["text"]) >= 8]
    right_long = [segment for segment in right if _text_weight(segment["text"]) >= 8]

    if len(left_long) < 6 or len(right_long) < 6:
        return None

    median_left_x1 = median(float(segment["x1"]) for segment in left_long)
    median_right_x0 = median(float(segment["x0"]) for segment in right_long)
    column_gap = median_right_x0 - median_left_x1

    if column_gap < page_width * 0.06:
        return None

    left_y_bands = _occupied_y_bands(left_long)
    right_y_bands = _occupied_y_bands(right_long)
    overlap_bands = left_y_bands.intersection(right_y_bands)

    if len(overlap_bands) < 4:
        return None

    date_like_right = sum(1 for segment in right if _looks_like_short_date(segment["text"]))
    if date_like_right and date_like_right / max(len(right), 1) > 0.65:
        return None

    detected_split_x = (median_left_x1 + median_right_x0) / 2
    min_split_x = page_width * 0.28
    max_split_x = page_width * 0.55
    detected_split_x = min(max(detected_split_x, min_split_x), max_split_x)

    return {"split_x": detected_split_x}


def _layout_candidate_segment(segment: dict) -> dict | None:
    text = str(segment.get("text") or "").strip()
    if _text_weight(text) < 2:
        return None
    return segment


def _text_weight(text: str) -> int:
    return len("".join(char for char in (text or "") if char.isalnum()))


def _occupied_y_bands(segments: list[dict], band_height: float = 48.0) -> set[int]:
    bands: set[int] = set()
    for segment in segments:
        y0 = float(segment["y0"])
        y1 = float(segment["y1"])
        start = int(y0 // band_height)
        end = int(y1 // band_height)
        bands.update(range(start, end + 1))
    return bands


def _looks_like_short_date(text: str) -> bool:
    cleaned = str(text or "").strip().lower()
    if len(cleaned.split()) > 4:
        return False
    return bool(
        re.fullmatch(r"(?:\d{1,2}/)?\d{4}(?:\s*[-–]\s*(?:present|\d{1,2}/)?\d{4})?", cleaned)
        or re.fullmatch(r"(?:present|current)", cleaned)
    )


def _is_full_width_segment(segment: dict, page_width: float) -> bool:
    width = float(segment["x1"]) - float(segment["x0"])
    return width >= page_width * 0.72


DocumentTextExtractionService._maybe_merge_local_ocr_text = _maybe_merge_local_ocr_text
DocumentTextExtractionService._merge_supplemental_pdf_text = _merge_supplemental_pdf_text
DocumentTextExtractionService._segments_to_reading_order_text = _segments_to_reading_order_text
