from io import BytesIO
import re
from typing import Any

import httpx
from docx import Document
from pypdf import PdfReader

LINE_Y_TOLERANCE = 3.0
COLUMN_SPLIT_RATIO = 0.38
COLUMN_SEGMENT_GAP_THRESHOLD = 36.0
URL_RE = re.compile(r"https?://[^\s)>,;]+", re.IGNORECASE)
IMPORTANT_PDF_SECTION_NAMES = {
    "experience": {"experience", "work experience"},
}
SECTION_STOP_NAMES = {
    "academic",
    "achievements",
    "awards",
    "certification",
    "certifications",
    "contact",
    "education",
    "honors",
    "honours",
    "languages",
    "profile",
    "projects",
    "skills",
    "summary",
    "technical skills",
}


class DocumentTextExtractionError(Exception):
    pass


class DocumentTextExtractionResult:
    def __init__(self, raw_text: str, method: str, warnings: list[str] | None = None):
        self.raw_text = raw_text
        self.method = method
        self.warnings = warnings or []


class DocumentTextExtractionService:
    def extract_from_signed_url(
        self,
        signed_url: str,
        file_type: str,
        file_name: str | None = None,
    ) -> DocumentTextExtractionResult:
        document_bytes = self._download_document(signed_url)
        normalized_file_type = self._normalize_file_type(file_type, file_name)

        if normalized_file_type == "PDF":
            return self._extract_pdf_text(document_bytes)

        if normalized_file_type == "DOCX":
            return self._extract_docx_text(document_bytes)

        raise DocumentTextExtractionError(
            f"Unsupported resume file type: {file_type or file_name or 'unknown'}",
        )

    def _download_document(self, signed_url: str) -> bytes:
        try:
            with httpx.Client(timeout=30.0, follow_redirects=True) as client:
                response = client.get(signed_url)
                response.raise_for_status()
                return response.content
        except httpx.HTTPError as exc:
            raise DocumentTextExtractionError(f"Failed to download resume document: {exc}") from exc

    def _normalize_file_type(self, file_type: str, file_name: str | None) -> str:
        upper_file_type = (file_type or "").upper()

        if upper_file_type in {"PDF", "DOCX"}:
            return upper_file_type

        lower_file_name = (file_name or "").lower()

        if lower_file_name.endswith(".pdf"):
            return "PDF"

        if lower_file_name.endswith(".docx"):
            return "DOCX"

        return upper_file_type

    def _extract_pdf_text(self, document_bytes: bytes) -> DocumentTextExtractionResult:
        warnings: list[str] = []

        pymupdf_text = self._try_extract_pdf_text_with_pymupdf(document_bytes, warnings)
        if pymupdf_text:
            pypdf_text = self._try_extract_pdf_text_with_pypdf(document_bytes, warnings)
            merged_text = self._merge_supplemental_pdf_text(pymupdf_text, pypdf_text, warnings)
            return DocumentTextExtractionResult(
                raw_text=merged_text,
                method="PDF_TEXT_PYMUPDF_WORDS",
                warnings=warnings,
            )

        pypdf_text = self._try_extract_pdf_text_with_pypdf(document_bytes, warnings)
        if pypdf_text:
            return DocumentTextExtractionResult(
                raw_text=pypdf_text,
                method="PDF_TEXT_PYPDF",
                warnings=warnings,
            )

        warnings.append(
            "No selectable text was extracted from the PDF. OCR fallback is not implemented yet.",
        )
        return DocumentTextExtractionResult(
            raw_text="",
            method="PDF_TEXT_EMPTY",
            warnings=warnings,
        )

    def _merge_supplemental_pdf_text(self, primary_text: str, fallback_text: str, warnings: list[str]) -> str:
        """Append important sections that PyMuPDF layout extraction missed.

        PyMuPDF words preserve layout better for multi-column resumes, so it
        remains the primary source. Some PDFs, however, drop bottom sections or
        text boxes in the word stream. pypdf sometimes still exposes those
        sections. This merger only appends known important missing sections,
        avoiding a broad raw-text concatenation that would duplicate most of the
        resume or destroy layout ordering.
        """
        primary = self._normalize_extracted_text(primary_text)
        fallback = self._normalize_extracted_text(fallback_text)

        if not primary or not fallback:
            return primary or fallback

        primary_normalized = self._normalize_section_key(primary)
        supplemental_sections: list[str] = []

        for section_name, aliases in IMPORTANT_PDF_SECTION_NAMES.items():
            if any(alias in primary_normalized for alias in aliases):
                continue

            section_text = self._extract_named_section(fallback, aliases)
            if section_text:
                supplemental_sections.append(section_text)
                warnings.append(f"Merged missing PDF section from pypdf fallback: {section_name}")

        if not supplemental_sections:
            return primary

        return self._normalize_extracted_text("\n".join([primary, *supplemental_sections]))

    def _extract_named_section(self, text: str, aliases: set[str]) -> str:
        lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
        collected: list[str] = []
        collecting = False

        for line in lines:
            normalized = self._normalize_section_key(line)

            if not collecting and normalized in aliases:
                collecting = True
                collected.append(line)
                continue

            if collecting and normalized in SECTION_STOP_NAMES and normalized not in aliases:
                break

            if collecting:
                collected.append(line)

        return "\n".join(collected).strip()

    def _normalize_section_key(self, value: str) -> str:
        lowered = (value or "").lower()
        lowered = re.sub(r"[^a-z0-9+#./ ]+", " ", lowered)
        return re.sub(r"\s+", " ", lowered).strip()

    def _try_extract_pdf_text_with_pymupdf(
        self,
        document_bytes: bytes,
        warnings: list[str],
    ) -> str:
        try:
            import fitz  # PyMuPDF
        except ImportError:
            warnings.append("PyMuPDF is not installed; falling back to pypdf for PDF text extraction.")
            return ""

        try:
            with fitz.open(stream=document_bytes, filetype="pdf") as document:
                page_texts = [self._extract_pymupdf_page_text(page) for page in document]
        except Exception as exc:  # PyMuPDF raises multiple parser-specific exceptions.
            warnings.append(f"PyMuPDF PDF extraction failed; falling back to pypdf: {exc}")
            return ""

        return self._normalize_extracted_text("\n".join(page_texts))

    def _extract_pymupdf_page_text(self, page: Any) -> str:
        words = page.get_text("words", sort=False)
        if not words:
            text = page.get_text("text", sort=True)
            return self._append_page_links(text, self._extract_page_links(page))

        page_width = float(page.rect.width)
        page_links = self._extract_page_links(page)
        normalized_words = []

        for word in words:
            if len(word) < 5:
                continue

            x0, y0, x1, y1, text = word[:5]
            item = {
                "x0": float(x0),
                "y0": float(y0),
                "x1": float(x1),
                "y1": float(y1),
                "text": str(text).strip(),
            }
            if item["text"]:
                normalized_words.append(item)

        line_segments = self._words_to_line_segments(normalized_words, page_links, page_width)
        page_text = self._segments_to_reading_order_text(line_segments, page_width)

        return self._append_unmatched_links(page_text, page_links)

    def _extract_page_links(self, page: Any) -> list[dict[str, Any]]:
        links: list[dict[str, Any]] = []

        try:
            raw_links = page.get_links()
        except Exception:
            return links

        for raw_link in raw_links:
            uri = self._normalize_url(raw_link.get("uri", ""))
            rect = raw_link.get("from")
            if not uri or rect is None:
                continue

            links.append(
                {
                    "url": uri,
                    "x0": float(rect.x0),
                    "y0": float(rect.y0),
                    "x1": float(rect.x1),
                    "y1": float(rect.y1),
                    "matched": False,
                },
            )

        return links

    def _words_to_text(self, words: list[dict[str, Any]], links: list[dict[str, Any]] | None = None) -> str:
        if not words:
            return ""

        line_texts = []
        for line in self._group_items_into_lines(words):
            ordered_line = sorted(line, key=lambda word: float(word["x0"]))
            line_text = " ".join(str(word["text"]) for word in ordered_line).strip()
            line_urls = self._urls_for_line(ordered_line, links or [])

            if line_text:
                line_texts.append(line_text)
            for url in line_urls:
                line_texts.append(url)

        return "\n".join(line_texts)

    def _words_to_line_segments(
        self,
        words: list[dict[str, Any]],
        links: list[dict[str, Any]],
        page_width: float,
    ) -> list[dict[str, Any]]:
        if not words:
            return []

        segments: list[dict[str, Any]] = []
        gap_threshold = max(COLUMN_SEGMENT_GAP_THRESHOLD, page_width * 0.06)

        for line in self._group_items_into_lines(words):
            ordered_line = sorted(line, key=lambda word: float(word["x0"]))
            current_segment: list[dict[str, Any]] = []

            for word in ordered_line:
                if current_segment:
                    previous_word = current_segment[-1]
                    gap = float(word["x0"]) - float(previous_word["x1"])
                    if gap > gap_threshold:
                        segments.append(self._build_text_segment(current_segment, links))
                        current_segment = []

                current_segment.append(word)

            if current_segment:
                segments.append(self._build_text_segment(current_segment, links))

        return segments

    def _build_text_segment(
        self,
        words: list[dict[str, Any]],
        links: list[dict[str, Any]],
    ) -> dict[str, Any]:
        ordered_words = sorted(words, key=lambda word: float(word["x0"]))
        return {
            "x0": min(float(word["x0"]) for word in ordered_words),
            "y0": min(float(word["y0"]) for word in ordered_words),
            "x1": max(float(word["x1"]) for word in ordered_words),
            "y1": max(float(word["y1"]) for word in ordered_words),
            "text": " ".join(str(word["text"]) for word in ordered_words).strip(),
            "urls": self._urls_for_line(ordered_words, links),
        }

    def _segments_to_reading_order_text(
        self,
        segments: list[dict[str, Any]],
        page_width: float,
    ) -> str:
        if not segments:
            return ""

        column_start_y = self._detect_two_column_start_y(segments, page_width)

        if column_start_y is None:
            ordered_segments = sorted(segments, key=lambda segment: (float(segment["y0"]), float(segment["x0"])))
        else:
            before_columns = [
                segment
                for segment in segments
                if float(segment["y0"]) < column_start_y - LINE_Y_TOLERANCE
            ]
            column_segments = [
                segment
                for segment in segments
                if float(segment["y0"]) >= column_start_y - LINE_Y_TOLERANCE
            ]
            split_x = page_width * COLUMN_SPLIT_RATIO
            left_column = [segment for segment in column_segments if float(segment["x0"]) < split_x]
            right_column = [segment for segment in column_segments if float(segment["x0"]) >= split_x]

            ordered_segments = [
                *sorted(before_columns, key=lambda segment: (float(segment["y0"]), float(segment["x0"]))),
                *sorted(left_column, key=lambda segment: (float(segment["y0"]), float(segment["x0"]))),
                *sorted(right_column, key=lambda segment: (float(segment["y0"]), float(segment["x0"]))),
            ]

        output_lines: list[str] = []
        for segment in ordered_segments:
            output_lines.extend(self._segment_output_lines(segment))

        return "\n".join(output_lines)

    def _detect_two_column_start_y(
        self,
        segments: list[dict[str, Any]],
        page_width: float,
    ) -> float | None:
        split_x = page_width * COLUMN_SPLIT_RATIO

        for line in self._group_items_into_lines(segments):
            if len(line) < 2:
                continue

            has_left_segment = any(float(segment["x0"]) < split_x for segment in line)
            has_right_segment = any(float(segment["x0"]) >= split_x for segment in line)

            if has_left_segment and has_right_segment:
                return min(float(segment["y0"]) for segment in line)

        return None

    def _segment_output_lines(self, segment: dict[str, Any]) -> list[str]:
        lines: list[str] = []
        text = str(segment.get("text") or "").strip()
        if text:
            lines.append(text)

        for url in segment.get("urls") or []:
            lines.append(str(url))

        return lines

    def _group_items_into_lines(self, items: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        sorted_items = sorted(items, key=lambda item: (float(item["y0"]), float(item["x0"])))
        lines: list[list[dict[str, Any]]] = []

        for item in sorted_items:
            if not lines:
                lines.append([item])
                continue

            current_line = lines[-1]
            current_y = sum(float(existing_item["y0"]) for existing_item in current_line) / len(current_line)
            if abs(float(item["y0"]) - current_y) <= LINE_Y_TOLERANCE:
                current_line.append(item)
            else:
                lines.append([item])

        return lines

    def _urls_for_line(
        self,
        line: list[dict[str, Any]],
        links: list[dict[str, Any]],
    ) -> list[str]:
        if not links or not line:
            return []

        min_x = min(float(word["x0"]) for word in line)
        max_x = max(float(word["x1"]) for word in line)
        min_y = min(float(word["y0"]) for word in line)
        max_y = max(float(word["y1"]) for word in line)
        urls: list[str] = []

        for link in links:
            if link["matched"]:
                continue
            horizontally_overlaps = float(link["x1"]) >= min_x - 4 and float(link["x0"]) <= max_x + 4
            vertically_overlaps = float(link["y1"]) >= min_y - 4 and float(link["y0"]) <= max_y + 4

            if horizontally_overlaps and vertically_overlaps:
                link["matched"] = True
                urls.append(str(link["url"]))

        return self._unique_urls(urls)

    def _append_page_links(self, text: str, links: list[dict[str, Any]]) -> str:
        urls = self._unique_urls([str(link["url"]) for link in links])
        if not urls:
            return text

        return "\n".join([text, *urls])

    def _append_unmatched_links(self, text: str, links: list[dict[str, Any]]) -> str:
        urls = self._unique_urls([str(link["url"]) for link in links if not link["matched"]])
        if not urls:
            return text

        existing_urls = set(URL_RE.findall(text or ""))
        missing_urls = [url for url in urls if url not in existing_urls]
        if not missing_urls:
            return text

        return "\n".join([text, *missing_urls])

    def _unique_urls(self, urls: list[str]) -> list[str]:
        result = []
        seen = set()

        for url in urls:
            normalized_url = self._normalize_url(url)
            if not normalized_url:
                continue
            key = normalized_url.rstrip("/").lower()
            if key in seen:
                continue
            seen.add(key)
            result.append(normalized_url)

        return result

    def _normalize_url(self, value: str) -> str | None:
        cleaned = (value or "").strip().rstrip(".,;)")
        if not re.match(r"^https?://", cleaned, flags=re.IGNORECASE):
            return None
        return cleaned

    def _try_extract_pdf_text_with_pypdf(
        self,
        document_bytes: bytes,
        warnings: list[str],
    ) -> str:
        try:
            reader = PdfReader(BytesIO(document_bytes))
            page_texts = [(page.extract_text() or "") for page in reader.pages]
        except Exception as exc:  # pypdf raises multiple parser-specific exceptions.
            warnings.append(f"pypdf PDF extraction failed: {exc}")
            return ""

        return self._normalize_extracted_text("\n".join(page_texts))

    def _extract_docx_text(self, document_bytes: bytes) -> DocumentTextExtractionResult:
        try:
            document = Document(BytesIO(document_bytes))
            raw_text = "\n".join(
                paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()
            ).strip()
        except Exception as exc:  # python-docx may raise zip/xml parser errors for invalid docs.
            raise DocumentTextExtractionError(f"Failed to extract DOCX text: {exc}") from exc

        raw_text = self._normalize_extracted_text(raw_text)
        warnings: list[str] = []

        if not raw_text:
            warnings.append("No text was extracted from the DOCX document.")

        return DocumentTextExtractionResult(
            raw_text=raw_text,
            method="DOCX_TEXT",
            warnings=warnings,
        )

    def _normalize_extracted_text(self, text: str) -> str:
        return "\n".join(
            re.sub(r"[ \t\f\v]+", " ", line).strip()
            for line in (text or "").replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
            if line.strip()
        ).strip()


document_text_extraction_service = DocumentTextExtractionService()
