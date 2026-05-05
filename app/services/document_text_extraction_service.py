from io import BytesIO
import re
from typing import Any

import httpx
from docx import Document
from pypdf import PdfReader

LINE_Y_TOLERANCE = 3.0
URL_RE = re.compile(r"https?://[^\s)>,;]+", re.IGNORECASE)


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
            return DocumentTextExtractionResult(
                raw_text=pymupdf_text,
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
        left_words = []
        right_words = []

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
            if not item["text"]:
                continue

            if item["x0"] < page_width * 0.38:
                left_words.append(item)
            else:
                right_words.append(item)

        if left_words and right_words:
            page_text = "\n".join(
                [
                    self._words_to_text(left_words, page_links),
                    self._words_to_text(right_words, page_links),
                ],
            )
        else:
            page_text = self._words_to_text(left_words or right_words, page_links)

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

    def _words_to_text(self, words: list[dict[str, float | str]], links: list[dict[str, Any]] | None = None) -> str:
        if not words:
            return ""

        sorted_words = sorted(words, key=lambda word: (float(word["y0"]), float(word["x0"])))
        lines: list[list[dict[str, float | str]]] = []

        for word in sorted_words:
            if not lines:
                lines.append([word])
                continue

            current_line = lines[-1]
            current_y = sum(float(item["y0"]) for item in current_line) / len(current_line)
            if abs(float(word["y0"]) - current_y) <= LINE_Y_TOLERANCE:
                current_line.append(word)
            else:
                lines.append([word])

        line_texts = []
        for line in lines:
            ordered_line = sorted(line, key=lambda word: float(word["x0"]))
            line_text = " ".join(str(word["text"]) for word in ordered_line).strip()
            line_urls = self._urls_for_line(ordered_line, links or [])

            if line_text:
                line_texts.append(line_text)
            for url in line_urls:
                line_texts.append(url)

        return "\n".join(line_texts)

    def _urls_for_line(
        self,
        line: list[dict[str, float | str]],
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
