from io import BytesIO
import re
from typing import Any

import httpx
from docx import Document
from pypdf import PdfReader

LINE_Y_TOLERANCE = 3.0


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
            return page.get_text("text", sort=True)

        page_width = float(page.rect.width)
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
            return "\n".join(
                [
                    self._words_to_text(left_words),
                    self._words_to_text(right_words),
                ],
            )

        return self._words_to_text(left_words or right_words)

    def _words_to_text(self, words: list[dict[str, float | str]]) -> str:
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
            line_text = " ".join(str(word["text"]) for word in ordered_line)
            if line_text.strip():
                line_texts.append(line_text)

        return "\n".join(line_texts)

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
