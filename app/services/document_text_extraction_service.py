from io import BytesIO
import re
from typing import Any

import httpx
from docx import Document
from pypdf import PdfReader

VIETNAMESE_WORD_RE = re.compile(r"^[A-Za-zÀ-ỹĐđ]+$")
COMMON_VIETNAMESE_ONSET_FRAGMENTS = {
    "b",
    "c",
    "ch",
    "d",
    "đ",
    "g",
    "gh",
    "gi",
    "h",
    "k",
    "kh",
    "l",
    "m",
    "n",
    "ng",
    "ngh",
    "nh",
    "p",
    "ph",
    "q",
    "qu",
    "r",
    "s",
    "t",
    "th",
    "tr",
    "v",
    "x",
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
            return DocumentTextExtractionResult(
                raw_text=pymupdf_text,
                method="PDF_TEXT_PYMUPDF",
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

        return self._normalize_extracted_text("\n".join(page_texts), repair_pdf_spacing=True)

    def _extract_pymupdf_page_text(self, page: Any) -> str:
        page_width = float(page.rect.width)
        blocks = []

        for block in page.get_text("blocks", sort=True):
            if len(block) < 5:
                continue

            x0, y0, _x1, _y1, text = block[:5]
            normalized_text = self._normalize_extracted_text(str(text), repair_pdf_spacing=True)
            if not normalized_text:
                continue

            blocks.append(
                {
                    "x0": float(x0),
                    "y0": float(y0),
                    "text": normalized_text,
                },
            )

        if not blocks:
            return ""

        left_column_blocks = [block for block in blocks if block["x0"] < page_width * 0.38]
        right_column_blocks = [block for block in blocks if block["x0"] >= page_width * 0.38]

        if left_column_blocks and right_column_blocks:
            ordered_blocks = [
                *sorted(left_column_blocks, key=lambda block: (block["y0"], block["x0"])),
                *sorted(right_column_blocks, key=lambda block: (block["y0"], block["x0"])),
            ]
        else:
            ordered_blocks = sorted(blocks, key=lambda block: (block["y0"], block["x0"]))

        return "\n".join(block["text"] for block in ordered_blocks)

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

        return self._normalize_extracted_text("\n".join(page_texts), repair_pdf_spacing=True)

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

    def _normalize_extracted_text(self, text: str, repair_pdf_spacing: bool = False) -> str:
        lines = []
        for raw_line in (
            (text or "").replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
        ):
            line = re.sub(r"[ \t\f\v]+", " ", raw_line).strip()
            if repair_pdf_spacing:
                line = self._repair_broken_vietnamese_spacing(line)
            if line:
                lines.append(line)

        return "\n".join(lines).strip()

    def _repair_broken_vietnamese_spacing(self, line: str) -> str:
        tokens = line.split()
        if len(tokens) < 2 or not self._looks_like_broken_vietnamese_line(tokens):
            return line

        repaired_tokens: list[str] = []
        index = 0

        while index < len(tokens):
            current = tokens[index]

            while index + 1 < len(tokens) and self._should_join_broken_vietnamese_token(
                current,
                tokens[index + 1],
            ):
                index += 1
                current = f"{current}{tokens[index]}"

            repaired_tokens.append(current)
            index += 1

        return " ".join(repaired_tokens)

    def _looks_like_broken_vietnamese_line(self, tokens: list[str]) -> bool:
        word_tokens = [token for token in tokens if self._is_word_token(token)]
        single_letter_count = sum(1 for token in word_tokens if len(token) == 1)
        short_fragment_count = sum(
            1
            for token in word_tokens
            if len(token) <= 2 and token.lower() in COMMON_VIETNAMESE_ONSET_FRAGMENTS
        )

        return single_letter_count >= 1 or short_fragment_count >= 2

    def _should_join_broken_vietnamese_token(self, left: str, right: str) -> bool:
        if not self._is_word_token(left) or not self._is_word_token(right):
            return False

        left_lower = left.lower()

        if len(right) == 1 and len(left) <= 4:
            return True

        if len(left) == 1 and len(right) <= 6:
            return True

        if left_lower in COMMON_VIETNAMESE_ONSET_FRAGMENTS and len(right) <= 6:
            return True

        return False

    def _is_word_token(self, token: str) -> bool:
        return bool(VIETNAMESE_WORD_RE.fullmatch(token))


document_text_extraction_service = DocumentTextExtractionService()
