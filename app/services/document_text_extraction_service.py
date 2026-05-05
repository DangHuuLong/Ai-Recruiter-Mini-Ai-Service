from io import BytesIO

import httpx
from docx import Document
from pypdf import PdfReader


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
        try:
            reader = PdfReader(BytesIO(document_bytes))
            page_texts = [(page.extract_text() or "") for page in reader.pages]
            raw_text = "\n".join(page_texts).strip()
        except Exception as exc:  # pypdf raises multiple parser-specific exceptions.
            raise DocumentTextExtractionError(f"Failed to extract PDF text: {exc}") from exc

        warnings: list[str] = []

        if not raw_text:
            warnings.append(
                "No selectable text was extracted from the PDF. OCR fallback is not implemented yet.",
            )

        return DocumentTextExtractionResult(
            raw_text=raw_text,
            method="PDF_TEXT",
            warnings=warnings,
        )

    def _extract_docx_text(self, document_bytes: bytes) -> DocumentTextExtractionResult:
        try:
            document = Document(BytesIO(document_bytes))
            raw_text = "\n".join(
                paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()
            ).strip()
        except Exception as exc:  # python-docx may raise zip/xml parser errors for invalid docs.
            raise DocumentTextExtractionError(f"Failed to extract DOCX text: {exc}") from exc

        warnings: list[str] = []

        if not raw_text:
            warnings.append("No text was extracted from the DOCX document.")

        return DocumentTextExtractionResult(
            raw_text=raw_text,
            method="DOCX_TEXT",
            warnings=warnings,
        )


document_text_extraction_service = DocumentTextExtractionService()
