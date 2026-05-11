import logging
import platform

from app.services.document_text_extraction_service import (
    DocumentTextExtractionService,
    PdfExtractionQuality,
)

logger = logging.getLogger(__name__)

_original_maybe_merge_local_ocr_text = DocumentTextExtractionService._maybe_merge_local_ocr_text


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


DocumentTextExtractionService._maybe_merge_local_ocr_text = _maybe_merge_local_ocr_text
