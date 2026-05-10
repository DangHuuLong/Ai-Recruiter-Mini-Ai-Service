import logging

from app.core.config import get_settings
from app.services.document_text_extraction_service import (
    OCR_RENDER_ZOOM,
    DocumentTextExtractionService,
    OcrExtractionResult,
    OcrLine,
)

logger = logging.getLogger(__name__)


def _try_extract_pdf_text_with_paddleocr(
    self: DocumentTextExtractionService,
    document_bytes: bytes,
    regions: list[str],
    warnings: list[str],
) -> OcrExtractionResult:
    try:
        import fitz  # PyMuPDF
        import numpy as np
        from paddleocr import PaddleOCR
        from PIL import Image
    except ImportError as exc:
        message = f"PaddleOCR fallback skipped because optional OCR dependencies are missing: {exc}"
        logger.exception(message)
        warnings.append(message)
        return OcrExtractionResult()

    settings = get_settings()
    max_pages = max(settings.pdf_ocr_max_pages, 1)
    min_confidence = settings.pdf_ocr_min_confidence
    lang = (settings.pdf_ocr_languages or "en").split("+")[0]

    try:
        logger.info("Initializing PaddleOCR fallback with lang=%s", lang)
        ocr = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)
    except Exception as exc:
        message = f"PaddleOCR fallback initialization failed: {exc}"
        logger.exception(message)
        warnings.append(message)
        return OcrExtractionResult()

    ocr_lines: list[OcrLine] = []
    stats = {"raw": 0, "accepted": 0, "empty": 0, "low_confidence": 0}
    unique_regions = list(dict.fromkeys(regions))

    try:
        with fitz.open(stream=document_bytes, filetype="pdf") as document:
            for page_index, page in enumerate(document):
                if page_index >= max_pages:
                    break
                for region_name, rect in self._ocr_regions_for_page(page, unique_regions):
                    logger.info(
                        "Running PaddleOCR page=%s region=%s rect=(%.1f, %.1f, %.1f, %.1f)",
                        page_index + 1,
                        region_name,
                        float(rect.x0),
                        float(rect.y0),
                        float(rect.x1),
                        float(rect.y1),
                    )
                    pix = page.get_pixmap(
                        matrix=fitz.Matrix(OCR_RENDER_ZOOM, OCR_RENDER_ZOOM),
                        clip=rect,
                        alpha=False,
                    )
                    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    image_array = np.asarray(image)
                    result = ocr.ocr(image_array, cls=True)
                    converted_lines, converted_stats = self._paddleocr_result_to_lines(
                        result,
                        page_index + 1,
                        region_name,
                        rect,
                        OCR_RENDER_ZOOM,
                        min_confidence,
                    )
                    ocr_lines.extend(converted_lines)
                    for key, value in converted_stats.items():
                        stats[key] += value
    except Exception as exc:
        message = f"PaddleOCR fallback failed: {exc}"
        logger.exception(message)
        warnings.append(message)
        return OcrExtractionResult(
            lines=ocr_lines,
            raw_items_count=stats["raw"],
            accepted_items_count=stats["accepted"],
            rejected_empty_count=stats["empty"],
            rejected_low_confidence_count=stats["low_confidence"],
        )

    sorted_lines = sorted(ocr_lines, key=lambda line: (line.page, line.y0 or 0, line.x0 or 0))
    raw_text = "\n".join(line.text for line in sorted_lines if line.text)
    if raw_text:
        sample = " | ".join(line.text for line in sorted_lines[:8])
        message = (
            f"PaddleOCR fallback extracted text from regions: {', '.join(unique_regions)}; "
            f"sample={sample[:500]}"
        )
        logger.info(message)
        warnings.append(message)

    return OcrExtractionResult(
        raw_text=raw_text,
        lines=sorted_lines,
        raw_items_count=stats["raw"],
        accepted_items_count=stats["accepted"],
        rejected_empty_count=stats["empty"],
        rejected_low_confidence_count=stats["low_confidence"],
    )


DocumentTextExtractionService._try_extract_pdf_text_with_paddleocr = _try_extract_pdf_text_with_paddleocr
