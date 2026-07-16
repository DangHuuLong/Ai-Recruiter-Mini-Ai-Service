from unittest.mock import patch

from app.services.document_text_extraction_service import (
    DocumentTextExtractionService,
    OcrExtractionResult,
)


def segment(text: str, x0: float, y0: float, x1: float | None = None) -> dict:
    return {
        "x0": x0,
        "y0": y0,
        "x1": x1 if x1 is not None else x0 + 100,
        "y1": y0 + 10,
        "text": text,
        "urls": [],
    }


def test_segments_to_reading_order_preserves_full_width_header_and_row_order_columns():
    service = DocumentTextExtractionService()

    text = service._segments_to_reading_order_text(
        [
            segment("Nguyen Thanh Hieu", 48, 20, 260),
            segment("nguyenthanhhieu17022005@gmail.com | 0587381816", 48, 40, 520),
            segment("OBJECTIVE", 48, 80, 160),
            segment("I am seeking an internship where I can apply and strengthen my skills", 48, 100, 520),
            segment("EDUCATION", 48, 160, 180),
            segment("WORK EXPERIENCE", 320, 160, 520),
            segment("University of Science and Technology", 48, 180, 230),
            segment("S-Group", 320, 180, 400),
            segment("PROJECTS", 320, 230, 430),
        ],
        page_width=600,
    )

    assert text.split("\n") == [
        "Nguyen Thanh Hieu",
        "nguyenthanhhieu17022005@gmail.com | 0587381816",
        "OBJECTIVE",
        "I am seeking an internship where I can apply and strengthen my skills",
        "EDUCATION",
        "WORK EXPERIENCE",
        "University of Science and Technology",
        "S-Group",
        "PROJECTS",
    ]


def test_words_to_line_segments_keeps_full_width_line_together_until_large_gap():
    service = DocumentTextExtractionService()

    words = [
        {"x0": 48, "y0": 20, "x1": 110, "y1": 30, "text": "I am"},
        {"x0": 116, "y0": 20, "x1": 190, "y1": 30, "text": "seeking"},
        {"x0": 196, "y0": 20, "x1": 280, "y1": 30, "text": "internship"},
        {"x0": 48, "y0": 60, "x1": 130, "y1": 70, "text": "EDUCATION"},
        {"x0": 320, "y0": 60, "x1": 460, "y1": 70, "text": "WORK EXPERIENCE"},
    ]

    segments = service._words_to_line_segments(words, links=[], page_width=600)

    assert [item["text"] for item in segments] == [
        "I am seeking internship",
        "EDUCATION",
        "WORK EXPERIENCE",
    ]


def test_configured_ocr_languages_splits_plus_separated_list():
    service = DocumentTextExtractionService()

    with patch("app.services.document_text_extraction_service.get_settings") as mock_settings:
        mock_settings.return_value.pdf_ocr_languages = "en+vi"
        assert service._configured_ocr_languages() == ["en", "vi"]


def test_configured_ocr_languages_defaults_to_en_when_unset():
    service = DocumentTextExtractionService()

    with patch("app.services.document_text_extraction_service.get_settings") as mock_settings:
        mock_settings.return_value.pdf_ocr_languages = ""
        assert service._configured_ocr_languages() == ["en"]


class TestPaddleOcrMultiLanguageFallback:
    """Previously only the FIRST configured OCR language was ever used
    (`.split("+")[0]`) — a Vietnamese CV/JD with corrupted PyMuPDF text
    (legacy font encoding) would trigger OCR fallback but then run an
    English-only PaddleOCR model against it, never actually fixing the
    Vietnamese text. _try_extract_pdf_text_with_paddleocr now tries each
    configured language and stops at the first clean (non-corrupted) result."""

    def test_stops_at_first_language_with_clean_result(self) -> None:
        service = DocumentTextExtractionService()

        with patch.object(service, "_configured_ocr_languages", return_value=["en", "vi"]), patch.object(
            service,
            "_run_paddleocr_for_language",
            side_effect=[OcrExtractionResult(raw_text="Clean English text")],
        ) as mock_run:
            result = service._try_extract_pdf_text_with_paddleocr(b"fake-pdf-bytes", ["header"], [])

        assert result.raw_text == "Clean English text"
        mock_run.assert_called_once_with(b"fake-pdf-bytes", ["header"], [], "en")

    def test_falls_through_to_next_language_when_first_is_corrupted(self) -> None:
        service = DocumentTextExtractionService()

        with patch.object(service, "_configured_ocr_languages", return_value=["en", "vi"]), patch.object(
            service,
            "_run_paddleocr_for_language",
            side_effect=[
                OcrExtractionResult(raw_text="corrupted �� text"),
                OcrExtractionResult(raw_text="Kinh nghiem lam viec"),
            ],
        ) as mock_run:
            result = service._try_extract_pdf_text_with_paddleocr(b"fake-pdf-bytes", ["header"], [])

        assert result.raw_text == "Kinh nghiem lam viec"
        assert mock_run.call_count == 2
        mock_run.assert_any_call(b"fake-pdf-bytes", ["header"], [], "en")
        mock_run.assert_any_call(b"fake-pdf-bytes", ["header"], [], "vi")

    def test_returns_longest_result_when_no_language_is_clean(self) -> None:
        service = DocumentTextExtractionService()

        with patch.object(service, "_configured_ocr_languages", return_value=["en", "vi"]), patch.object(
            service,
            "_run_paddleocr_for_language",
            side_effect=[
                OcrExtractionResult(raw_text="short �"),
                OcrExtractionResult(raw_text="longer but still � corrupted text"),
            ],
        ):
            result = service._try_extract_pdf_text_with_paddleocr(b"fake-pdf-bytes", ["header"], [])

        assert result.raw_text == "longer but still � corrupted text"
