from app.services.document_text_extraction_service import DocumentTextExtractionService


def segment(text: str, x0: float, y0: float, x1: float | None = None) -> dict:
    return {
        "x0": x0,
        "y0": y0,
        "x1": x1 if x1 is not None else x0 + 100,
        "y1": y0 + 10,
        "text": text,
        "urls": [],
    }


def test_segments_to_reading_order_preserves_full_width_header_before_columns():
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
        "University of Science and Technology",
        "WORK EXPERIENCE",
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
