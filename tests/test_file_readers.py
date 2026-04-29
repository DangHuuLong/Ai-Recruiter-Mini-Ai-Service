import sys
import types
from pathlib import Path

import pytest

from app.parsers.docx_reader import read_docx
from app.parsers.pdf_reader import read_pdf


def test_read_pdf_extracts_text_with_pypdf(monkeypatch):
    pdf_path = Path("tests/.tmp_resume_reader.pdf")
    pdf_path.write_bytes(b"%PDF-1.4")

    class FakePage:
        def __init__(self, text):
            self._text = text

        def extract_text(self):
            return self._text

    class FakePdfReader:
        def __init__(self, path):
            assert path == str(pdf_path)
            self.pages = [FakePage("John Doe"), FakePage("Python FastAPI")]

    fake_pypdf = types.ModuleType("pypdf")
    fake_pypdf.PdfReader = FakePdfReader
    monkeypatch.setitem(sys.modules, "pypdf", fake_pypdf)

    try:
        assert read_pdf(pdf_path) == "John Doe\nPython FastAPI"
    finally:
        pdf_path.unlink(missing_ok=True)


def test_read_pdf_rejects_missing_file():
    with pytest.raises(FileNotFoundError):
        read_pdf("tests/.missing.pdf")


def test_read_docx_extracts_paragraph_and_table_text(monkeypatch):
    docx_path = Path("tests/.tmp_resume_reader.docx")
    docx_path.write_bytes(b"docx")

    class Paragraph:
        def __init__(self, text):
            self.text = text

    class Cell:
        def __init__(self, text):
            self.text = text

    class Row:
        cells = [Cell("GitHub: github.com/test")]

    class Table:
        rows = [Row()]

    class FakeDocument:
        paragraphs = [Paragraph("Jane Doe"), Paragraph("Python, Django")]
        tables = [Table()]

    fake_docx = types.ModuleType("docx")
    fake_docx.Document = lambda path: FakeDocument()
    monkeypatch.setitem(sys.modules, "docx", fake_docx)

    try:
        assert read_docx(docx_path) == "Jane Doe\nPython, Django\nGitHub: github.com/test"
    finally:
        docx_path.unlink(missing_ok=True)


def test_read_docx_rejects_missing_file():
    with pytest.raises(FileNotFoundError):
        read_docx("tests/.missing.docx")
