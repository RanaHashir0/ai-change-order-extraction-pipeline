"""Unit tests for pdf_reader module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import fitz
import pytest

from change_order_extractor.pdf_reader import (
    CorruptDocumentError,
    DocumentContent,
    EmptyDocumentError,
    ScannedPDFError,
    UnsupportedFormatError,
    is_ocr_available,
    read_document,
)

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"


def test_read_real_sample_pdfs():
    """Verify all 6 sample PDFs in samples/ can be successfully read with PyMuPDF."""
    sample_files = list(SAMPLES_DIR.glob("*.pdf"))
    assert len(sample_files) == 6, f"Expected 6 sample files, found {len(sample_files)}"

    for pdf_file in sample_files:
        doc_content = read_document(pdf_file)
        assert isinstance(doc_content, DocumentContent)
        assert doc_content.file_type == "pdf"
        assert doc_content.page_count >= 1
        assert doc_content.char_count > 100
        assert not doc_content.is_scanned
        assert not doc_content.ocr_applied
        # Ensure document text preserves important markers
        assert any(
            marker in doc_content.raw_text.upper()
            for marker in ["CHANGE ORDER", "CO SAMPLE", "CONTRACT"]
        )


def test_read_plain_text_file(tmp_path: Path):
    """Test reading a valid plain text document."""
    txt_path = tmp_path / "change_order.txt"
    sample_text = (
        "Project: Oakridge Hospital Expansion\n"
        "Contract: CT-2024-99\n"
        "Amount: $15,250.00\n"
        "Schedule Impact: +5 days"
    )
    txt_path.write_text(sample_text, encoding="utf-8")

    doc = read_document(txt_path)
    assert doc.file_type == "txt"
    assert doc.page_count == 1
    assert "$15,250.00" in doc.raw_text
    assert "+5 days" in doc.raw_text
    assert not doc.is_scanned


def test_read_nonexistent_file():
    """Test that a non-existent file path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        read_document("samples/does_not_exist.pdf")


def test_unsupported_format(tmp_path: Path):
    """Test that an unsupported extension raises UnsupportedFormatError."""
    docx_path = tmp_path / "document.docx"
    docx_path.write_text("dummy", encoding="utf-8")

    with pytest.raises(UnsupportedFormatError) as exc_info:
        read_document(docx_path)
    assert "Unsupported file format" in str(exc_info.value)


def test_empty_zero_byte_file(tmp_path: Path):
    """Test that a 0-byte file raises EmptyDocumentError."""
    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("", encoding="utf-8")

    with pytest.raises(EmptyDocumentError) as exc_info:
        read_document(empty_file)
    assert "empty (0 bytes)" in str(exc_info.value)


def test_whitespace_only_text_file(tmp_path: Path):
    """Test that a text file containing only whitespace raises EmptyDocumentError."""
    ws_file = tmp_path / "whitespace.txt"
    ws_file.write_text("   \n\t  \r\n   ", encoding="utf-8")

    with pytest.raises(EmptyDocumentError) as exc_info:
        read_document(ws_file)
    assert "contains no readable text" in str(exc_info.value)


def test_empty_pdf_document(tmp_path: Path):
    """Test that a PDF with a blank page and no text raises EmptyDocumentError or ScannedPDFError."""
    pdf_path = tmp_path / "blank.pdf"
    doc = fitz.open()
    doc.new_page()  # blank page, 0 characters
    doc.save(str(pdf_path))
    doc.close()

    # When OCR is unavailable, a low-char PDF will trigger ScannedPDFError
    with pytest.raises((EmptyDocumentError, ScannedPDFError)):
        read_document(pdf_path)


def test_scanned_pdf_detection_without_ocr(tmp_path: Path):
    """Test that a PDF with very little text is detected as scanned and raises ScannedPDFError when OCR is disabled or missing."""
    pdf_path = tmp_path / "scanned_stub.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Hi")  # only 2 chars, below threshold
    doc.save(str(pdf_path))
    doc.close()

    with patch("change_order_extractor.pdf_reader.is_ocr_available", return_value=False):
        with pytest.raises(ScannedPDFError) as exc_info:
            read_document(pdf_path)
        assert "Optical Character Recognition (OCR) is required" in str(exc_info.value)


def test_scanned_pdf_with_successful_mocked_ocr(tmp_path: Path):
    """Test that OCR fallback succeeds when pytesseract OCR returns text."""
    pdf_path = tmp_path / "scanned_image.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "X")  # below threshold
    doc.save(str(pdf_path))
    doc.close()

    mock_ocr_text = "--- Page 1 (OCR) ---\nPROJECT: Scanned Hospital Project\nAMOUNT: $50,000"

    with patch("change_order_extractor.pdf_reader.is_ocr_available", return_value=True):
        with patch("change_order_extractor.pdf_reader._perform_ocr", return_value=mock_ocr_text):
            result = read_document(pdf_path)
            assert result.is_scanned is True
            assert result.ocr_applied is True
            assert "Scanned Hospital Project" in result.raw_text
            assert "$50,000" in result.raw_text

