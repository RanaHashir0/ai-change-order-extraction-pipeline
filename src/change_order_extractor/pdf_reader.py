"""Document reader and pre-processing module for construction change orders.

Supports PDF (via PyMuPDF) and plain text documents with empty document detection,
scanned document detection, and optional OCR fallback.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF


class DocumentReaderError(Exception):
    """Base exception for document reading errors."""


class UnsupportedFormatError(DocumentReaderError):
    """Raised when a file format other than PDF or TXT is supplied."""


class EmptyDocumentError(DocumentReaderError):
    """Raised when an ingested document has no readable text or is empty."""


class ScannedPDFError(DocumentReaderError):
    """Raised when a PDF is scanned/image-only and OCR is unavailable or fails."""


class CorruptDocumentError(DocumentReaderError):
    """Raised when a document is damaged or cannot be parsed."""


@dataclass(frozen=True)
class DocumentContent:
    """Represents normalized text and metadata extracted from an input document."""

    raw_text: str
    page_count: int
    source_path: str
    file_type: str  # "pdf" or "txt"
    char_count: int
    is_scanned: bool = False
    ocr_applied: bool = False
    metadata: dict[str, object] = field(default_factory=dict)


def is_ocr_available() -> bool:
    """Check if pytesseract and the underlying Tesseract binary are installed and functional."""
    try:
        import pytesseract  # type: ignore

        # Attempt to query version to verify binary accessibility
        _ = pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def _perform_ocr(doc: fitz.Document) -> str:
    """Execute OCR on each page of a PyMuPDF document using pytesseract."""
    import pytesseract  # type: ignore
    from PIL import Image

    ocr_pages: list[str] = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap(dpi=300)
        img_bytes = pix.tobytes("png")
        image = Image.open(io.BytesIO(img_bytes))
        page_text = pytesseract.image_to_string(image)
        if page_text.strip():
            ocr_pages.append(f"--- Page {page_num + 1} (OCR) ---\n{page_text.strip()}")

    return "\n\n".join(ocr_pages)


def read_document(
    file_path: str | Path,
    *,
    min_chars_per_page: int = 30,
    min_total_chars: int = 50,
    enable_ocr: bool = True,
) -> DocumentContent:
    """Read a PDF or plain text document, returning normalized text and metadata.

    Args:
        file_path: Path to the target file.
        min_chars_per_page: Minimum character density per page to classify as non-scanned.
        min_total_chars: Minimum overall character threshold for valid digital text.
        enable_ocr: Whether to attempt OCR fallback when a scanned PDF is detected.

    Returns:
        DocumentContent with raw text, page count, and diagnostic metadata.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        UnsupportedFormatError: If the file is not a supported .pdf or .txt file.
        EmptyDocumentError: If the file is completely blank or 0 bytes.
        ScannedPDFError: If the PDF is scanned and OCR is unavailable or disabled.
        CorruptDocumentError: If the PDF file is corrupted and unreadable.
    """
    path = Path(file_path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"Target document not found at: {path}")

    if not path.is_file():
        raise DocumentReaderError(f"Target path is not a file: {path}")

    # Check for empty file on disk
    if path.stat().st_size == 0:
        raise EmptyDocumentError(f"Document file is empty (0 bytes): {path.name}")

    suffix = path.suffix.lower()

    if suffix in {".txt", ".text"}:
        return _read_text_file(path)
    elif suffix == ".pdf":
        return _read_pdf_file(
            path,
            min_chars_per_page=min_chars_per_page,
            min_total_chars=min_total_chars,
            enable_ocr=enable_ocr,
        )
    else:
        raise UnsupportedFormatError(
            f"Unsupported file format '{path.suffix}'. "
            f"The extraction pipeline only supports PDF (.pdf) and plain text (.txt) documents."
        )


def _read_text_file(path: Path) -> DocumentContent:
    """Read and validate a plain text document."""
    raw_content: Optional[str] = None
    encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]

    for enc in encodings:
        try:
            with path.open("r", encoding=enc) as f:
                raw_content = f.read()
            break
        except UnicodeDecodeError:
            continue

    if raw_content is None:
        raise CorruptDocumentError(f"Unable to decode text file '{path.name}' with supported encodings.")

    stripped = raw_content.strip()
    if not stripped:
        raise EmptyDocumentError(f"Text document contains no readable text: {path.name}")

    return DocumentContent(
        raw_text=raw_content,
        page_count=1,
        source_path=str(path),
        file_type="txt",
        char_count=len(raw_content),
        is_scanned=False,
        ocr_applied=False,
        metadata={"encoding_used": encodings[0]},
    )


def _read_pdf_file(
    path: Path,
    min_chars_per_page: int,
    min_total_chars: int,
    enable_ocr: bool,
) -> DocumentContent:
    """Read and validate a PDF document using PyMuPDF with scanned detection and OCR."""
    try:
        doc = fitz.open(path)
    except Exception as exc:
        raise CorruptDocumentError(f"Failed to open PDF '{path.name}': {exc}") from exc

    page_count = len(doc)
    if page_count == 0:
        doc.close()
        raise EmptyDocumentError(f"PDF document contains 0 pages: {path.name}")

    extracted_pages: list[str] = []
    total_chars = 0
    has_images = False

    for page_idx in range(page_count):
        page = doc[page_idx]
        text = page.get_text("text")
        if len(page.get_images()) > 0:
            has_images = True
        total_chars += len(text.strip())
        if text.strip():
            extracted_pages.append(f"--- Page {page_idx + 1} ---\n{text.strip()}")

    full_text = "\n\n".join(extracted_pages).strip()
    avg_chars_per_page = total_chars / max(page_count, 1)

    # Scanned detection heuristic:
    # 1. Total characters is below absolute threshold, OR
    # 2. Average chars per page is very low AND pages contain images / graphical elements
    is_scanned = (total_chars < min_total_chars) or (
        avg_chars_per_page < min_chars_per_page and (has_images or total_chars == 0)
    )

    ocr_applied = False
    if is_scanned:
        if not enable_ocr:
            doc.close()
            raise ScannedPDFError(
                f"PDF '{path.name}' appears to be a scanned or image-only document "
                f"({total_chars} extractable digital characters across {page_count} page(s)). "
                f"OCR processing is disabled in configuration."
            )

        if not is_ocr_available():
            doc.close()
            raise ScannedPDFError(
                f"PDF '{path.name}' appears to be a scanned or image-only document "
                f"({total_chars} extractable digital characters across {page_count} page(s)). "
                "Optical Character Recognition (OCR) is required, but the Tesseract OCR engine "
                "is not installed or accessible in system PATH. "
                "Please install Tesseract or provide a digital text-based PDF."
            )

        try:
            ocr_text = _perform_ocr(doc)
            if ocr_text.strip():
                full_text = ocr_text
                ocr_applied = True
                total_chars = len(full_text)
            else:
                doc.close()
                raise ScannedPDFError(
                    f"OCR executed on scanned PDF '{path.name}', but no extractable text could be recognized."
                )
        except Exception as ocr_err:
            doc.close()
            raise ScannedPDFError(f"OCR processing failed for '{path.name}': {ocr_err}") from ocr_err

    doc.close()

    if not full_text.strip():
        raise EmptyDocumentError(f"PDF document contains no readable text: {path.name}")

    return DocumentContent(
        raw_text=full_text,
        page_count=page_count,
        source_path=str(path),
        file_type="pdf",
        char_count=len(full_text),
        is_scanned=is_scanned,
        ocr_applied=ocr_applied,
        metadata={
            "avg_chars_per_page": avg_chars_per_page,
            "has_images": has_images,
        },
    )

