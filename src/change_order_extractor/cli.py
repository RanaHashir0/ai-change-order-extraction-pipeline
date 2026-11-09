"""Command-line interface for the Change Order Extraction Pipeline.

Prints strictly one validated JSON object to stdout.
Directs all diagnostic messages, progress logs, and errors to stderr.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from change_order_extractor.extractor import ChangeOrderExtractor
from change_order_extractor.llm_client import LLMClientError
from change_order_extractor.pdf_reader import DocumentReaderError


def setup_stderr_logging(verbose: bool = False) -> None:
    """Configure root logger to emit exclusively to stderr."""
    level = logging.DEBUG if verbose else logging.WARNING
    handler = logging.StreamHandler(sys.stderr)
    formatter = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    # Remove existing handlers to prevent duplicate output
    root_logger.handlers = [handler]


def main() -> int:
    """Main CLI entry point.

    Returns:
        0 on successful extraction and valid JSON emission, non-zero on failure.
    """
    parser = argparse.ArgumentParser(
        description="Extract structured change order fields from PDF or plain text construction documents."
    )
    parser.add_argument(
        "file",
        nargs="?",
        default=None,
        help="Path to the PDF or TXT change order document.",
    )
    parser.add_argument(
        "--file",
        dest="file_opt",
        default=None,
        help="Alternative flag to specify target document path.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable detailed diagnostic logs emitted to stderr.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        help="Confidence threshold for triggering manual review (default: 0.85).",
    )
    parser.add_argument(
        "--no-ocr",
        action="store_true",
        help="Disable optical character recognition fallback.",
    )

    args = parser.parse_args()
    target_path = args.file_opt or args.file

    setup_stderr_logging(verbose=args.verbose)

    if not target_path:
        sys.stderr.write("Error: Missing required argument. Please specify a document path.\n")
        sys.stderr.write("Usage: python -m change_order_extractor.cli <path_to_file>\n")
        sys.stderr.write("   or: make extract FILE=<path_to_file>\n")
        return 2

    path = Path(target_path)
    if not path.exists():
        sys.stderr.write(f"Error: File not found at '{path}'.\n")
        return 1

    try:
        extractor = ChangeOrderExtractor(
            confidence_threshold=args.threshold,
        )
        result = extractor.process_document(
            file_path=path,
            enable_ocr=not args.no_ocr,
        )

        # Output strictly valid JSON to stdout
        output_json = result.model_dump_json(indent=2)
        sys.stdout.write(output_json + "\n")
        sys.stdout.flush()
        return 0

    except (DocumentReaderError, FileNotFoundError) as doc_err:
        sys.stderr.write(f"Document Processing Error: {doc_err}\n")
        return 1
    except LLMClientError as llm_err:
        sys.stderr.write(f"LLM Extraction Error: {llm_err}\n")
        return 1
    except Exception as exc:
        sys.stderr.write(f"Fatal Unexpected Pipeline Error: {exc}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
