"""Unit and integration tests for ChangeOrderExtractor and CLI."""

import io
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import jsonschema
import pytest

from change_order_extractor.cli import main as cli_main
from change_order_extractor.extractor import ChangeOrderExtractor
from change_order_extractor.models import (
    ChangeOrderExtractionResult,
    InternalFieldExtraction,
    RawLLMExtraction,
)

SAMPLE_RAW_EXTRACTION = RawLLMExtraction(
    project_name=InternalFieldExtraction(
        value="Riverside Medical Center Expansion",
        evidence="Riverside Medical Center Expansion",
    ),
    contract_number=InternalFieldExtraction(
        value="RMC-2026-014",
        evidence="Contract number\nRMC-2026-014",
    ),
    change_order_number=InternalFieldExtraction(
        value="CO-017",
        evidence="Change order no.\nCO-017",
    ),
    date=InternalFieldExtraction(
        value="2026-10-14",
        evidence="Date\nOctober 14, 2026",
    ),
    requested_by=InternalFieldExtraction(
        value="Elena Brooks, Project Manager",
        evidence="Requested by\nElena Brooks, Project Manager",
    ),
    description=InternalFieldExtraction(
        value="Provide and install an underground electrical duct bank along the east service drive",
        evidence="Provide and install an underground electrical duct bank along the east service drive",
    ),
    amount=InternalFieldExtraction(
        value=18500.0,
        evidence="TOTAL CHANGE ORDER AMOUNT\n+$18,500.00",
    ),
    schedule_impact_days=InternalFieldExtraction(
        value=12,
        evidence="This change requires twelve (12) additional calendar days.",
    ),
    approval_status=InternalFieldExtraction(
        value="Approved",
        evidence="OWNER APPROVAL\nAPPROVED",
    ),
)


@pytest.fixture
def sample_pdf_path() -> Path:
    return Path("samples/01_complete_approved_increase.pdf")


def test_extractor_end_to_end_mocked(sample_pdf_path: Path):
    """Test full extraction pipeline orchestrator with mocked LLM response."""
    mock_llm = MagicMock()
    mock_llm.extract.return_value = SAMPLE_RAW_EXTRACTION

    extractor = ChangeOrderExtractor(llm_client=mock_llm)
    result = extractor.process_document(sample_pdf_path)

    assert isinstance(result, ChangeOrderExtractionResult)
    assert result.project_name.value == "Riverside Medical Center Expansion"
    assert result.amount.value == 18500.0
    assert result.amount.confidence > 0.85
    assert result.schedule_impact_days.value == 12
    assert result.approval_status.value == "Approved"
    assert result.needs_review is False

    # Validate output against Draft 2020-12 schema
    schema = json.loads(Path("schema/output_schema.json").read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(result.model_dump())


def test_cli_missing_argument(monkeypatch, capsys):
    """Verify CLI returns exit code 2 and writes error to stderr when no file is passed."""
    monkeypatch.setattr(sys, "argv", ["change_order_extractor"])
    exit_code = cli_main()

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Missing required argument" in captured.err
    assert captured.out == ""


def test_cli_nonexistent_file(monkeypatch, capsys):
    """Verify CLI returns exit code 1 and writes error to stderr for non-existent files."""
    monkeypatch.setattr(sys, "argv", ["change_order_extractor", "non_existent_file.pdf"])
    exit_code = cli_main()

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "File not found" in captured.err
    assert captured.out == ""


def test_cli_successful_run(monkeypatch, capsys, sample_pdf_path: Path):
    """Verify CLI emits strictly valid JSON to stdout and nothing else on success."""
    mock_extractor = MagicMock()
    mock_result = ChangeOrderExtractionResult(
        project_name={"value": "Test Project", "confidence": 0.95},
        contract_number={"value": "CT-100", "confidence": 0.90},
        change_order_number={"value": "CO-1", "confidence": 0.92},
        date={"value": "2026-10-15", "confidence": 0.93},
        requested_by={"value": "Architect", "confidence": 0.89},
        description={"value": "Work description", "confidence": 0.94},
        amount={"value": 10000.0, "confidence": 0.96},
        schedule_impact_days={"value": 5, "confidence": 0.91},
        approval_status={"value": "Approved", "confidence": 0.95},
        needs_review=False,
    )
    mock_extractor.process_document.return_value = mock_result

    with patch("change_order_extractor.cli.ChangeOrderExtractor", return_value=mock_extractor):
        monkeypatch.setattr(sys, "argv", ["change_order_extractor", str(sample_pdf_path)])
        exit_code = cli_main()

    captured = capsys.readouterr()
    assert exit_code == 0
    # stdout must be parseable JSON without surrounding prose
    parsed = json.loads(captured.out)
    assert parsed["project_name"]["value"] == "Test Project"
    assert parsed["needs_review"] is False
