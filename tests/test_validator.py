"""Unit tests for ChangeOrderValidator and JSON Schema Draft 2020-12 compliance."""

import json
from pathlib import Path

import jsonschema
import pytest

from change_order_extractor.models import (
    ChangeOrderExtractionResult,
    FieldOutput,
    InternalFieldExtraction,
    RawLLMExtraction,
    export_json_schema,
)
from change_order_extractor.validator import ChangeOrderValidator, ValidationReport


@pytest.fixture
def sample_valid_raw() -> RawLLMExtraction:
    return RawLLMExtraction(
        project_name=InternalFieldExtraction(value="Riverside Medical Center", evidence="Riverside Medical Center"),
        contract_number=InternalFieldExtraction(value="CT-2024-001", evidence="Contract: CT-2024-001"),
        change_order_number=InternalFieldExtraction(value="CO-001", evidence="Change Order CO-001"),
        date=InternalFieldExtraction(value="2026-10-15", evidence="Date: 2026-10-15"),
        requested_by=InternalFieldExtraction(value="Apex Architecture", evidence="Requested by Apex Architecture"),
        description=InternalFieldExtraction(value="Additional conduit runs for MRI suite", evidence="Scope: MRI conduit"),
        amount=InternalFieldExtraction(value=24750.0, evidence="Total Increase: $24,750.00"),
        schedule_impact_days=InternalFieldExtraction(value=12, evidence="12 additional calendar days"),
        approval_status=InternalFieldExtraction(value="Approved", evidence="Status: APPROVED"),
    )


def test_validator_clean_complete_approved(sample_valid_raw):
    """Test standard approved positive change order passes with needs_review=False."""
    validator = ChangeOrderValidator()
    report = validator.validate(sample_valid_raw, "Sample document text without conflicts.")

    assert report.cleaned_values["project_name"] == "Riverside Medical Center"
    assert report.cleaned_values["contract_number"] == "CT-2024-001"
    assert report.cleaned_values["change_order_number"] == "CO-001"
    assert report.cleaned_values["date"] == "2026-10-15"
    assert report.cleaned_values["amount"] == 24750.0
    assert report.cleaned_values["schedule_impact_days"] == 12
    assert report.cleaned_values["approval_status"] == "Approved"
    assert report.needs_review is False


def test_validator_deductive_price_decrease(sample_valid_raw):
    """Test credit/deductive price change validates correctly as negative."""
    sample_valid_raw.amount = InternalFieldExtraction(
        value=-6800.0,
        evidence="Original contract scope credit: -$6,800.00",
    )
    sample_valid_raw.schedule_impact_days = InternalFieldExtraction(
        value=0,
        evidence="Schedule impact is zero (0) calendar days.",
    )

    validator = ChangeOrderValidator()
    report = validator.validate(sample_valid_raw, "Deductive change order scope credit.")

    assert report.cleaned_values["amount"] == -6800.0
    assert report.cleaned_values["schedule_impact_days"] == 0
    assert report.needs_review is False


def test_validator_credit_sign_auto_correction(sample_valid_raw):
    """Test that positive amount is negated if evidence explicitly specifies credit."""
    sample_valid_raw.amount = InternalFieldExtraction(
        value=5000.0,  # incorrectly positive from LLM
        evidence="Owner credit of $5,000 for deleted millwork",
    )

    validator = ChangeOrderValidator()
    report = validator.validate(sample_valid_raw, "Credit of $5,000 for deleted millwork.")

    assert report.cleaned_values["amount"] == -5000.0
    assert report.needs_review is False


def test_validator_schedule_only_extension(sample_valid_raw):
    """Test schedule extension with no amount specified."""
    sample_valid_raw.amount = InternalFieldExtraction(
        value=None,
        evidence="No amount or price adjustment is stated in this document.",
    )
    sample_valid_raw.schedule_impact_days = InternalFieldExtraction(
        value=18,
        evidence="Extension of eighteen (18) calendar days.",
    )

    validator = ChangeOrderValidator()
    doc_text = "This request is for a schedule extension only. No amount or price adjustment is stated."
    report = validator.validate(sample_valid_raw, doc_text)

    assert report.cleaned_values["amount"] is None
    assert report.cleaned_values["schedule_impact_days"] == 18
    assert report.needs_review is False


def test_validator_conflicting_amounts_resets_to_null_and_flags_review(sample_valid_raw):
    """Test conflicting amounts are set to null and trigger needs_review."""
    sample_valid_raw.amount = InternalFieldExtraction(
        value=15000.0,
        conflicts_detected=["Cover sheet: $15,000", "Attachment B: $14,800"],
    )

    validator = ChangeOrderValidator()
    report = validator.validate(sample_valid_raw, "Document has discrepancies on amount.")

    assert report.cleaned_values["amount"] is None
    assert report.needs_review is True
    assert "amount" in report.conflicts


def test_validator_conflicting_dates_resets_to_null_and_flags_review(sample_valid_raw):
    """Test conflicting dates are set to null and trigger needs_review."""
    sample_valid_raw.date = InternalFieldExtraction(
        value="2026-11-05",
        conflicts_detected=["Stamp date: Nov 5, 2026", "Attachment date: Nov 7, 2026"],
    )

    validator = ChangeOrderValidator()
    report = validator.validate(sample_valid_raw, "Document has discrepancies on date.")

    assert report.cleaned_values["date"] is None
    assert report.needs_review is True
    assert "date" in report.conflicts


def test_validator_invalid_calendar_date(sample_valid_raw):
    """Test invalid calendar date (e.g. Feb 31) triggers review and returns null."""
    sample_valid_raw.date = InternalFieldExtraction(value="2026-02-31")

    validator = ChangeOrderValidator()
    report = validator.validate(sample_valid_raw, "Contract text.")

    assert report.cleaned_values["date"] is None
    assert report.needs_review is True


def test_validator_unrecognized_approval_status(sample_valid_raw):
    """Test invented or unrecognized approval status is rejected."""
    sample_valid_raw.approval_status = InternalFieldExtraction(value="Maybe Approved Soon")

    validator = ChangeOrderValidator()
    report = validator.validate(sample_valid_raw, "Contract text.")

    assert report.cleaned_values["approval_status"] is None
    assert report.needs_review is True


def test_json_schema_draft_2020_12_validation(tmp_path: Path):
    """Test that exported schema is Draft 2020-12 and validates compliant output objects."""
    schema_path = tmp_path / "test_schema.json"
    schema = export_json_schema(schema_path)

    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"

    validator_cls = jsonschema.Draft202012Validator
    validator_cls.check_schema(schema)
    v = validator_cls(schema)

    # Valid result instance
    valid_payload = {
        "project_name": {"value": "Riverside Medical Center", "confidence": 0.95},
        "contract_number": {"value": "CT-2024-001", "confidence": 0.90},
        "change_order_number": {"value": "CO-001", "confidence": 0.95},
        "date": {"value": "2026-10-15", "confidence": 0.92},
        "requested_by": {"value": "Apex Architecture", "confidence": 0.88},
        "description": {"value": "Conduit work", "confidence": 0.93},
        "amount": {"value": 24750.0, "confidence": 0.97},
        "schedule_impact_days": {"value": 12, "confidence": 0.91},
        "approval_status": {"value": "Approved", "confidence": 0.96},
        "needs_review": False,
    }

    # Should validate without exceptions
    v.validate(valid_payload)

    # Pydantic validation
    pydantic_res = ChangeOrderExtractionResult.model_validate(valid_payload)
    assert pydantic_res.amount.value == 24750.0
    assert pydantic_res.needs_review is False

    # Invalid instance (extra unexpected field)
    invalid_payload = dict(valid_payload)
    invalid_payload["unexpected_internal_field"] = "bad"
    with pytest.raises(jsonschema.ValidationError):
        v.validate(invalid_payload)

