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


def test_validator_estimated_amount_extracts_value_and_flags_review(sample_valid_raw):
    """Test that explicit estimated amount extracts 6200.0 instead of null and triggers needs_review."""
    sample_valid_raw.amount = InternalFieldExtraction(
        value=6200.0,
        evidence="Estimated amount\nADD $6,200.00 to contract value",
        reasoning="Amount shown is an estimate and is not a final approved adjustment",
    )

    validator = ChangeOrderValidator()
    doc_text = "Estimated amount: ADD $6,200.00 to contract value. The amount shown is an estimate and is not a final approved adjustment."
    report = validator.validate(sample_valid_raw, doc_text)

    assert report.cleaned_values["amount"] == 6200.0
    assert report.needs_review is True
    assert any(iss.issue_type == "estimate_only" for iss in report.issues)


def test_validator_unsigned_document_returns_null_status(sample_valid_raw):
    """Test that an unsigned document with no explicit status returns null, never inferring Pending."""
    sample_valid_raw.approval_status = InternalFieldExtraction(
        value="Awaiting signature",
        evidence="Signature block: Awaiting signature",
    )

    validator = ChangeOrderValidator()
    report = validator.validate(sample_valid_raw, "Change order document with no explicit status stated.")

    assert report.cleaned_values["approval_status"] is None
    assert report.needs_review is True


def test_validator_distinguish_change_order_date_from_attachment_prep_date(sample_valid_raw):
    """Test that attachment preparation date is not treated as contradiction unless document establishes conflict."""
    # 1. Without explicit document conflict: attachment prep date is a different semantic field -> date preserved
    sample_valid_raw.date = InternalFieldExtraction(
        value="2026-11-04",
        conflicts_detected=["Attachment preparation date: November 6, 2026"],
    )
    validator = ChangeOrderValidator()
    normal_doc_text = "Cover sheet Date: November 4, 2026. Attachment B Prepared date: November 6, 2026."
    report = validator.validate(sample_valid_raw, normal_doc_text)
    assert report.cleaned_values["date"] == "2026-11-04"

    # 2. With explicit conflict (e.g. Document 06): review note establishes disagreement -> date is set to null
    doc_06_text = (
        "Cover sheet Date: November 4, 2026. Attachment B Prepared date: November 6, 2026. "
        "REVIEW NOTE: The cover sheet and Attachment B disagree on contract number, date, amount, schedule impact, and final approval."
    )
    report_conflict = validator.validate(sample_valid_raw, doc_06_text)
    assert report_conflict.cleaned_values["date"] is None
    assert report_conflict.needs_review is True


def test_validator_synonymous_approval_status_does_not_conflict(sample_valid_raw):
    """Test that complementary review mentions ('Submitted for review') do not cancel 'Under Review'."""
    sample_valid_raw.approval_status = InternalFieldExtraction(
        value="Under Review",
        evidence="STATUS: UNDER REVIEW",
        conflicts_detected=["Submitted for review - no approval recorded"],
    )

    validator = ChangeOrderValidator()
    doc_text = "Approval status: Submitted for review - no approval recorded. STATUS: UNDER REVIEW"
    report = validator.validate(sample_valid_raw, doc_text)

    assert report.cleaned_values["approval_status"] == "Under Review"
    assert "approval_status" not in report.conflicts


def test_validator_preserves_requester_with_professional_designation(sample_valid_raw):
    """Test that requested_by preserves full name and professional designation when present in text."""
    sample_valid_raw.requested_by = InternalFieldExtraction(
        value="Jane Doe",
        evidence="From: Jane Doe, Project Architect",
    )
    validator = ChangeOrderValidator()
    doc_text = "From: Jane Doe, Project Architect\nProject: Highland School"
    report = validator.validate(sample_valid_raw, doc_text)

    assert report.cleaned_values["requested_by"] == "Jane Doe, Project Architect"


def test_validator_preserves_specific_scope_description(sample_valid_raw):
    """Test that a description referencing specific materials/hazards is accepted without truncation."""
    specific_desc = (
        "During demolition in the west classroom wing, the contractor discovered "
        "concealed asbestos-containing floor adhesive beneath the existing vinyl tile. "
        "Safe abatement, air-clearance testing, and replacement of the floor finish required."
    )
    sample_valid_raw.description = InternalFieldExtraction(
        value=specific_desc,
        evidence="During demolition in the west classroom wing, the contractor discovered concealed asbestos-containing floor adhesive",
    )
    validator = ChangeOrderValidator()
    doc_text = (
        "RE: Request for additional contract time due to concealed conditions\n"
        "During demolition in the west classroom wing, the contractor discovered "
        "concealed asbestos-containing floor adhesive beneath the existing vinyl tile."
    )
    report = validator.validate(sample_valid_raw, doc_text)

    # Full description including the hazard detail must be preserved
    assert "asbestos-containing floor adhesive" in (report.cleaned_values["description"] or "")
    assert report.needs_review is False


def test_validator_normalizes_unicode_hyphens_in_description(sample_valid_raw):
    """Test that Unicode non-breaking hyphens (\u2011) and dashes in description are normalized to standard ASCII hyphens."""
    sample_valid_raw.description = InternalFieldExtraction(
        value="concealed asbestos\u2011containing floor adhesive beneath existing vinyl\u2011tile",
        evidence="concealed asbestos-containing floor adhesive",
    )
    validator = ChangeOrderValidator()
    report = validator.validate(
        sample_valid_raw, "source text with asbestos-containing floor adhesive"
    )
    assert (
        report.cleaned_values["description"]
        == "concealed asbestos-containing floor adhesive beneath existing vinyl-tile"
    )
    assert "\u2011" not in report.cleaned_values["description"]
    assert "asbestos-containing floor adhesive" in report.cleaned_values["description"]


def test_validator_keeps_existing_full_requester_designation(sample_valid_raw):
    """Test that requested_by keeps already-extracted designation intact."""
    sample_valid_raw.requested_by = InternalFieldExtraction(
        value="Jane Doe, Project Architect",
        evidence="From: Jane Doe, Project Architect",
    )
    validator = ChangeOrderValidator()
    doc_text = "From: Jane Doe, Project Architect\nProject: Highland School"
    report = validator.validate(sample_valid_raw, doc_text)

    assert report.cleaned_values["requested_by"] == "Jane Doe, Project Architect"



