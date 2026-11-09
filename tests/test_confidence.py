"""Unit tests for the ConfidenceScorer module."""

import pytest

from change_order_extractor.confidence import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    ConfidenceScorer,
)
from change_order_extractor.models import (
    InternalFieldExtraction,
    RawLLMExtraction,
)
from change_order_extractor.validator import ValidationReport


@pytest.fixture
def clean_document_text() -> str:
    return (
        "Riverside Medical Center Expansion\n"
        "Contract No: CT-2024-001\n"
        "CHANGE ORDER: CO-001\n"
        "Date: October 15, 2026\n"
        "Requested by: Apex Architecture\n"
        "Scope: Additional conduit runs for MRI suite\n"
        "Total Change Order Amount: $24,750.00\n"
        "Schedule Impact: 12 additional calendar days\n"
        "Status: APPROVED by Owner\n"
    )


@pytest.fixture
def sample_raw_extraction() -> RawLLMExtraction:
    return RawLLMExtraction(
        project_name=InternalFieldExtraction(
            value="Riverside Medical Center Expansion",
            evidence="Riverside Medical Center Expansion",
        ),
        contract_number=InternalFieldExtraction(
            value="CT-2024-001",
            evidence="Contract No: CT-2024-001",
        ),
        change_order_number=InternalFieldExtraction(
            value="CO-001",
            evidence="CHANGE ORDER: CO-001",
        ),
        date=InternalFieldExtraction(
            value="2026-10-15",
            evidence="Date: October 15, 2026",
        ),
        requested_by=InternalFieldExtraction(
            value="Apex Architecture",
            evidence="Requested by: Apex Architecture",
        ),
        description=InternalFieldExtraction(
            value="Additional conduit runs for MRI suite",
            evidence="Scope: Additional conduit runs for MRI suite",
        ),
        amount=InternalFieldExtraction(
            value=24750.0,
            evidence="Total Change Order Amount: $24,750.00",
        ),
        schedule_impact_days=InternalFieldExtraction(
            value=12,
            evidence="Schedule Impact: 12 additional calendar days",
        ),
        approval_status=InternalFieldExtraction(
            value="Approved",
            evidence="Status: APPROVED by Owner",
        ),
    )


def test_high_confidence_clean_extraction(sample_raw_extraction, clean_document_text):
    """Test that explicit fields with verified quotes receive high confidence scores (> 0.90)."""
    scorer = ConfidenceScorer(threshold=0.85)

    report = ValidationReport(
        cleaned_values={
            "project_name": "Riverside Medical Center Expansion",
            "contract_number": "CT-2024-001",
            "change_order_number": "CO-001",
            "date": "2026-10-15",
            "requested_by": "Apex Architecture",
            "description": "Additional conduit runs for MRI suite",
            "amount": 24750.0,
            "schedule_impact_days": 12,
            "approval_status": "Approved",
        },
        needs_review=False,
    )

    result = scorer.evaluate(sample_raw_extraction, report, clean_document_text)

    assert result.needs_review is False
    assert result.mean_confidence > 0.90
    for field_name, score_breakdown in result.field_scores.items():
        assert score_breakdown.final_confidence >= 0.85
        assert score_breakdown.evidence_grounded == 1.0


def test_low_confidence_when_evidence_not_in_document(sample_raw_extraction, clean_document_text):
    """Test that hallucinated evidence (not found in document) drastically lowers confidence."""
    scorer = ConfidenceScorer(threshold=0.85)

    # Corrupt the evidence to something not present in the document
    sample_raw_extraction.amount.evidence = "Fabricated price note never written in contract: $999,999"

    report = ValidationReport(
        cleaned_values={"amount": 24750.0},
        needs_review=False,
    )

    result = scorer.evaluate(sample_raw_extraction, report, clean_document_text)
    amount_score = result.field_scores["amount"]

    assert amount_score.evidence_grounded < 0.5
    assert amount_score.final_confidence < 0.85
    assert "amount" in result.low_confidence_fields
    assert result.needs_review is True  # Critical field low confidence triggers review


def test_conflicting_field_receives_zero_confidence(sample_raw_extraction, clean_document_text):
    """Test that a field with detected contradictions drops to 0.0 confidence."""
    scorer = ConfidenceScorer()

    report = ValidationReport(
        cleaned_values={"amount": None},  # validator set conflicting amount to null
        conflicts={"amount": ["$15,000 on cover", "$14,800 on sheet"]},
        needs_review=True,
    )

    result = scorer.evaluate(sample_raw_extraction, report, clean_document_text)
    amount_score = result.field_scores["amount"]

    assert amount_score.final_confidence == 0.0
    assert amount_score.conflict_penalty == 0.0
    assert result.needs_review is True


def test_missing_field_absence_confidence(clean_document_text):
    """Test that confidence in absence reflects document quality and coverage, distinct from value confidence."""
    scorer = ConfidenceScorer()

    empty_raw = RawLLMExtraction()
    report = ValidationReport(
        cleaned_values={},  # All fields absent
        needs_review=True,
    )

    result = scorer.evaluate(empty_raw, report, clean_document_text)

    for field_name, score_breakdown in result.field_scores.items():
        assert not score_breakdown.is_present
        # Confidence in absence is high because the document is clean and legible
        assert score_breakdown.final_confidence > 0.80
        assert "absent" in score_breakdown.scoring_notes.lower()


def test_ocr_quality_discount():
    """Test that documents parsed with OCR receive an appropriate quality calibration factor."""
    scorer = ConfidenceScorer(ocr_quality_discount=0.88)

    doc_text = "Clean OCR text from scanned change order with high legibility."
    q_digital = scorer._calculate_doc_quality(doc_text, is_ocr=False)
    q_ocr = scorer._calculate_doc_quality(doc_text, is_ocr=True)

    assert q_digital == 1.0
    assert q_ocr == 0.88
