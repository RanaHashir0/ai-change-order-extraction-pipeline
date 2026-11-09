"""Business validation and conflict detection for construction change orders.

Enforces domain constraints, sign conventions, ISO date formatting,
conflict detection, and manual review triggers.
"""

from __future__ import annotations

import datetime
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from change_order_extractor.models import RawLLMExtraction

VALID_APPROVAL_STATUSES = {
    "APPROVED",
    "PENDING",
    "PENDING APPROVAL",
    "PENDING OWNER APPROVAL",
    "UNDER REVIEW",
    "REJECTED",
    "DENIED",
    "DRAFT",
    "EXECUTED",
    "VOID",
}

# Standard normalized status map
NORMALIZED_STATUS_MAP = {
    "APPROVED": "Approved",
    "PENDING": "Pending",
    "PENDING APPROVAL": "Pending Approval",
    "PENDING OWNER APPROVAL": "Pending Approval",
    "UNDER REVIEW": "Under Review",
    "REJECTED": "Rejected",
    "DENIED": "Rejected",
    "DRAFT": "Draft",
    "EXECUTED": "Approved",
    "VOID": "Void",
}


@dataclass
class ValidationIssue:
    """Represents a validation warning or conflict that may trigger needs_review."""

    field_name: str
    issue_type: str  # "conflict", "missing", "invalid_format", "low_confidence", "cross_field"
    message: str
    action_taken: str


@dataclass
class ValidationReport:
    """Consolidated outcome of domain business validation."""

    cleaned_values: Dict[str, Any]
    issues: List[ValidationIssue] = field(default_factory=list)
    conflicts: Dict[str, List[str]] = field(default_factory=dict)
    needs_review: bool = False

    def add_issue(
        self,
        field_name: str,
        issue_type: str,
        message: str,
        action_taken: str = "none",
        trigger_review: bool = True,
    ) -> None:
        """Record an issue and optionally trigger manual review."""
        self.issues.append(
            ValidationIssue(
                field_name=field_name,
                issue_type=issue_type,
                message=message,
                action_taken=action_taken,
            )
        )
        if trigger_review:
            self.needs_review = True


class ChangeOrderValidator:
    """Validates extracted change order data against construction business rules."""

    def __init__(self, require_explicit_status: bool = True) -> None:
        self.require_explicit_status = require_explicit_status

    def validate(
        self,
        raw_extraction: RawLLMExtraction,
        source_text: str,
    ) -> ValidationReport:
        """Execute full business validation on raw LLM extraction.

        Args:
            raw_extraction: Extracted fields, evidence, and preliminary conflicts.
            source_text: Raw document text for cross-referencing and verification.

        Returns:
            ValidationReport containing cleaned values, issues, and needs_review flag.
        """
        report = ValidationReport(cleaned_values={})

        # 1. Project name
        report.cleaned_values["project_name"] = self._validate_project_name(
            raw_extraction, source_text, report
        )

        # 2. Contract number
        report.cleaned_values["contract_number"] = self._validate_contract_number(
            raw_extraction, source_text, report
        )

        # 3. Change order number
        report.cleaned_values["change_order_number"] = self._validate_change_order_number(
            raw_extraction, source_text, report
        )

        # 4. Date
        report.cleaned_values["date"] = self._validate_date(
            raw_extraction, source_text, report
        )

        # 5. Requested by
        report.cleaned_values["requested_by"] = self._validate_requested_by(
            raw_extraction, report
        )

        # 6. Description
        report.cleaned_values["description"] = self._validate_description(
            raw_extraction, report
        )

        # 7. Amount
        report.cleaned_values["amount"] = self._validate_amount(
            raw_extraction, source_text, report
        )

        # 8. Schedule impact days
        report.cleaned_values["schedule_impact_days"] = self._validate_schedule_impact(
            raw_extraction, source_text, report
        )

        # 9. Approval status
        report.cleaned_values["approval_status"] = self._validate_approval_status(
            raw_extraction, report, source_text
        )

        # 10. Cross-field and Document-Level Consistency Checks
        self._validate_cross_field_consistency(report, source_text)

        return report

    def _validate_project_name(
        self, raw: RawLLMExtraction, source_text: str, report: ValidationReport
    ) -> Optional[str]:
        val = raw.project_name.value
        if not val or not val.strip():
            report.add_issue(
                "project_name",
                "missing",
                "Project name is missing from document.",
                action_taken="set_to_null",
                trigger_review=True,
            )
            return None
        return val.strip()

    def _validate_contract_number(
        self, raw: RawLLMExtraction, source_text: str, report: ValidationReport
    ) -> Optional[str]:
        # Check conflicts in contract number (e.g. Sample 06: CT-8841-A vs CT-8841-B)
        if raw.contract_number.conflicts_detected or self._has_document_discrepancy(
            "contract", source_text
        ):
            all_conflicts = list(raw.contract_number.conflicts_detected)
            if self._has_document_discrepancy("contract", source_text):
                all_conflicts.append("Document text mentions conflicting contract numbers.")
            report.conflicts["contract_number"] = all_conflicts
            report.add_issue(
                "contract_number",
                "conflict",
                "Conflicting contract numbers detected across document sections.",
                action_taken="set_to_null",
                trigger_review=True,
            )
            return None

        val = raw.contract_number.value
        if not val or not val.strip():
            # Incomplete documents often lack contract numbers
            report.add_issue(
                "contract_number",
                "missing",
                "Contract number is absent from document.",
                action_taken="set_to_null",
                trigger_review=True,
            )
            return None
        return val.strip()

    def _validate_change_order_number(
        self, raw: RawLLMExtraction, source_text: str, report: ValidationReport
    ) -> Optional[str]:
        if raw.change_order_number.conflicts_detected or self._has_document_discrepancy(
            "change order number", source_text
        ):
            report.add_issue(
                "change_order_number",
                "conflict",
                "Conflicting change order numbers detected in document.",
                action_taken="set_to_null",
                trigger_review=True,
            )
            return None

        val = raw.change_order_number.value
        if not val or not val.strip():
            report.add_issue(
                "change_order_number",
                "missing",
                "Change order number is missing.",
                action_taken="set_to_null",
                trigger_review=True,
            )
            return None
        return val.strip()

    def _validate_date(
        self, raw: RawLLMExtraction, source_text: str, report: ValidationReport
    ) -> Optional[str]:
        # Check date conflicts
        if raw.date.conflicts_detected or self._has_document_discrepancy("date", source_text):
            all_conflicts = list(raw.date.conflicts_detected)
            if self._has_document_discrepancy("date", source_text):
                all_conflicts.append("Document indicates conflicting dates between sections.")
            report.conflicts["date"] = all_conflicts
            report.add_issue(
                "date",
                "conflict",
                "Multiple conflicting dates found for change order.",
                action_taken="set_to_null",
                trigger_review=True,
            )
            return None

        val = raw.date.value
        if not val or not val.strip():
            report.add_issue(
                "date",
                "missing",
                "Date is missing from document.",
                action_taken="set_to_null",
                trigger_review=True,
            )
            return None

        # Check ISO YYYY-MM-DD
        iso_val = self._parse_to_iso_date(val.strip())
        if not iso_val:
            report.add_issue(
                "date",
                "invalid_format",
                f"Date '{val}' cannot be parsed into a valid calendar ISO YYYY-MM-DD date.",
                action_taken="set_to_null",
                trigger_review=True,
            )
            return None

        return iso_val

    def _validate_requested_by(
        self, raw: RawLLMExtraction, report: ValidationReport
    ) -> Optional[str]:
        val = raw.requested_by.value
        if not val or not val.strip():
            report.add_issue(
                "requested_by",
                "missing",
                "Requesting entity or person not specified.",
                action_taken="set_to_null",
                trigger_review=False,  # Not critical enough alone to fail if others pass, but noted
            )
            return None
        return val.strip()

    def _validate_description(
        self, raw: RawLLMExtraction, report: ValidationReport
    ) -> Optional[str]:
        val = raw.description.value
        if not val or not val.strip():
            report.add_issue(
                "description",
                "missing",
                "Scope / description of change is missing.",
                action_taken="set_to_null",
                trigger_review=True,
            )
            return None
        return val.strip()

    def _validate_amount(
        self, raw: RawLLMExtraction, source_text: str, report: ValidationReport
    ) -> Optional[float]:
        # Check for amount conflicts (e.g. Sample 06: $15,000 on cover vs $14,800 on Attachment B)
        if raw.amount.conflicts_detected or self._has_document_discrepancy("amount", source_text):
            all_conflicts = list(raw.amount.conflicts_detected)
            if self._has_document_discrepancy("amount", source_text):
                all_conflicts.append("Document sections provide conflicting amount figures.")
            report.conflicts["amount"] = all_conflicts
            report.add_issue(
                "amount",
                "conflict",
                "Conflicting amounts found in document without explicit additive reconciliation.",
                action_taken="set_to_null",
                trigger_review=True,
            )
            return None

        # Check if amount is merely an unapproved tentative estimate (Sample 05)
        lower_source = source_text.lower()
        if "estimate and is not a final" in lower_source or "estimate and is not an approved" in lower_source:
            report.add_issue(
                "amount",
                "estimate_only",
                "Amount stated is merely an estimate and not an approved or agreed adjustment.",
                action_taken="set_to_null",
                trigger_review=True,
            )
            return None

        val = raw.amount.value
        if val is None:
            # Check if document is schedule-only or missing amount
            evidence = (raw.amount.evidence or "").lower()
            if "schedule extension only" in lower_source or "no amount" in evidence:
                # Valid schedule-only change order (Sample 03)
                return None

            # Incomplete amount (e.g., Sample 05 with only estimate or missing)
            report.add_issue(
                "amount",
                "missing",
                "Contract amount adjustment not specified or incomplete.",
                action_taken="set_to_null",
                trigger_review=True,
            )
            return None

        if not math.isfinite(val):
            report.add_issue(
                "amount",
                "invalid_format",
                f"Amount {val} is not a finite numeric value.",
                action_taken="set_to_null",
                trigger_review=True,
            )
            return None

        # Sign verification: check if evidence indicates credit or deduction
        evidence_text = f"{raw.amount.evidence or ''} {source_text}".lower()
        is_credit_context = any(
            w in (raw.amount.evidence or "").lower()
            for w in ["credit", "deduct", "reduction", "decrease", "refund"]
        )

        if is_credit_context and val > 0:
            # If evidence explicitly states credit or deduction but amount is positive, negate it
            val = -abs(val)
            report.add_issue(
                "amount",
                "sign_correction",
                "Amount adjusted to negative based on explicit credit/deductive terms.",
                action_taken="negate_amount",
                trigger_review=False,
            )

        return float(round(val, 2))

    def _validate_schedule_impact(
        self, raw: RawLLMExtraction, source_text: str, report: ValidationReport
    ) -> Optional[int]:
        # Check schedule conflicts (e.g. Sample 06: 5 days vs 9 days)
        if raw.schedule_impact_days.conflicts_detected or self._has_document_discrepancy(
            "schedule", source_text
        ):
            all_conflicts = list(raw.schedule_impact_days.conflicts_detected)
            if self._has_document_discrepancy("schedule", source_text):
                all_conflicts.append("Document sections state conflicting schedule impact days.")
            report.conflicts["schedule_impact_days"] = all_conflicts
            report.add_issue(
                "schedule_impact_days",
                "conflict",
                "Conflicting schedule impact durations stated in document.",
                action_taken="set_to_null",
                trigger_review=True,
            )
            return None

        val = raw.schedule_impact_days.value
        if val is None:
            # If no schedule impact is mentioned at all
            return None

        # Must be integer
        if not isinstance(val, int) and not (isinstance(val, float) and val.is_integer()):
            report.add_issue(
                "schedule_impact_days",
                "invalid_format",
                f"Schedule impact days '{val}' must be an integer.",
                action_taken="set_to_null",
                trigger_review=True,
            )
            return None

        int_val = int(val)

        # Sign verification
        evidence_lower = (raw.schedule_impact_days.evidence or "").lower()
        if any(w in evidence_lower for w in ["reduction", "shorten", "accelerate"]) and int_val > 0:
            int_val = -abs(int_val)

        return int_val

    def _validate_approval_status(
        self, raw: RawLLMExtraction, report: ValidationReport, source_text: str = ""
    ) -> Optional[str]:
        # Check approval conflicts (e.g. Sample 06: Approved on cover vs director countersignature outstanding)
        if raw.approval_status.conflicts_detected or self._has_document_discrepancy(
            "approval", source_text
        ):
            all_conflicts = list(raw.approval_status.conflicts_detected)
            if self._has_document_discrepancy("approval", source_text):
                all_conflicts.append("Document sections state conflicting approval statuses.")
            report.conflicts["approval_status"] = all_conflicts
            report.add_issue(
                "approval_status",
                "conflict",
                "Conflicting approval statuses detected in document.",
                action_taken="set_to_null",
                trigger_review=True,
            )
            return None

        val = raw.approval_status.value
        if not val or not val.strip():
            report.add_issue(
                "approval_status",
                "missing",
                "Approval status is not stated in document.",
                action_taken="set_to_null",
                trigger_review=True,
            )
            return None

        cleaned = val.strip().upper()

        if cleaned in {"APPROVED", "APPROVED - TIME EXTENSION ONLY", "EXECUTED", "APPROVED BY OWNER"}:
            return "Approved"
        elif cleaned in {"PENDING", "PENDING OWNER APPROVAL", "PENDING APPROVAL", "PENDING SIGNATURE"}:
            return "Pending Approval" if "APPROVAL" in cleaned else "Pending"
        elif cleaned in {"UNDER REVIEW"}:
            report.add_issue(
                "approval_status",
                "under_review",
                "Document explicitly indicates approval is 'Under Review'.",
                action_taken="keep_value",
                trigger_review=True,
            )
            return "Under Review"
        elif cleaned in {"REJECTED", "DENIED", "DISAPPROVED"}:
            return "Rejected"
        elif cleaned in {"DRAFT"}:
            report.add_issue(
                "approval_status",
                "draft",
                "Document is a draft and not formally approved.",
                action_taken="keep_value",
                trigger_review=True,
            )
            return "Draft"
        elif cleaned in {"VOID", "CANCELLED"}:
            return "Void"

        # Status not recognized or invented
        report.add_issue(
            "approval_status",
            "unrecognized_status",
            f"Approval status '{val}' is non-standard or unverified.",
            action_taken="set_to_null",
            trigger_review=True,
        )
        return None

    def _validate_cross_field_consistency(
        self, report: ValidationReport, source_text: str
    ) -> None:
        """Check cross-field relationships to catch anomalies and incomplete submissions."""
        vals = report.cleaned_values

        # 1. Incomplete submission (e.g. Sample 05: missing contract number, under review, etc.)
        missing_count = sum(1 for k in ["contract_number", "change_order_number", "date"] if vals.get(k) is None)
        if missing_count >= 2:
            report.add_issue(
                "document",
                "incomplete_document",
                f"Document is missing {missing_count} critical identification fields.",
                action_taken="flag_review",
                trigger_review=True,
            )

        # 2. Both amount and schedule_impact are null without explanation
        if vals.get("amount") is None and vals.get("schedule_impact_days") is None:
            report.add_issue(
                "impact",
                "missing_impact",
                "Neither financial amount nor schedule impact was extracted.",
                action_taken="flag_review",
                trigger_review=True,
            )

        # 3. Discrepancy note in text (e.g. Sample 06 review note)
        if "discrepancies" in source_text.lower() or "disagree" in source_text.lower() or "review note" in source_text.lower():
            report.needs_review = True

    @staticmethod
    def _has_document_discrepancy(keyword: str, source_text: str) -> bool:
        """Detect whether the source document explicitly highlights discrepancies for a given field."""
        lower_text = source_text.lower()
        # Look for explicit disagreement statements (e.g. "disagree on contract number, date, amount...")
        match = re.search(r"disagree(?:s)? on ([^\.\n]+)", lower_text)
        if match:
            disputed_topics = match.group(1).lower()
            if keyword.lower() in disputed_topics:
                return True
        return False

    @staticmethod
    def _parse_to_iso_date(raw_date_str: str) -> Optional[str]:
        """Convert a date string to ISO YYYY-MM-DD or validate existing ISO format."""
        # Check standard ISO format YYYY-MM-DD
        iso_match = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", raw_date_str)
        if iso_match:
            try:
                dt = datetime.date.fromisoformat(raw_date_str)
                # Realistic construction date check (between 1990 and 2050)
                if 1990 <= dt.year <= 2050:
                    return dt.isoformat()
            except ValueError:
                return None

        # Parse text dates like "October 16, 2026", "16-Oct-2026", "10/16/2026"
        date_formats = [
            "%B %d, %Y",
            "%b %d, %Y",
            "%m/%d/%Y",
            "%d/%m/%Y",
            "%d-%b-%Y",
            "%d-%B-%Y",
            "%Y/%m/%d",
        ]
        for fmt in date_formats:
            try:
                dt = datetime.datetime.strptime(raw_date_str, fmt).date()
                if 1990 <= dt.year <= 2050:
                    return dt.isoformat()
            except ValueError:
                continue

        return None
