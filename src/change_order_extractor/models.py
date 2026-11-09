"""Data models for change order extraction.

Defines both internal extraction models (capturing evidence, confidence assessments,
and detected conflicts) and public output models conforming to Draft 2020-12 JSON Schema.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Generic, List, Optional, TypeVar
from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


class InternalFieldExtraction(BaseModel, Generic[T]):
    """Internal representation of an extracted field with supporting evidence."""

    value: Optional[T] = Field(
        default=None,
        description="The extracted value, or null if absent or indeterminate.",
    )
    evidence: Optional[str] = Field(
        default=None,
        description="Verbatim text quote from the document supporting the extraction.",
    )
    confidence_score: Optional[float] = Field(
        default=None,
        description="Initial confidence estimate (0.0 to 1.0) based on textual explicitness.",
    )
    conflicts_detected: List[str] = Field(
        default_factory=list,
        description="List of conflicting values or discrepancies found in the document.",
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Reasoning or notes regarding ambiguity, absence, or extraction logic.",
    )


class RawLLMExtraction(BaseModel):
    """Internal model returned by the LLM containing all 9 fields plus evidence."""

    project_name: InternalFieldExtraction[str] = Field(
        default_factory=InternalFieldExtraction,
        description="Formal name of the construction project.",
    )
    contract_number: InternalFieldExtraction[str] = Field(
        default_factory=InternalFieldExtraction,
        description="Prime contract or agreement identifier.",
    )
    change_order_number: InternalFieldExtraction[str] = Field(
        default_factory=InternalFieldExtraction,
        description="Change order designation or number (e.g. CO-01, #3).",
    )
    date: InternalFieldExtraction[str] = Field(
        default_factory=InternalFieldExtraction,
        description="Date of the change order (ISO YYYY-MM-DD preferred if unambiguous).",
    )
    requested_by: InternalFieldExtraction[str] = Field(
        default_factory=InternalFieldExtraction,
        description="Entity, contractor, architect, or person requesting the change.",
    )
    description: InternalFieldExtraction[str] = Field(
        default_factory=InternalFieldExtraction,
        description="Scope or summary of work included in the change order.",
    )
    amount: InternalFieldExtraction[float] = Field(
        default_factory=InternalFieldExtraction,
        description="Net dollar impact. Positive for price increase, negative for credit/decrease.",
    )
    schedule_impact_days: InternalFieldExtraction[int] = Field(
        default_factory=InternalFieldExtraction,
        description="Net schedule impact in integer days. Positive for extensions, negative for reductions.",
    )
    approval_status: InternalFieldExtraction[str] = Field(
        default_factory=InternalFieldExtraction,
        description="Document approval state (e.g. Approved, Pending, Rejected, Draft).",
    )
    document_summary: Optional[str] = Field(
        default=None,
        description="High-level summary of the document context.",
    )
    unresolved_ambiguities: List[str] = Field(
        default_factory=list,
        description="General document ambiguities or missing critical elements.",
    )


# --- Public Output Models ---


class FieldOutput(BaseModel, Generic[T]):
    """Standardized field output structure containing value and confidence."""

    model_config = ConfigDict(extra="forbid")

    value: Optional[T] = Field(
        default=None,
        description="Extracted value, or null when absent, invalid, or conflicting.",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0 reflecting extraction certainty.",
    )


class ChangeOrderExtractionResult(BaseModel):
    """Complete public output model for construction change order extraction."""

    model_config = ConfigDict(extra="forbid")

    project_name: FieldOutput[str] = Field(
        description="Formal name of the construction project."
    )
    contract_number: FieldOutput[str] = Field(
        description="Contract or agreement reference number."
    )
    change_order_number: FieldOutput[str] = Field(
        description="Change order identifier or number."
    )
    date: FieldOutput[str] = Field(
        description="Date of change order formatted as ISO YYYY-MM-DD, or null."
    )
    requested_by: FieldOutput[str] = Field(
        description="Requesting party, contractor, architect, or engineer."
    )
    description: FieldOutput[str] = Field(
        description="Detailed description or scope of the requested change."
    )
    amount: FieldOutput[float] = Field(
        description="Net financial amount (positive for increase, negative for decrease)."
    )
    schedule_impact_days: FieldOutput[int] = Field(
        description="Net schedule impact in days (positive for extension, negative for reduction)."
    )
    approval_status: FieldOutput[str] = Field(
        description="Document approval status (e.g. Approved, Pending, Rejected)."
    )
    needs_review: bool = Field(
        description="Flag set to true if information is missing, ambiguous, conflicting, or low confidence."
    )


def export_json_schema(output_path: Path | str = "schema/output_schema.json") -> dict[str, Any]:
    """Generate and save JSON Schema Draft 2020-12 for ChangeOrderExtractionResult."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    schema = ChangeOrderExtractionResult.model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["title"] = "ChangeOrderExtractionResult"
    schema["description"] = (
        "JSON Schema Draft 2020-12 for AI-Powered Construction Change Order Extraction"
    )

    with open(path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)

    return schema
