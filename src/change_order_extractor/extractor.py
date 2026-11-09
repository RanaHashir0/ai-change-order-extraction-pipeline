"""Extraction pipeline orchestrator.

Coordinates document reading, LLM information extraction, business validation,
and confidence assessment into a unified pipeline producing validated JSON output.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

from change_order_extractor.confidence import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    ConfidenceScorer,
)
from change_order_extractor.llm_client import ChangeOrderLLMClient
from change_order_extractor.models import (
    ChangeOrderExtractionResult,
    FieldOutput,
    RawLLMExtraction,
)
from change_order_extractor.pdf_reader import DocumentContent, read_document
from change_order_extractor.validator import ChangeOrderValidator

logger = logging.getLogger(__name__)


class ChangeOrderExtractor:
    """End-to-end pipeline orchestrator for construction change order extraction."""

    def __init__(
        self,
        llm_client: Optional[ChangeOrderLLMClient] = None,
        validator: Optional[ChangeOrderValidator] = None,
        confidence_scorer: Optional[ConfidenceScorer] = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> None:
        """Initialize pipeline components.

        Args:
            llm_client: Configured LLM client instance (or created lazily).
            validator: Business rule validator instance.
            confidence_scorer: Multi-factor confidence scoring instance.
            confidence_threshold: Cutoff score triggering manual review.
        """
        self._llm_client = llm_client
        self.validator = validator or ChangeOrderValidator()
        self.confidence_scorer = confidence_scorer or ConfidenceScorer(
            threshold=confidence_threshold
        )

    @property
    def llm_client(self) -> ChangeOrderLLMClient:
        """Lazy loader for LLM client to allow offline testing when credentials aren't loaded."""
        if self._llm_client is None:
            self._llm_client = ChangeOrderLLMClient()
        return self._llm_client

    def process_document(
        self,
        file_path: Union[str, Path],
        enable_ocr: bool = True,
    ) -> ChangeOrderExtractionResult:
        """Execute complete extraction pipeline on a target PDF or TXT file.

        Args:
            file_path: Path to the target PDF or TXT document.
            enable_ocr: Whether to run OCR if a scanned PDF is detected.

        Returns:
            Validated ChangeOrderExtractionResult conforming to JSON Schema.
        """
        logger.info("Ingesting document: %s", file_path)
        doc: DocumentContent = read_document(file_path, enable_ocr=enable_ocr)

        logger.info(
            "Read %d chars across %d page(s) (format: %s, scanned: %s)",
            doc.char_count,
            doc.page_count,
            doc.file_type,
            doc.is_scanned,
        )

        logger.info("Calling LLM extraction engine...")
        raw_extraction: RawLLMExtraction = self.llm_client.extract(doc.raw_text)

        logger.info("Performing domain business validation and conflict checks...")
        validation_report = self.validator.validate(raw_extraction, doc.raw_text)

        logger.info("Computing transparent multi-factor confidence scores...")
        confidence_report = self.confidence_scorer.evaluate(
            raw_extraction=raw_extraction,
            validation_report=validation_report,
            source_text=doc.raw_text,
            is_ocr=doc.ocr_applied,
        )

        # Assemble final compliant output
        cleaned = validation_report.cleaned_values
        scores = confidence_report.field_scores

        result = ChangeOrderExtractionResult(
            project_name=FieldOutput[str](
                value=cleaned.get("project_name"),
                confidence=scores["project_name"].final_confidence,
            ),
            contract_number=FieldOutput[str](
                value=cleaned.get("contract_number"),
                confidence=scores["contract_number"].final_confidence,
            ),
            change_order_number=FieldOutput[str](
                value=cleaned.get("change_order_number"),
                confidence=scores["change_order_number"].final_confidence,
            ),
            date=FieldOutput[str](
                value=cleaned.get("date"),
                confidence=scores["date"].final_confidence,
            ),
            requested_by=FieldOutput[str](
                value=cleaned.get("requested_by"),
                confidence=scores["requested_by"].final_confidence,
            ),
            description=FieldOutput[str](
                value=cleaned.get("description"),
                confidence=scores["description"].final_confidence,
            ),
            amount=FieldOutput[float](
                value=cleaned.get("amount"),
                confidence=scores["amount"].final_confidence,
            ),
            schedule_impact_days=FieldOutput[int](
                value=cleaned.get("schedule_impact_days"),
                confidence=scores["schedule_impact_days"].final_confidence,
            ),
            approval_status=FieldOutput[str](
                value=cleaned.get("approval_status"),
                confidence=scores["approval_status"].final_confidence,
            ),
            needs_review=confidence_report.needs_review,
        )

        return result
