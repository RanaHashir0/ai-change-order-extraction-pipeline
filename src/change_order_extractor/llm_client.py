"""LLM client for construction change order extraction using OpenAI-compatible APIs."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Optional

from dotenv import load_dotenv
from openai import APIConnectionError, APIError, APITimeoutError, OpenAI

from change_order_extractor.models import RawLLMExtraction

logger = logging.getLogger(__name__)

EXTRACTION_SYSTEM_PROMPT = """You are an expert Construction Contract Administrator and AI Information Extraction Engine.
Your task is to analyze construction change order documents (PDF or plain text) and extract nine structured fields with verbatim supporting evidence.

### EXTRACTION RULES:
1. TRUTHFULNESS & GROUNDING:
   - NEVER fabricate or guess missing information.
   - If a field is absent, omitted, or unstated in the document, its value MUST be null.
   - For every non-null field, you MUST provide the exact verbatim quote from the document text in "evidence".

2. SIGN CONVENTIONS:
   - "amount": Net contract price impact as a float.
     * Price increases / additions MUST be POSITIVE (e.g., "$15,250.00" -> 15250.0).
     * Deductive change orders, price reductions, or credits MUST be NEGATIVE (e.g., "($4,500.00)" or "Credit of $4,500" -> -4500.0).
     * If an estimated amount is explicitly stated (e.g. "Estimated amount: ADD $6,200.00"), extract the numeric value (e.g. 6200.0) with verbatim evidence and note the estimate in reasoning.
     * If the document indicates $0 or schedule-only, use 0.0 or null according to explicit document statements.
   - "schedule_impact_days": Net schedule impact in integer days.
     * Schedule extensions / additional days MUST be POSITIVE integers (e.g., "14 calendar days" -> 14).
     * Schedule reductions / acceleration / time saved MUST be NEGATIVE integers (e.g., "3 days reduction" -> -3).
     * If explicitly stated as 0 days, use 0. If unmentioned, use null.

3. DATES & FORMATS:
   - "date": Represent as ISO format "YYYY-MM-DD" when the date is unambiguous.
   - Distinguish a change-order date from an attachment preparation date or signature/approval date.
   - Only report a contradiction if the dates refer to the same semantic field or the document explicitly establishes a conflict.

4. DESCRIPTION — SCOPE OF WORK:
   - "description": Capture the specific scope of work, reason for change, materials involved, hazardous conditions, or work items described in the document body.
   - Prefer the detailed body paragraph over a subject-line heading or RE: line. For example, if the RE line says "additional contract time due to concealed conditions" but the body paragraph specifies "concealed asbestos-containing floor adhesive beneath the existing vinyl tile", the description MUST include the specific material and location detail from the body.
   - Include all key specifics: materials, locations, hazard types, affected systems, or work activities explicitly stated.
   - Do NOT reduce the description to only the subject-line paraphrase if the document body contains more precise scope detail.
   - Do NOT fabricate details not present in the document.

5. REQUESTED BY & DESIGNATIONS:
   - "requested_by": Extract the person, contractor, architect, engineer, or representative requesting the change order.
   - When a professional role, title, or designation is explicitly associated with the requester's name in the document (e.g. "From: Name, Title" or "Name, Role"), preserve BOTH the full name and the professional designation (e.g., "Priya Shah, Project Architect" or "Elena Brooks, Project Manager").
   - Do NOT truncate or omit explicit professional designations appearing alongside the requester's name.
   - Do NOT invent or fabricate designations if none are stated in the document.

6. APPROVAL STATUS:
   - Extract the exact state explicitly indicated by the document (e.g., "Approved", "Pending", "Pending Approval", "Rejected", "Under Review", "Draft").
   - NEVER infer "Pending" approval simply because a document or signature block is unsigned.
   - If approval status is not explicitly stated in the document, return null.

7. CONFLICT DETECTION:
   - Thoroughly inspect the document for conflicting information (e.g. differing amounts on cover sheet vs. itemized table, differing dates, conflicting contract numbers).
   - NOTE: Accounting parentheses "($X)" and negative sign "-$X" denote the exact same negative credit amount and are NOT a conflict.
   - Line items or cost breakdowns that match or sum up to the total are NOT conflicts.
   - Distinguish a change-order date from an attachment preparation date. Only report a contradiction if the dates refer to the same semantic field or the document explicitly establishes a conflict.
   - Only report in "conflicts_detected" when two separate sections of the document genuinely contradict or disagree on the final value (e.g. cover sheet says $18,400 but pricing sheet says $14,800, or document explicitly states sections disagree).
   - If genuine conflicting values exist, list every conflicting mention in the "conflicts_detected" array for that field.

8. OUTPUT FORMAT:
   - Return a valid, pure JSON object strictly conforming to the requested schema. Do not output markdown code blocks or conversational text.
"""

EXTRACTION_USER_PROMPT_TEMPLATE = """Please extract the 9 change order fields from the following document text:

--- BEGIN DOCUMENT TEXT ---
{document_text}
--- END DOCUMENT TEXT ---

Output a JSON object matching this structure:
{{
  "project_name": {{"value": string or null, "evidence": string or null, "confidence_score": float, "conflicts_detected": [string], "reasoning": string}},
  "contract_number": {{"value": string or null, "evidence": string or null, "confidence_score": float, "conflicts_detected": [string], "reasoning": string}},
  "change_order_number": {{"value": string or null, "evidence": string or null, "confidence_score": float, "conflicts_detected": [string], "reasoning": string}},
  "date": {{"value": string or null, "evidence": string or null, "confidence_score": float, "conflicts_detected": [string], "reasoning": string}},
  "requested_by": {{"value": string or null, "evidence": string or null, "confidence_score": float, "conflicts_detected": [string], "reasoning": string}},
  "description": {{"value": string or null, "evidence": string or null, "confidence_score": float, "conflicts_detected": [string], "reasoning": string}},
  "amount": {{"value": number or null, "evidence": string or null, "confidence_score": float, "conflicts_detected": [string], "reasoning": string}},
  "schedule_impact_days": {{"value": integer or null, "evidence": string or null, "confidence_score": float, "conflicts_detected": [string], "reasoning": string}},
  "approval_status": {{"value": string or null, "evidence": string or null, "confidence_score": float, "conflicts_detected": [string], "reasoning": string}},
  "document_summary": string or null,
  "unresolved_ambiguities": [string]
}}
"""


class LLMClientError(Exception):
    """Base exception for LLM client operations."""


class MissingAPIConfigurationError(LLMClientError):
    """Raised when required API credentials or URLs are missing."""


class LLMAPIError(LLMClientError):
    """Raised when the LLM API encounters a connection, timeout, or server error."""


class LLMResponseParsingError(LLMClientError):
    """Raised when the LLM returns invalid or unparseable JSON."""


class ChangeOrderLLMClient:
    """OpenAI-compatible client for extracting change order data."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 60.0,
        max_retries: int = 2,
        load_env: bool = True,
    ) -> None:
        """Initialize LLM client with environment variable loading.

        Args:
            api_key: Optional explicit API key.
            base_url: Optional explicit base URL.
            model: Optional explicit model name.
            timeout: API request timeout in seconds.
            max_retries: Maximum automatic retries on transient errors.
            load_env: Whether to load environment from .env file.
        """
        if load_env:
            load_dotenv(override=False)

        resolved_api_key = api_key if api_key is not None else os.getenv("LLM_API_KEY")
        resolved_base_url = base_url if base_url is not None else os.getenv("LLM_BASE_URL")
        resolved_model = model if model is not None else os.getenv("LLM_MODEL")

        self.api_key = resolved_api_key
        self.base_url = resolved_base_url
        self.model = resolved_model
        self.timeout = timeout
        self.max_retries = max_retries

        if not self.api_key or not self.api_key.strip():
            raise MissingAPIConfigurationError(
                "LLM_API_KEY environment variable is missing or empty. "
                "Please configure LLM_API_KEY in your .env file or environment."
            )

        if not self.model or not self.model.strip():
            raise MissingAPIConfigurationError(
                "LLM_MODEL environment variable is missing or empty. "
                "Please configure LLM_MODEL in your .env file or environment."
            )

        # Standard OpenAI client supports any compatible endpoint via base_url
        client_kwargs: dict[str, Any] = {
            "api_key": self.api_key,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }
        if self.base_url and self.base_url.strip():
            client_kwargs["base_url"] = self.base_url.strip()

        self.client = OpenAI(**client_kwargs)

    def extract(self, document_text: str) -> RawLLMExtraction:
        """Send document text to LLM and parse into RawLLMExtraction.

        Args:
            document_text: Raw or OCR text extracted from the document.

        Returns:
            RawLLMExtraction containing extracted fields, evidence, and detected conflicts.

        Raises:
            LLMAPIError: On network, timeout, or provider API errors.
            LLMResponseParsingError: If LLM output cannot be parsed into the expected schema.
        """
        user_prompt = EXTRACTION_USER_PROMPT_TEMPLATE.format(document_text=document_text)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
        except (APITimeoutError, APIConnectionError) as conn_err:
            raise LLMAPIError(f"LLM API connection or timeout failure: {conn_err}") from conn_err
        except APIError as api_err:
            raise LLMAPIError(f"LLM API returned an error: {api_err}") from api_err
        except Exception as exc:
            raise LLMAPIError(f"Unexpected error calling LLM API: {exc}") from exc

        choice = response.choices[0]
        raw_content = choice.message.content

        if not raw_content or not raw_content.strip():
            raise LLMResponseParsingError("LLM returned an empty response.")

        cleaned_json = self._clean_json_content(raw_content)

        try:
            parsed_data = json.loads(cleaned_json)
        except json.JSONDecodeError as json_err:
            raise LLMResponseParsingError(
                f"Failed to decode LLM response as JSON: {json_err}. Content: {raw_content[:200]}"
            ) from json_err

        try:
            return RawLLMExtraction.model_validate(parsed_data)
        except Exception as val_err:
            raise LLMResponseParsingError(
                f"LLM JSON response does not conform to RawLLMExtraction schema: {val_err}"
            ) from val_err

    @staticmethod
    def _clean_json_content(content: str) -> str:
        """Strip markdown code fence blocks if present."""
        trimmed = content.strip()
        # Remove ```json ... ``` or ``` ... ```
        pattern = r"^```(?:json)?\s*\n?(.*?)\n?```$"
        match = re.search(pattern, trimmed, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return trimmed
