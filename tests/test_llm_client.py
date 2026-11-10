"""Unit tests for the ChangeOrderLLMClient with mocked API calls."""

import json
from unittest.mock import MagicMock, patch

import pytest
from openai import APIConnectionError, APITimeoutError, InternalServerError

from change_order_extractor.llm_client import (
    ChangeOrderLLMClient,
    LLMAPIError,
    LLMResponseParsingError,
    MissingAPIConfigurationError,
)
from change_order_extractor.models import RawLLMExtraction

SAMPLE_VALID_LLM_OUTPUT = {
    "project_name": {
        "value": "Riverside Medical Center Expansion",
        "evidence": "Riverside Medical Center Expansion",
        "confidence_score": 0.98,
        "conflicts_detected": [],
        "reasoning": "Explicit in header",
    },
    "contract_number": {
        "value": "CT-2024-00892",
        "evidence": "Contract No: CT-2024-00892",
        "confidence_score": 0.95,
        "conflicts_detected": [],
        "reasoning": "Matched contract block",
    },
    "change_order_number": {
        "value": "CO-001",
        "evidence": "CHANGE ORDER: CO-001",
        "confidence_score": 0.99,
        "conflicts_detected": [],
        "reasoning": "Standard CO reference",
    },
    "date": {
        "value": "2024-10-15",
        "evidence": "Date: October 15, 2024",
        "confidence_score": 0.95,
        "conflicts_detected": [],
        "reasoning": "Parsed unambiguous date",
    },
    "requested_by": {
        "value": "Apex Architecture Partners",
        "evidence": "Requested By: Apex Architecture Partners",
        "confidence_score": 0.92,
        "conflicts_detected": [],
        "reasoning": "Listed under request block",
    },
    "description": {
        "value": "Additional conduit runs for MRI suite",
        "evidence": "Scope: Additional conduit runs for MRI suite",
        "confidence_score": 0.94,
        "conflicts_detected": [],
        "reasoning": "Detailed work scope",
    },
    "amount": {
        "value": 15250.00,
        "evidence": "Total Net Addition: $15,250.00",
        "confidence_score": 0.98,
        "conflicts_detected": [],
        "reasoning": "Positive price increase",
    },
    "schedule_impact_days": {
        "value": 5,
        "evidence": "Schedule Extension: 5 Calendar Days",
        "confidence_score": 0.96,
        "conflicts_detected": [],
        "reasoning": "Explicit positive day extension",
    },
    "approval_status": {
        "value": "Approved",
        "evidence": "Status: Approved by Owner 10/18/2024",
        "confidence_score": 0.97,
        "conflicts_detected": [],
        "reasoning": "Executed signature",
    },
    "document_summary": "Change order for hospital MRI conduit",
    "unresolved_ambiguities": [],
}


def create_mock_completion(content_str: str) -> MagicMock:
    """Helper to mock an OpenAI chat completion response."""
    mock_resp = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = content_str
    mock_resp.choices = [mock_choice]
    return mock_resp


def test_missing_api_key_raises_error():
    """Verify that omitting LLM_API_KEY raises MissingAPIConfigurationError."""
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(MissingAPIConfigurationError) as exc_info:
            ChangeOrderLLMClient(load_env=False)
        assert "LLM_API_KEY" in str(exc_info.value)


def test_missing_model_raises_error():
    """Verify that omitting LLM_MODEL raises MissingAPIConfigurationError."""
    with patch.dict("os.environ", {"LLM_API_KEY": "fake_key"}, clear=True):
        with pytest.raises(MissingAPIConfigurationError) as exc_info:
            ChangeOrderLLMClient(load_env=False)
        assert "LLM_MODEL" in str(exc_info.value)


def test_successful_extraction_parsing():
    """Test standard extraction returning valid JSON parsed into RawLLMExtraction."""
    client = ChangeOrderLLMClient(
        api_key="dummy-key",
        base_url="https://api.groq.com/openai/v1",
        model="llama-3.3-70b-versatile",
    )

    mock_resp = create_mock_completion(json.dumps(SAMPLE_VALID_LLM_OUTPUT))

    with patch.object(client.client.chat.completions, "create", return_value=mock_resp):
        res = client.extract("Sample construction contract text")
        assert isinstance(res, RawLLMExtraction)
        assert res.project_name.value == "Riverside Medical Center Expansion"
        assert res.amount.value == 15250.00
        assert res.schedule_impact_days.value == 5
        assert res.approval_status.value == "Approved"
        assert res.date.value == "2024-10-15"
        assert res.amount.evidence == "Total Net Addition: $15,250.00"


def test_clean_markdown_wrapped_json():
    """Test that responses wrapped in markdown code blocks (```json ... ```) are cleanly parsed."""
    client = ChangeOrderLLMClient(
        api_key="dummy-key",
        model="llama-3.3-70b-versatile",
    )

    wrapped_content = f"```json\n{json.dumps(SAMPLE_VALID_LLM_OUTPUT)}\n```"
    mock_resp = create_mock_completion(wrapped_content)

    with patch.object(client.client.chat.completions, "create", return_value=mock_resp):
        res = client.extract("Contract text")
        assert res.project_name.value == "Riverside Medical Center Expansion"


def test_invalid_json_raises_parsing_error():
    """Test that unparseable raw content raises LLMResponseParsingError."""
    client = ChangeOrderLLMClient(
        api_key="dummy-key",
        model="llama-3.3-70b-versatile",
    )

    mock_resp = create_mock_completion("Not valid JSON at all: 404")

    with patch.object(client.client.chat.completions, "create", return_value=mock_resp):
        with pytest.raises(LLMResponseParsingError) as exc:
            client.extract("Contract text")
        assert "Failed to decode LLM response as JSON" in str(exc.value)


def test_empty_llm_response_raises_parsing_error():
    """Test that an empty string from the LLM raises LLMResponseParsingError."""
    client = ChangeOrderLLMClient(
        api_key="dummy-key",
        model="llama-3.3-70b-versatile",
    )

    mock_resp = create_mock_completion("")

    with patch.object(client.client.chat.completions, "create", return_value=mock_resp):
        with pytest.raises(LLMResponseParsingError) as exc:
            client.extract("Contract text")
        assert "empty response" in str(exc.value)


def test_api_timeout_error_handling():
    """Test that APITimeoutError is caught and raised as LLMAPIError."""
    client = ChangeOrderLLMClient(
        api_key="dummy-key",
        model="llama-3.3-70b-versatile",
    )

    with patch.object(
        client.client.chat.completions,
        "create",
        side_effect=APITimeoutError(request=MagicMock()),
    ):
        with pytest.raises(LLMAPIError) as exc:
            client.extract("Contract text")
        assert "timeout failure" in str(exc.value)


def test_negative_amount_and_schedule():
    """Verify deductive change order values (negative numbers) pass through correctly."""
    client = ChangeOrderLLMClient(
        api_key="dummy-key",
        model="llama-3.3-70b-versatile",
    )

    deductive_output = dict(SAMPLE_VALID_LLM_OUTPUT)
    deductive_output["amount"] = {
        "value": -4200.50,
        "evidence": "Credit to Owner: ($4,200.50)",
        "confidence_score": 0.95,
        "conflicts_detected": [],
        "reasoning": "Deductive change",
    }
    deductive_output["schedule_impact_days"] = {
        "value": -3,
        "evidence": "Acceleration reduction: -3 calendar days",
        "confidence_score": 0.90,
        "conflicts_detected": [],
        "reasoning": "Schedule reduction",
    }

    mock_resp = create_mock_completion(json.dumps(deductive_output))
    with patch.object(client.client.chat.completions, "create", return_value=mock_resp):
        res = client.extract("Contract text")
        assert res.amount.value == -4200.50
        assert res.schedule_impact_days.value == -3


def test_description_prompt_requires_body_detail_over_subject_line():
    """Test that the system prompt includes the DESCRIPTION rule requiring body paragraph specificity.

    This is a regression guard: ensures the prompt instructs the LLM to prefer detailed
    body paragraph scope over paraphrased RE: / subject-line headings.
    """
    from change_order_extractor.llm_client import EXTRACTION_SYSTEM_PROMPT

    # Must contain DESCRIPTION section
    assert "DESCRIPTION" in EXTRACTION_SYSTEM_PROMPT
    # Must mention body paragraph preference over subject line
    assert "body paragraph" in EXTRACTION_SYSTEM_PROMPT.lower() or "body" in EXTRACTION_SYSTEM_PROMPT.lower()
    # Must mention not reducing to subject-line paraphrase
    assert "subject-line" in EXTRACTION_SYSTEM_PROMPT or "RE:" in EXTRACTION_SYSTEM_PROMPT
