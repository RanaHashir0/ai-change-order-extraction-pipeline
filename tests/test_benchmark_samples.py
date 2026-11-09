"""Automated tests validating all 6 generated outputs in outputs/."""

import json
from pathlib import Path

import jsonschema
import pytest

from change_order_extractor.models import ChangeOrderExtractionResult

OUTPUTS_DIR = Path("outputs")
SCHEMA_PATH = Path("schema/output_schema.json")
GROUND_TRUTH_PATH = Path("tests/ground_truth.json")


def test_all_six_outputs_exist_and_validate():
    """Verify all 6 output JSON files exist and strictly conform to schema and business rules."""
    output_files = sorted(OUTPUTS_DIR.glob("*.json"))
    assert len(output_files) == 6, f"Expected 6 output files, found {len(output_files)}"

    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema = json.load(f)
    validator = jsonschema.Draft202012Validator(schema)

    with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    for out_file in output_files:
        content = json.loads(out_file.read_text(encoding="utf-8"))

        # 1. JSON Schema Draft 2020-12 validation
        validator.validate(content)

        # 2. Pydantic validation
        result = ChangeOrderExtractionResult.model_validate(content)
        assert isinstance(result, ChangeOrderExtractionResult)

        # 3. Ground truth checks
        gt = ground_truth.get(out_file.stem)
        assert gt is not None, f"Ground truth missing for {out_file.stem}"
        assert result.needs_review == gt["needs_review"], (
            f"needs_review mismatch for {out_file.stem}: expected {gt['needs_review']}, got {result.needs_review}"
        )
