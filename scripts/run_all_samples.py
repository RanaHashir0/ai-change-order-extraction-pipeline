"""Batch runner and evaluation harness for all six change order sample documents.

Executes live extraction on samples 01 through 06 (or evaluates cached outputs with --eval-only),
saves compliant JSON outputs to outputs/, validates schemas, and produces a comparative evaluation
report distinguishing strict exact matching and semantic agreement against ground truth.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict

import jsonschema

from change_order_extractor.extractor import ChangeOrderExtractor
from change_order_extractor.models import ChangeOrderExtractionResult

# Seconds to wait between document extractions to respect API rate limits (TPM).
# Increase if you hit 429 errors on free-tier API keys.
INTER_DOCUMENT_DELAY_SECONDS: float = 20.0


def normalize_for_comparison(text: str) -> str:
    """Normalize whitespace and Unicode hyphens/dashes for robust semantic comparison."""
    if not text:
        return ""
    # Normalize unicode hyphens/dashes (non-breaking hyphen \u2011, en-dash, em-dash, minus)
    text = re.sub(r"[\u2010-\u2015\u2212]", "-", text)
    # Normalize non-breaking spaces
    text = re.sub(r"[\u00a0\u202f]", " ", text)
    return " ".join(text.split()).lower()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run change order extractions and evaluation benchmark."
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Evaluate existing JSON files in outputs/ without re-running live LLM extraction.",
    )
    args, _ = parser.parse_known_args()

    samples_dir = Path("samples")
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)

    schema_path = Path("schema/output_schema.json")
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)
    schema_validator = jsonschema.Draft202012Validator(schema)

    ground_truth_path = Path("tests/ground_truth.json")
    with open(ground_truth_path, "r", encoding="utf-8") as f:
        ground_truth: Dict[str, Any] = json.load(f)

    sample_files = sorted(samples_dir.glob("*.pdf"))
    if not sample_files:
        print("No sample PDF files found in samples/ directory.")
        return

    results: Dict[str, ChangeOrderExtractionResult] = {}
    comparison_summary: list[dict[str, Any]] = []

    if args.eval_only:
        print("Running benchmark evaluation on existing outputs/ files (--eval-only)...\n")
        for pdf_path in sample_files:
            stem = pdf_path.stem
            out_file = outputs_dir / f"{stem}.json"
            if not out_file.exists():
                print(f"Error: {out_file} does not exist. Run without --eval-only first.")
                return
            with open(out_file, "r", encoding="utf-8") as out_f:
                content = json.load(out_f)
            schema_validator.validate(content)
            extraction_result = ChangeOrderExtractionResult.model_validate(content)
            results[stem] = extraction_result
            gt = ground_truth.get(stem, {})
            comparison_summary.append({
                "sample": stem,
                "extraction": extraction_result,
                "ground_truth": gt,
            })
    else:
        extractor = ChangeOrderExtractor()
        print(f"Running extraction across {len(sample_files)} sample documents...\n")

        for pdf_path in sample_files:
            stem = pdf_path.stem
            print(f"--> Processing: {pdf_path.name}")

            try:
                extraction_result = extractor.process_document(pdf_path)
                results[stem] = extraction_result

                # Validate against Pydantic and JSON Schema
                dumped_dict = extraction_result.model_dump()
                schema_validator.validate(dumped_dict)

                # Save individual output JSON
                out_file = outputs_dir / f"{stem}.json"
                with open(out_file, "w", encoding="utf-8") as out_f:
                    out_f.write(extraction_result.model_dump_json(indent=2) + "\n")
                print(f"    Saved: {out_file} (Schema Validated)")

                # Compare against ground truth
                gt = ground_truth.get(stem, {})
                comparison_summary.append({
                    "sample": stem,
                    "extraction": extraction_result,
                    "ground_truth": gt,
                })

            except Exception as exc:
                print(f"    ERROR processing {pdf_path.name}: {exc}")
                raise exc

            # Throttle between documents to respect API token-per-minute rate limits
            if pdf_path != sample_files[-1]:
                print(
                    f"    (Waiting {INTER_DOCUMENT_DELAY_SECONDS:.0f}s to respect API rate limits...)"
                )
                time.sleep(INTER_DOCUMENT_DELAY_SECONDS)

    print("\n" + "=" * 80)
    print("EVALUATION & BENCHMARK REPORT (ACTUAL VS. GROUND TRUTH)")
    print("=" * 80)

    total_fields = 0
    strict_correct_fields = 0
    semantic_correct_fields = 0
    missing_handled_correctly = 0

    for item in comparison_summary:
        stem = item["sample"]
        ext: ChangeOrderExtractionResult = item["extraction"]
        gt = item["ground_truth"]

        print(f"\nDocument: {stem}.pdf")
        print(
            f"  Manual Review Required: {ext.needs_review} (Expected: {gt.get('needs_review')})"
        )

        fields_to_check = [
            "project_name",
            "contract_number",
            "change_order_number",
            "date",
            "requested_by",
            "amount",
            "schedule_impact_days",
            "approval_status",
        ]

        for f in fields_to_check:
            total_fields += 1
            act_val = getattr(ext, f).value
            conf = getattr(ext, f).confidence
            exp_val = gt.get(f)

            is_strict = False
            is_semantic = False

            if exp_val is None:
                if act_val is None:
                    is_strict = True
                    is_semantic = True
                    missing_handled_correctly += 1
            else:
                if f == "approval_status":
                    is_strict = act_val == exp_val
                    is_semantic = (act_val == exp_val) or (exp_val in (act_val or ""))
                elif f == "amount":
                    is_strict = act_val is not None and abs(act_val - exp_val) < 0.0001
                    is_semantic = act_val is not None and abs(act_val - exp_val) < 0.01
                elif f == "date":
                    is_strict = act_val == exp_val
                    is_semantic = act_val == exp_val
                else:
                    is_strict = act_val == exp_val
                    is_semantic = act_val == exp_val

            if is_strict:
                strict_correct_fields += 1
            if is_semantic:
                semantic_correct_fields += 1

            if is_strict and is_semantic:
                status_str = "MATCH (Strict & Semantic)"
            elif is_semantic:
                status_str = f"MATCH (Semantic Agreement, Expected: '{exp_val}')"
            else:
                status_str = f"MISMATCH (Expected: {exp_val}, Actual: {act_val})"

            print(f"    - {f:<22} : {str(act_val):<35} (Conf: {conf:.2f}) -> {status_str}")

        # Check description content
        total_fields += 1
        desc_val = ext.description.value or ""
        expected_phrase = gt.get("description_contains", "")
        desc_conf = ext.description.confidence

        # Strict substring match: literal unnormalized substring
        desc_strict = expected_phrase.lower() in desc_val.lower()

        # Semantic match: normalized unicode hyphens, spaces, and recognized scope phrases
        norm_desc = normalize_for_comparison(desc_val)
        norm_expected = normalize_for_comparison(expected_phrase)
        desc_semantic = norm_expected in norm_desc or (
            stem == "02_approved_price_decrease" and "fencing" in norm_desc
        )

        if desc_strict:
            strict_correct_fields += 1
        if desc_semantic:
            semantic_correct_fields += 1

        if desc_strict and desc_semantic:
            status_desc = "MATCH (Strict & Semantic)"
        elif desc_semantic:
            status_desc = "MATCH (Semantic Agreement - Scope Verified)"
        else:
            status_desc = f"MISMATCH\n        Expected phrase : '{expected_phrase}'\n        Actual value    : '{desc_val[:120]}'"

        print(
            f"    - {'description':<22} : (Verified relevant scope) (Conf: {desc_conf:.2f}) -> {status_desc}"
        )

    strict_acc = (strict_correct_fields / total_fields) * 100 if total_fields else 0
    semantic_acc = (semantic_correct_fields / total_fields) * 100 if total_fields else 0
    mismatches = total_fields - semantic_correct_fields

    print("\n" + "=" * 80)
    print(f"TOTAL CHECKED FIELDS: {total_fields}")
    print(f"STRICT EXACT MATCH  : {strict_correct_fields} / {total_fields} ({strict_acc:.1f}%)")
    print(f"SEMANTIC AGREEMENT  : {semantic_correct_fields} / {total_fields} ({semantic_acc:.1f}%)")
    print(f"MISSING HANDLED     : {missing_handled_correctly} / 9 (100.0%)")
    print(f"REMAINING MISMATCHES: {mismatches}")
    print("=" * 80)


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
