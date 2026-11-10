# AI-Powered Construction Change Order Extraction Pipeline

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![JSON Schema Draft 2020-12](https://img.shields.io/badge/JSON%20Schema-Draft%202020--12-green.svg)](https://json-schema.org/draft/2020-12/schema)
[![Tests Passing](https://img.shields.io/badge/tests-46%20passed-brightgreen.svg)]()
[![OpenAI-Compatible](https://img.shields.io/badge/LLM-OpenAI%20%7C%20Groq%20Compatible-orange.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A robust, production-grade backend Python extraction pipeline developed for the **Sledge AI Engineer technical assessment**. The engine ingests construction change-order documents (digital PDF, scanned PDF with OCR fallback, or plain text), parses and validates domain-specific data, detects intra-document contradictions across cover sheets and attachments, computes grounded heuristic confidence scores, and outputs strictly compliant JSON conforming to **JSON Schema Draft 2020-12**.

**GitHub Repository:** [https://github.com/RanaHashir0/ai-change-order-extraction-pipeline](https://github.com/RanaHashir0/ai-change-order-extraction-pipeline)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [Solution and Architecture](#3-solution-and-architecture)
4. [Technology Stack](#4-technology-stack)
5. [Installation Instructions](#5-installation-instructions)
6. [Environment Configuration](#6-environment-configuration)
7. [How to Use the Application](#7-how-to-use-the-application)
8. [Example Input and JSON Output](#8-example-input-and-json-output)
9. [JSON Schema Explanation](#9-json-schema-explanation)
10. [Validation Rules](#10-validation-rules)
11. [Heuristic Confidence-Scoring Methodology](#11-heuristic-confidence-scoring-methodology)
12. [Testing Instructions](#12-testing-instructions)
13. [Benchmark Results Across Six Sample Documents](#13-benchmark-results-across-six-sample-documents)
14. [Failure Modes and Limitations](#14-failure-modes-and-limitations)
15. [Security Considerations](#15-security-considerations)
16. [Future Improvements](#16-future-improvements)

---

## 1. Project Overview

Construction change orders represent legally binding contractual amendments adjusting financial compensation, scope of work, and project delivery schedules. In commercial construction management, automating change-order ingestion demands strict reliability:
- Missing contractual information must never be fabricated or guessed.
- Conflicting figures across cover sheets, proposals, and itemized schedules must be flagged for human review rather than silently resolved.
- Price additions must remain positive numeric values, while deductive credits must be explicitly negative.
- Output JSON must strictly adhere to contractual schemas without internal prompt or token leakage.
- Low-confidence or ambiguous fields must proactively set `needs_review: true`.

This pipeline delivers a resilient, deterministic, and verifiable backend extraction workflow meeting every requirement of the Sledge AI Engineer assessment.

### The Six Benchmark Documents & Evaluation Cases

The test suite evaluates the pipeline across six distinct change order documents, each specifically crafted to represent critical contractual and operational edge cases in commercial construction management:

| # | Document File | Primary Scenario | Key Construction Edge Cases Evaluated | Expected `needs_review` |
|:---:|:---|:---|:---|:---:|
| **01** | [`01_complete_approved_increase.pdf`](samples/01_complete_approved_increase.pdf) | **Standard Approved Price Increase** | • Full data availability (all 9 fields present)<br>• Positive monetary impact (`+$18,500.00` -> `18500.0`)<br>• Positive schedule extension (`+12 calendar days` -> `12`)<br>• Requester name with professional title (`Elena Brooks, Project Manager`)<br>• Explicit `Approved` status | `False` |
| **02** | [`02_approved_price_decrease.pdf`](samples/02_approved_price_decrease.pdf) | **Approved Deductive Scope Credit** | • Negative monetary impact (`-$8,750.00` / credit -> `-8750.0`)<br>• Explicit zero schedule impact (`0 calendar days` -> integer `0`, distinct from `null`)<br>• Requester with designation (`Marco Ruiz, Owner Representative`)<br>• No false-positive review triggered for credits | `False` |
| **03** | [`03_schedule_only_extension.pdf`](samples/03_schedule_only_extension.pdf) | **Schedule-Only Time Extension** | • Legitimate `null` amount (document explicitly states cost is unstated/separate)<br>• Positive schedule impact (`+18 calendar days` -> `18`)<br>• Body-paragraph specific scope extraction (`concealed asbestos-containing floor adhesive`)<br>• Requester with role (`Priya Shah, Project Architect`)<br>• `Approved - Time Extension Only` mapped cleanly to `Approved` | `False` |
| **04** | [`04_pending_approval_increase.pdf`](samples/04_pending_approval_increase.pdf) | **Pending Owner Approval Request** | • Explicit in-flight status (`Pending Approval` / `Pending`)<br>• Never infers or defaults to "Approved" for unexecuted requests<br>• Positive price addition (`+$42,500.00` -> `42500.0`)<br>• Schedule extension (`+7 calendar days` -> `7`)<br>• Requester designation (`David Kim, Construction Manager`) | `False` |
| **05** | [`05_incomplete_missing_fields.pdf`](samples/05_incomplete_missing_fields.pdf) | **Incomplete Draft & Tentative Estimate** | • Truthful missing field handling (`contract_number`, `requested_by`, `schedule_impact_days` are `null`)<br>• Estimated amount extraction (`+$6,200.00` extracted as `6200.00` rather than discarded)<br>• Heuristic confidence discount (`0.81` < `0.85` threshold)<br>• Explicit `Under Review` status<br>• Mandatory review escalation triggered | `True` |
| **06** | [`06_conflicting_values_needs_review.pdf`](samples/06_conflicting_values_needs_review.pdf) | **Cross-Section Audit Discrepancies** | • Cover sheet vs Attachment B contradictory contract numbers (`CT-8841-A` vs `CT-8841-B` -> `null`)<br>• Contradictory amounts (`$18,400.00` vs `$14,800.00` -> `null`)<br>• Contradictory schedule impacts (`+15 days` vs `+10 days` -> `null`)<br>• Contradictory approval statuses (`Approved` vs `Draft` -> `null`)<br>• Explicit review note establishes date contradiction (`null`)<br>• Conflicting fields receive `0.00` confidence; review flag enforced | `True` |


---

## 2. Problem Statement

Construction procurement documents exhibit substantial structural and semantic variance:
- **Diverse formats**: Multi-page digital PDFs, scanned paper documents, and plain-text memoranda.
- **Cross-section discrepancies**: Cover sheets disagreeing with attached contractor worksheets (e.g., differing total amounts, dates, or contract identifiers).
- **Legitimate schedule-only changes**: Time extension requests where no financial compensation is specified.
- **Incomplete or tentative drafts**: Documents lacking prime contract identifiers or stating unapproved estimates.
- **LLM overconfidence & hallucination**: General LLMs frequently hallucinate missing fields, invent approval statuses from blank signature lines, or assign arbitrary 0.99 confidence to guessed values.

The objective is an automated extraction engine that eliminates hallucination, detects domain contradictions, enforces sign conventions, and provides transparent heuristic confidence scores grounded in verbatim document evidence.

---

## 3. Solution and Architecture

The application adopts a modular, layered architecture separating document ingestion, LLM prompting, business logic validation, confidence assessment, and CLI presentation.

```text
ai-change-order-extraction-pipeline/
├── samples/                                # 6 Synthetic change order test PDFs
│   ├── 01_complete_approved_increase.pdf
│   ├── 02_approved_price_decrease.pdf
│   ├── 03_schedule_only_extension.pdf
│   ├── 04_pending_approval_increase.pdf
│   ├── 05_incomplete_missing_fields.pdf
│   └── 06_conflicting_values_needs_review.pdf
│
├── outputs/                                # Validated JSON output per document
│   ├── 01_complete_approved_increase.json
│   ├── ...
│   └── 06_conflicting_values_needs_review.json
│
├── schema/
│   └── output_schema.json                  # Formal JSON Schema Draft 2020-12
│
├── src/
│   └── change_order_extractor/
│       ├── __init__.py                     # Package export
│       ├── cli.py                          # CLI entry point (stdout JSON, stderr logs)
│       ├── pdf_reader.py                   # PyMuPDF ingestion, scanned detection & OCR
│       ├── llm_client.py                   # OpenAI-compatible API client & structured prompts
│       ├── models.py                       # Pydantic internal & public output schemas
│       ├── validator.py                    # Domain rules, conflict & anomaly detector
│       ├── confidence.py                   # Multi-factor grounded confidence scoring
│       └── extractor.py                    # Pipeline orchestrator
│
├── tests/
│   ├── ground_truth.json                   # Verified reference truth for all 6 samples
│   ├── test_pdf_reader.py                  # Document reader & OCR unit tests
│   ├── test_llm_client.py                  # Mocked API client & error handling tests
│   ├── test_validator.py                   # Domain rule & JSON schema unit tests
│   ├── test_confidence.py                  # Multi-factor confidence scoring tests
│   ├── test_extractor.py                   # CLI & orchestrator integration tests
│   └── test_benchmark_samples.py           # Regression tests against outputs/
│
├── scripts/
│   └── run_all_samples.py                  # Batch runner and comparative benchmark report
│
├── .env.example                            # Configuration template
├── .gitignore                              # Git exclusion rules
├── Makefile                                # make extract FILE=<path> and test targets
├── pyproject.toml                          # Package specification & editable install
├── pytest.ini                              # Test runner configuration
├── requirements.txt                        # Dependency pins
└── README.md                               # System documentation
```

### Pipeline Flow

```mermaid
flowchart TD
    A["Input Document (PDF or TXT)"] --> B["pdf_reader.py\n(PyMuPDF / OCR Fallback)"]
    B --> C["llm_client.py\n(OpenAI SDK / Groq Structured JSON)"]
    C --> D["RawLLMExtraction\n(Candidate Values + Verbatim Evidence + Conflicts)"]
    D --> E["validator.py\n(Domain Business Rules & Contradiction Detection)"]
    E --> F["confidence.py\n(Multi-Factor Evidence Grounding & Absence Scoring)"]
    F --> G["ChangeOrderExtractionResult\n(9 Fields + needs_review)"]
    G --> H["JSON Output\n(Validated against Draft 2020-12)"]
```

---

## 4. Technology Stack

- **Python**: `3.11+`
- **PDF Extraction**: `PyMuPDF` (`fitz` `>= 1.24.0`)
- **LLM Integration**: `openai` (`>= 1.40.0`) using OpenAI-compatible endpoints (Groq API supported)
- **Data Validation & Schemas**: `pydantic` (`>= 2.8.0`), `jsonschema` (`>= 4.23.0`) Draft 2020-12
- **Environment Management**: `python-dotenv` (`>= 1.0.1`)
- **OCR Engine (Optional Fallback)**: `pytesseract` (`>= 0.3.10`)
- **Test Suite**: `pytest` (`>= 8.3.0`)
- **Task Automation**: GNU Make / Windows `mingw32-make`

---

## 5. Installation Instructions

### Prerequisites
- Python 3.11 or newer installed.
- (Optional for OCR) Tesseract OCR engine installed and accessible in system PATH.

### 1. Clone or Download Repository

#### Option A: Clone with Git
```bash
git clone https://github.com/RanaHashir0/ai-change-order-extraction-pipeline.git
cd ai-change-order-extraction-pipeline
```

#### Option B: Download ZIP
Download the repository as a ZIP archive from GitHub, extract the contents, and open your terminal in the extracted root directory:
```bash
cd ai-change-order-extraction-pipeline
```

### 2. Create and Activate Virtual Environment

**On Windows PowerShell:**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**On Linux/macOS:**
```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

---

## 6. Environment Configuration

The application reads configuration from environment variables via `python-dotenv`.

### 1. Initialize Local Environment File

**On Windows PowerShell:**
```powershell
Copy-Item .env.example .env
```

**On Linux/macOS:**
```bash
cp .env.example .env
```

### 2. Configure API Credentials

Edit `.env` with your OpenAI-compatible or Groq API credentials:

```env
# LLM API Configuration (OpenAI-compatible)
LLM_API_KEY=gsk_your_groq_api_key_here
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_MODEL=openai/gpt-oss-120b
```

> [!IMPORTANT]
> - Never hardcode API keys or commit `.env` to version control.
> - `.gitignore` explicitly excludes `.env` and all local variant files.
> - Any OpenAI-compatible provider (Groq, OpenAI, Together, vLLM) can be used by setting `LLM_BASE_URL` and `LLM_MODEL`.

---

## 7. How to Use the Application

### 1. Run Automated Tests
Execute the complete test suite verifying unit, integration, and schema validation components:
```bash
python -m pytest
```
Or using Make:
```bash
make test
```

### 2. Run All Six Benchmark Documents
Execute extraction across all six sample PDFs, validate outputs against JSON Schema Draft 2020-12, save outputs to `outputs/`, and generate a comparative evaluation report:
```bash
python scripts/run_all_samples.py
```

### 3. Extract a Single Document via Make
```bash
make extract FILE=samples/01_complete_approved_increase.pdf
```
*(On Windows systems with MinGW, run `mingw32-make extract FILE=samples/01_complete_approved_increase.pdf`)*.

### 4. Direct CLI Execution (PDF or TXT)
The CLI module accepts both PDF (`.pdf`) and plain-text (`.txt`) documents:
```bash
# PDF document extraction
python -m change_order_extractor.cli samples/01_complete_approved_increase.pdf

# Plain text document extraction
python -m change_order_extractor.cli path/to/document.txt
```

### CLI Stream Separation
- **stdout**: Strictly outputs exactly **one** valid JSON object matching the schema.
- **stderr**: Diagnostic messages, character density stats, and error traces.
- **Exit Codes**:
  - `0`: Successful extraction and validation.
  - `1`: Document parsing, validation, or LLM API error.
  - `2`: Missing required command-line arguments.

---

## 8. Example Input and JSON Output

### Input Document Excerpt (`samples/01_complete_approved_increase.pdf`)
```text
CHANGE ORDER REQUEST
Project name: Riverside Medical Center Expansion
Contract number: RMC-2026-014
Change order no.: CO-017
Date: October 14, 2026
Requested by: Elena Brooks, Project Manager
Status: APPROVED
Description of change: Provide and install an underground electrical duct bank...
TOTAL CHANGE ORDER AMOUNT: +$18,500.00
Schedule impact: This change requires twelve (12) additional calendar days.
```

### Produced Output (`outputs/01_complete_approved_increase.json`)
```json
{
  "project_name": {
    "value": "Riverside Medical Center Expansion",
    "confidence": 1.0
  },
  "contract_number": {
    "value": "RMC-2026-014",
    "confidence": 1.0
  },
  "change_order_number": {
    "value": "CO-017",
    "confidence": 1.0
  },
  "date": {
    "value": "2026-10-14",
    "confidence": 1.0
  },
  "requested_by": {
    "value": "Elena Brooks, Project Manager",
    "confidence": 1.0
  },
  "description": {
    "value": "Provide and install an underground electrical duct bank along the east service drive, as shown on revised site utility drawing E-204, Revision 3. Work includes saw cutting, trench excavation, four 4-inch PVC conduits, concrete encasement, warning tape, backfill, compaction testing, and connection to the existing electrical manhole. Restore disturbed paving to match existing conditions.",
    "confidence": 1.0
  },
  "amount": {
    "value": 18500.0,
    "confidence": 1.0
  },
  "schedule_impact_days": {
    "value": 12,
    "confidence": 1.0
  },
  "approval_status": {
    "value": "Approved",
    "confidence": 1.0
  },
  "needs_review": false
}
```

---

## 9. JSON Schema Explanation

The schema is maintained at [`schema/output_schema.json`](schema/output_schema.json) and complies with **Draft 2020-12**:
- The root object requires all 9 contractual fields plus `needs_review: boolean`.
- Every field follows `FieldOutput[T]`:
  ```json
  {
    "value": "extracted value or null",
    "confidence": 0.0
  }
  ```
- `additionalProperties: false` is strictly enforced to prevent internal prompt tokens, reasoning chains, or quotes from leaking into public output.
- Field definitions:
  - `project_name`: `string` or `null`
  - `contract_number`: `string` or `null`
  - `change_order_number`: `string` or `null`
  - `date`: `string` (ISO `YYYY-MM-DD`) or `null`
  - `requested_by`: `string` or `null`
  - `description`: `string` or `null`
  - `amount`: `number` or `null`
  - `schedule_impact_days`: `integer` or `null`
  - `approval_status`: `string` or `null`
  - `needs_review`: `boolean`

---

## 10. Validation Rules

The `ChangeOrderValidator` enforces domain-specific business logic:

1. **Sign Conventions**:
   - Price additions: Must be positive numeric floats (e.g. `+$18,500.00` -> `18500.0`).
   - Deductive credits: Must be negative numeric floats (e.g. `($8,750.00)` credit -> `-8750.0`). Automatic sign verification flips positive numbers to negative if the evidence explicitly specifies a credit or deductive change.
   - Schedule extensions: Must be positive integers (e.g. `12 days` -> `12`).
   - Schedule reductions: Must be negative integers.
2. **Date Normalization & Semantic Distinction**:
   - Dates must resolve to unambiguous ISO `YYYY-MM-DD`.
   - Rejects impossible calendar dates (e.g. `2026-02-31`).
   - Distinguishes a change-order date from an attachment preparation date or signature/approval date. An attachment preparation date is not inherently contradictory to a change-order date.
   - Contradictions are only reported if the dates refer to the same semantic field (e.g. two conflicting change-order dates) or the document explicitly establishes a conflict (such as Document 06, where an explicit review note establishes that the cover sheet and Attachment B disagree on date).
   - If genuine conflicting dates exist, returns `null` with `0.0` confidence and triggers `needs_review: true`.
3. **No Guessing Missing Information**:
   - Omitted fields return `null`.
   - Incomplete documents lacking required identification (e.g. missing contract number and requester) set `needs_review: true`.
4. **Approval Status Verification**:
   - Only recognized statuses (`Approved`, `Pending`, `Pending Approval`, `Under Review`, `Rejected`, `Draft`, `Void`) are accepted.
   - **Never infer 'Pending' approval simply because a document or signature block is unsigned.** If approval status is not explicitly stated in the document text, return `null` and trigger `needs_review: true`.
   - Complementary review phrasing (e.g., `"Submitted for review - no approval recorded"` alongside `"STATUS: UNDER REVIEW"`) is reconciled as `"Under Review"` rather than treated as a false contradiction.
5. **Conflict Handling**:
   - When different sections of a document state contradictory figures (e.g. cover sheet states $18,400 but pricing attachment states $14,800), the system does **not** silently pick a number. It sets `amount.value = null`, drops confidence to `0.0`, and triggers `needs_review: true`.
6. **Legitimate Schedule-Only Changes**:
   - If a document is explicitly for time extension only with no financial adjustment (Sample 03), `amount` is legitimately `null` without triggering an error.
7. **Estimated Amounts (Document 05)**:
   - When a document explicitly contains an estimated or tentative amount (such as Document 05 stating an estimated amount of `+$6,200.00`), the pipeline extracts the numeric value (`6200.00`) rather than discarding it as `null`.
   - It assigns a discounted confidence score (`0.71` / `0.81`, below the `0.85` threshold) and sets `needs_review: true` for mandatory human review.

---

## 11. Heuristic Confidence-Scoring Methodology

### Heuristic Nature of Confidence Scores

> [!IMPORTANT]
> Confidence scores in this pipeline are **heuristic estimates** computed from multi-factor evidence grounding, text explicitness, and business validation rules. They should **not** be claimed as statistically calibrated probabilities. A sample of six documents cannot establish statistical calibration. Empirical calibration (e.g., via Platt scaling, isotonic regression, Brier score minimization, or Expected Calibration Error) requires a sufficiently large verified dataset of hundreds or thousands of annotated change orders.

### The Multi-Factor Grounding Formula

LLM self-reported confidence is notoriously uncalibrated. This pipeline computes transparent, evidence-grounded confidence estimates:

$$\text{Confidence} = Q_{\text{doc}} \times \left(0.50 \cdot G_{\text{evidence}} + 0.30 \cdot E_{\text{explicitness}} + 0.20 \cdot V_{\text{validation}}\right)$$

Where:
- **$Q_{\text{doc}}$ (Document Quality)**: Assesses character extraction density, digital text integrity, and applies an OCR discount factor ($0.90$) for scanned documents.
- **$G_{\text{evidence}}$ (Evidence Grounding)**:
  - $1.0$: Verbatim evidence quote is located as an exact substring in the source document.
  - $0.8$: High token overlap ($\ge 70\%$) with source text.
  - $0.2$: Quoted evidence cannot be found in the document (potential hallucination).
  - $0.3$: Value asserted with no supporting quote.
- **$E_{\text{explicitness}}$ (Explicitness)**: Direct contractual labels receive $1.0$; vague, estimated, or inferred numbers receive $0.5 - 0.6$.
- **$V_{\text{validation}}$ (Validation Status)**: Clean validation passes receive $1.0$; fields requiring sign normalization or estimated amounts receive $0.8$.
- **Contradiction Drop**: Any field subject to conflicting document figures receives **$0.0$** confidence.

### Absence Confidence vs. Value Confidence
For missing fields (`value = null`), the system estimates confidence in the conclusion that the field is **genuinely absent** based on document legibility and text coverage ($0.85 - 0.95$). This ensures absence certainty is never confused with value certainty.

### Evaluation Considerations
On small test sets (such as 6 documents), statistical calibration metrics like Brier scores or reliability diagrams are illustrative rather than statistically conclusive. In production with $\ge 500$ documents, confidence scores should be benchmarked against ground truth using Brier Score ($BS = \frac{1}{N}\sum (f_t - o_t)^2$) and expected calibration error (ECE).

---

## 12. Testing Instructions

The repository features automated test coverage with 46 tests spanning unit, integration, and regression suites. All 46 tests pass cleanly.

### Run All Tests
```bash
python -m pytest
```
Or via Make:
```bash
make test
```

### Test Suite Breakdown
1. `tests/test_pdf_reader.py`: PyMuPDF ingestion, plain text reading, empty document detection, scanned PDF heuristics, and OCR fallback.
2. `tests/test_llm_client.py`: Mocked OpenAI API client, environment loading (`LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`), JSON decoding, markdown fence stripping, and timeout handling.
3. `tests/test_validator.py`: Business validation rules, credit sign auto-correction, schedule-only handling, conflicting values, unsigned document handling (null status, never inferred Pending), attachment preparation date distinction, synonymous approval status reconciliation, estimated amount extraction (Document 05), and Draft 2020-12 schema validation.
4. `tests/test_confidence.py`: Multi-factor evidence grounding, hallucination penalty, absence confidence, and OCR calibration factors.
5. `tests/test_extractor.py`: CLI execution, argument parsing, stdout/stderr stream separation, and exit codes.
6. `tests/test_benchmark_samples.py`: Schema and regression verification across all 6 live output files in `outputs/`.

---

## 13. Benchmark Results Across Six Sample Documents

Live extraction executed against all six benchmark PDFs:

| Sample Document | Primary Scenario | Extracted Amount | Impact Days | Approval Status | Needs Review | Schema Validated | Benchmark Match |
|---|---|---|---|---|---|---|---|
| **01_complete_approved_increase.pdf** | Standard approved increase | `+$18,500.00` | `+12` | `Approved` | `False` | Pass | 9/9 (100%) |
| **02_approved_price_decrease.pdf** | Approved deductive scope credit | `-$8,750.00` | `0` | `Approved` | `False` | Pass | 9/9 (100%) |
| **03_schedule_only_extension.pdf** | Approved time extension only | `null` | `+18` | `Approved` | `False` | Pass | 9/9 (100%) |
| **04_pending_approval_increase.pdf** | Pending owner price increase | `+$42,500.00` | `+7` | `Pending Approval` | `False` | Pass | 9/9 (100%) |
| **05_incomplete_missing_fields.pdf** | Incomplete draft / estimate only | `+$6,200.00` *(Est.)* | `null` | `Under Review` | `True` | Pass | 9/9 (100%) |
| **06_conflicting_values_needs_review.pdf** | Contradictory cover & worksheet | `null` | `null` | `null` | `True` | Pass | 9/9 (100%) |

### Benchmark Evaluation Summary:
- **Total Checked Fields**: **54 / 54** across 6 sample documents.
- **Strict Exact-Match Accuracy**: **53 / 54 (98.1%)** fields matching reference strings verbatim (Document 04 extracts `"Pending Approval"` reflecting explicit header text where ground truth lists `"Pending"`).
- **Semantic Agreement**: **54 / 54 (100.0%)** fields matching verified contractual intent and ground-truth values.
- **Missing Fields Correctly Handled**: **9 / 9 (100.0%)** legitimately omitted or conflicting fields correctly assigned `null`.
- **Remaining Mismatches**: **0**.
- **Document 03 Handling**: Both requester (`"Priya Shah, Project Architect"`) and complete scope description (`"During demolition in the west classroom wing, the contractor discovered concealed asbestos-containing floor adhesive..."`) are fully extracted and grounded.
- **Document 05 Handling**: Explicitly contains an estimated amount of `+$6,200.00`. Extracted as `6200.00` with lower confidence score (`0.71` < `0.85` threshold) and `needs_review: true`.
- **Document 06 Handling**: Discrepant contract numbers, dates, amounts, schedule impacts, and approvals are identified; date contradiction is reported because the document explicitly establishes conflict via review note.

### Detailed Case-by-Case Breakdown for All 6 Documents

#### Document 01: Standard Approved Cost & Schedule Increase (`01_complete_approved_increase.pdf`)
- **Document Nature**: Standard, fully executed AIA-style change order.
- **Project & Identifiers**: Riverside Medical Center Expansion (`RMC-2026-014`), Change Order `CO-017`.
- **Cases Tested & Validated**:
  1. *Complete Field Presence*: All 9 fields explicitly present; no missing or conflicting values.
  2. *Monetary Addition*: Net price increase stated as `$18,500.00`, parsed as positive float `18500.0`.
  3. *Schedule Addition*: Time impact stated as `12 additional calendar days`, parsed as integer `12`.
  4. *Party Extraction*: Requested by `"Elena Brooks, Project Manager"` — full name and professional designation preserved.
  5. *Approval Status*: Explicitly `"Approved"` with approval date `2026-10-18`.
- **Extraction Outcome**: 9/9 fields matched (Strict & Semantic), `needs_review: false`, high confidence (`1.00`).

#### Document 02: Deductive Change Order / Credit (`02_approved_price_decrease.pdf`)
- **Document Nature**: Approved scope reduction / credit memorandum.
- **Project & Identifiers**: Northgate Logistics Warehouse (`NGL-25-088`), Change Order `CO-009`.
- **Cases Tested & Validated**:
  1. *Negative Monetary Sign Convention*: Credit of `$8,750.00` correctly parsed as negative float `-8750.0`. Accounting credits must decrease contract value.
  2. *Zero Schedule Impact*: Schedule impact explicitly stated as `0 calendar days`. Parsed as integer `0` (strictly distinguished from `null` or unstated).
  3. *Party Extraction*: Requested by `"Marco Ruiz, Owner Representative"`.
  4. *No False Review Alarm*: Legitimate price reductions do not trigger false-positive review flags (`needs_review: false`).
- **Extraction Outcome**: 9/9 fields matched (Strict & Semantic), `needs_review: false`, high confidence (`1.00`).

#### Document 03: Schedule-Only Time Extension (`03_schedule_only_extension.pdf`)
- **Document Nature**: Environmental/unforeseen conditions time extension request without price change.
- **Project & Identifiers**: Cedar Grove School Renovation (`CGS-26-003`), Change Order `CO-006`.
- **Cases Tested & Validated**:
  1. *Legitimate Null Amount*: Document explicitly states `"No amount or price adjustment is stated in this document"`. Value is correctly `null` without error.
  2. *Schedule Extension*: Schedule impact of `18 calendar days` extracted as positive integer `18`.
  3. *Party Extraction*: `"Priya Shah, Project Architect"` preserved with full designation.
  4. *Detailed Scope Extraction*: Deep body extraction captures the specific hazardous condition (`concealed asbestos-containing floor adhesive`) rather than a generic subject-line summary.
  5. *Unicode Punctuation Normalization*: Hyphen variants normalized to standard ASCII hyphens.
  6. *Status Normalization*: `"APPROVED - TIME EXTENSION ONLY"` mapped cleanly to `"Approved"`.
- **Extraction Outcome**: 9/9 fields matched (Strict & Semantic), `needs_review: false`.

#### Document 04: Pending Owner Approval Request (`04_pending_approval_increase.pdf`)
- **Document Nature**: Proposed / unexecuted change order submitted for owner authorization.
- **Project & Identifiers**: Harbor Point Residences (`HPR-2026-022`), Change Order `CO-012`.
- **Cases Tested & Validated**:
  1. *Unexecuted Status Handling*: Document explicitly states `"PENDING OWNER APPROVAL"`. Preserved truthfully as `"Pending Approval"` (matching semantic intent of `"Pending"`).
  2. *Never Inferring Approval*: The engine never defaults an unexecuted request to "Approved".
  3. *Cost & Schedule Impact*: Amount parsed as `42500.0`, schedule parsed as `7` calendar days.
  4. *Party Extraction*: `"David Kim, Construction Manager"` extracted with professional role.
- **Extraction Outcome**: 8/9 Strict Match, 9/9 Semantic Agreement, `needs_review: false`.

#### Document 05: Incomplete Draft & Estimated Amount (`05_incomplete_missing_fields.pdf`)
- **Document Nature**: Preliminary contractor change proposal lacking standard prime contract metadata.
- **Project & Identifiers**: Pine Ridge Community Center, Change Order `CO-004`.
- **Cases Tested & Validated**:
  1. *Truthful Missing Field Nullification*: `contract_number`, `requested_by`, and `schedule_impact_days` are omitted from the document text. The pipeline assigns `null` with high absence confidence rather than guessing.
  2. *Estimated Amount Extraction*: Document states `"Estimated amount: ADD $6,200.00. The amount shown is an estimate and is not a final approved adjustment."` The pipeline extracts `6200.00` rather than nullifying it.
  3. *Heuristic Confidence Discount*: Because the amount is explicitly estimated, its confidence is discounted to `0.71` (< `0.85` threshold).
  4. *Approval Status*: Explicitly `"Under Review"`.
  5. *Mandatory Human Review*: `needs_review: true` triggered due to missing contract identifier and tentative estimate.
- **Extraction Outcome**: 9/9 fields matched (Strict & Semantic), `needs_review: true`.

#### Document 06: Cross-Section Audit Discrepancies (`06_conflicting_values_needs_review.pdf`)
- **Document Nature**: Multi-page audit test document with deliberate contradictions between cover sheet and Attachment B.
- **Project & Identifiers**: Westfield Water Treatment Upgrade, Change Order `CO-021`.
- **Cases Tested & Validated**:
  1. *Contract Number Contradiction*: Cover sheet states `CT-8841-A`; Attachment B states `CT-8841-B`. Reset to `null`.
  2. *Amount Contradiction*: Cover sheet states `$18,400.00`; Attachment B states `$14,800.00`. Reset to `null`.
  3. *Schedule Impact Contradiction*: Cover sheet states `15 days`; Attachment B states `10 days`. Reset to `null`.
  4. *Approval Status Contradiction*: Cover sheet states `Approved`; Attachment B states `Draft / Pending Review`. Reset to `null`.
  5. *Semantic Date Distinction*: Cover sheet date (`2026-11-04`) vs Attachment B prepared date (`2026-11-06`). An explicit review note states: `"The cover sheet and Attachment B disagree on contract number, date, amount, schedule impact, and final approval."` Because the document explicitly establishes conflict, date is set to `null`.
  6. *Confidence Zeroing*: All 5 contradictory fields receive `confidence: 0.00` with detailed conflict logs.
  7. *Safety Escalation*: `needs_review: true` enforced.
- **Extraction Outcome**: 9/9 fields matched (Strict & Semantic), `needs_review: true`.

### Strict Matching vs. Semantic Agreement
- **Strict Matching**: Character-for-character equality with reference strings (`53/54`, `98.1%`). The single strict difference is Document 04 (`"Pending Approval"` vs `"Pending"`), where both representations are grounded in document headers.
- **Semantic Agreement**: Evaluation verifying that domain meaning, signs, ISO dates, contractual statuses, and scope descriptions reflect reality without fabrication or loss of intent (`54/54`, `100.0%`).
- **Statistical Note**: Six synthetic documents confirm functional pipeline correctness across specific edge cases; empirical statistical generalization requires validation on hundreds of diverse historical change orders.

---

## 14. Failure Modes and Limitations

| Failure Mode | Impact | Mitigation in Pipeline |
|---|---|---|
| **Poor-quality Scanned PDFs** | Low or garbled digital text extraction | System computes character density per page; detects scanned PDFs and attempts OCR if enabled. If OCR is unavailable, halts with `ScannedPDFError` instead of fabricating data. |
| **OCR Mistakes & Letter Misreads** | Garbled numbers (e.g. `S` for `5` or `O` for `0`) | Multi-factor confidence applies OCR quality discount ($0.90$); Pydantic numeric validation rejects unparseable amounts. |
| **Handwritten Text & Signatures** | Digital text extractors and Tesseract skip or misread handwriting | **Limitation**: Tesseract OCR is designed primarily for printed/typeset text and does **not** reliably recognize cursive handwriting, handwritten annotations, or signatures. When handwritten text is encountered and uncertain, it is treated as an extraction limitation that sets `needs_review: true` for mandatory manual review, rather than guessing. |
| **Unsigned Documents** | Document lacks signatures or approvals | **Rule**: Never infer 'Pending' approval simply because a document or signature block is unsigned. If approval status is not explicitly stated in the document text, return `null` and flag `needs_review: true`. |
| **Missing Information** | Contract omits fields (e.g. no contract number) | Strict instruction never to hallucinate; missing fields are assigned `null` with absence confidence; missing critical identifiers trigger `needs_review: true`. |
| **Conflicting Contract Amounts** | Cover sheet says $18,400; attachment says $14,800 | Pipeline detects discrepancies across sections, returns `null` for amount, drops confidence to `0.0`, and sets `needs_review: true`. |
| **Attachment Preparation Dates vs Change Order Dates** | Attachment has preparation date different from change order | Distinguishes change-order dates from attachment preparation dates. Only reports a contradiction if the dates refer to the same semantic field or the document explicitly establishes a conflict (e.g. Document 06). |
| **Ambiguous or Non-standard Dates** | "10/11/12" (ambiguous day vs month) | Strict ISO `YYYY-MM-DD` parser. If a date cannot be unambiguously resolved, it is set to `null` with `needs_review: true`. |
| **Unsupported Document Layouts** | Complex multi-column or rotated drawings | PyMuPDF extracts text in reading blocks; if text density falls below threshold, document is routed through scanned detection. |
| **LLM Hallucinations** | Model invents unstated facts or approvals | Evidence grounding checks verify that quoted strings exist in the raw document text. Citations not found in text receive a severe confidence penalty. |
| **API Timeouts & Rate Limits** | Provider downtime or rate limit exhaustion | Handled via exponential backoff with `max_retries=2` and explicit `LLMAPIError` exceptions. Never falls back to fabricated results. |
| **Heuristic Confidence Estimates** | Scores are heuristic estimates, not probabilities | Clearly identified as heuristic indicators until empirically calibrated on $\ge 500$ verified documents. Discards ungrounded LLM self-confidence. |

---

## 15. Security Considerations

1. **Zero Secret Leakage**:
   - API keys are read exclusively from environment variables or `.env`.
   - `.gitignore` strictly prevents committing `.env`, `.env.local`, or any credentials.
   - CLI error messages sanitize output and never print raw environment tokens or authentication headers.
2. **Strict Output Hygiene**:
   - Pydantic models enforce `extra="forbid"`, preventing internal prompts, tokens, or system metadata from escaping into public JSON outputs.
3. **No Code Execution**:
   - Document text is handled purely as data; no `eval()` or shell execution is performed on ingested text.

---

## 16. Future Improvements

1. **Table Extraction Engine**: Integrate specialized tabular extraction (e.g. PyMuPDF `find_tables()` or layout-aware vision models) for complex multi-page schedule-of-values tables.
2. **Visual Multimodal Support**: Support multimodal LLMs (e.g. GPT-4o / Gemini Flash Vision) to process architectural blueprints, stamps, and handwritten signatures directly.
3. **Automated Calibration with Large Corpora**: Implement Platt scaling / isotonic regression on a verified dataset of 1,000+ construction change orders to calibrate confidence scores to exact statistical probabilities.
4. **Asynchronous Batch Processing**: Provide an `asyncio`-driven worker queue for high-throughput batch ingestion of thousands of change order PDFs.
