<img width="1898" height="1012" alt="image" src="https://github.com/user-attachments/assets/253c7530-8de8-41e1-bd0d-024b5bc8ebe7" />


# LLM-Judge Fake Receipt Detector

A forensic document analysis system that uses multiple OpenAI vision models as
independent judges to classify receipt images as **FAKE**, **REAL**, or
**UNCERTAIN**. Includes a full-featured dashboard, dataset statistics, batch
evaluation, and comprehensive observability via Langfuse.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [How It Works](#how-it-works)
4. [Project Structure](#project-structure)
5. [Quick Start](#quick-start)
6. [Configuration](#configuration)
7. [Ways to Run](#ways-to-run)
8. [Dashboard Guide](#dashboard-guide)
9. [Dataset Management](#dataset-management)
10. [Langfuse Observability](#langfuse-observability)
11. [Test Suite](#test-suite)
12. [Judge Output Schema](#judge-output-schema)
13. [Security Notes](#security-notes)

---

## Overview

The system answers one question per receipt image: **is this receipt forged?**

It does so by:

- Extracting lightweight image features (blur, brightness, dimensions, OCR total)
- Sending the image to **3 OpenAI vision models** acting as forensic judges with
  different personas (strict / balanced / lenient)
- Aggregating the three verdicts via **majority vote**
- Logging every step — prompts, completions, token usage, latency, scores — to
  **Langfuse** via a local MCP server

---

## Dataset Analisis

- Image dimensions and file size distributions.
- Aspect ratio distributions
- Sharpnes, Brightness and Contrast distributions

<img width="1895" height="993" alt="image" src="https://github.com/user-attachments/assets/15a9b5b7-304a-4068-8589-8c827f871e36" />

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT LAYER                             │
│  Receipt image (.png/.jpg)   +   OCR text (.txt, optional)      │
└───────────────────┬─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FEATURE EXTRACTION                           │
│                                                                 │
│  image_basic_stats()         blur_variance_of_laplacian()       │
│  → width, height,            → sharpness proxy                  │
│    aspect_ratio, file_kb     (cv2 Laplacian variance)           │
│                                                                 │
│  brightness_contrast()       extract_total_from_text()          │
│  → mean brightness,          → parsed receipt total             │
│    contrast std              (keyword + next-line search)        │
└───────────────────┬─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                      JUDGE PANEL  (3× parallel)                 │
│                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │    judge_1      │  │    judge_2      │  │    judge_3      │ │
│  │  gpt-4o-mini    │  │  gpt-4.1-mini   │  │    gpt-4o       │ │
│  │  temp=0.2       │  │  temp=0.4       │  │  temp=0.7       │ │
│  │  strict /       │  │  balanced /     │  │  lenient /      │ │
│  │  skeptical      │  │  artifact-aware │  │  benefit-of-    │ │
│  │                 │  │                 │  │  the-doubt      │ │
│  │ → label         │  │ → label         │  │ → label         │ │
│  │ → confidence    │  │ → confidence    │  │ → confidence    │ │
│  │ → reasons[]     │  │ → reasons[]     │  │ → reasons[]     │ │
│  │ → flags[]       │  │ → flags[]       │  │ → flags[]       │ │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘ │
└───────────┼────────────────────┼────────────────────┼──────────┘
            │                    │                    │
            └────────────────────┼────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                      MAJORITY VOTE                              │
│                                                                 │
│   Needs ≥ 2 of 3 for FAKE or REAL; otherwise → UNCERTAIN        │
│                                                                 │
│   Examples:                                                     │
│   [FAKE, FAKE, REAL]     → FAKE      (2/3)                      │
│   [REAL, REAL, UNCERTAIN]→ REAL      (2/3)                      │
│   [FAKE, REAL, UNCERTAIN]→ UNCERTAIN (split)                    │
└───────────────────┬─────────────────────────────────────────────┘
                    │
          ┌─────────┴──────────┐
          │                    │
          ▼                    ▼
┌──────────────────┐  ┌─────────────────────────────────────────┐
│   RESULT CACHE   │  │           LANGFUSE  (via MCP)            │
│                  │  │                                          │
│ data/            │  │  Trace ──────────────────────────────┐  │
│  eval_cache.json │  │    ├─ span: dataset_analysis          │  │
│                  │  │    ├─ generation: judge_1             │  │
│  Persists across │  │    ├─ generation: judge_2             │  │
│  sessions; shown │  │    ├─ generation: judge_3             │  │
│  in Browse tab   │  │    ├─ span: vote_aggregation          │  │
│                  │  │    └─ scores (8×):                    │  │
└──────────────────┘  │        judge_N_confidence (0–100)     │  │
                       │        judge_N_correctness (0/0.5/1)  │  │
                       │        final_correct  (0 or 1)        │  │
                       │        inter_judge_agreement (0 or 1) │  │
                       └─────────────────────────────────────────┘
```

---

## How It Works

### 1. Feature extraction (lightweight, CPU-only)

Before any LLM call the system extracts deterministic signals from the image and
optional OCR file:

| Feature | Method | Purpose |
|---|---|---|
| `width`, `height` | PIL `Image.size` | Dimension sanity check |
| `aspect_ratio` | w / h | Long vs short receipt |
| `file_kb` | `stat().st_size` | Compression proxy |
| `blur_variance` | OpenCV Laplacian variance | Sharpness / scan quality |
| `brightness_mean` | Pixel mean / 255 | Exposure level |
| `contrast_std` | Pixel std / 255 | Dynamic range |
| `ocr_total` | Regex keyword+next-line scan | Parsed receipt total |

> OCR is optional. If no `.txt` file exists, `ocr_total` is `null` and the
> judges rely solely on the image.

### 2. Judge prompt

Each judge receives the same base system prompt but with a different **persona**
injected. The persona shifts the judge's prior, giving the ensemble diverse
perspectives:

```
You are an expert forensic document examiner evaluating whether a receipt
image is forged.
Persona: {persona}

Return ONLY valid JSON with this schema:
{
  "label": "FAKE|REAL|UNCERTAIN",
  "confidence": 0-100,
  "reasons": ["short reason 1", "short reason 2"],
  "flags": ["optional tag"]
}
```

The image is sent as a base-64 data URL in the `image_url` content block.
The `response_format: json_object` mode is used where supported; otherwise the
JSON is extracted from free text.


<img width="1849" height="961" alt="image" src="https://github.com/user-attachments/assets/ce1f30be-63cb-4f70-b64d-499aa404ddf7" />

### 3. Majority vote

```
judges = [judge_1.label, judge_2.label, judge_3.label]

FAKE      if count(FAKE)      >= 2
REAL      if count(REAL)      >= 2
UNCERTAIN otherwise (split or all-uncertain)
```

### 4. Scoring (for evaluation runs)

When a ground-truth label is known, eight numeric scores are attached to the
Langfuse trace:

| Score name | Value | Meaning |
|---|---|---|
| `judge_N_confidence` | 0–100 | Raw confidence the judge reported |
| `judge_N_correctness` | 1.0 / 0.5 / 0.0 | Correct / uncertain / wrong |
| `final_correct` | 1.0 or 0.0 | Final verdict matches ground truth |
| `inter_judge_agreement` | 1.0 or 0.0 | All three judges agreed |

---

<img width="1849" height="929" alt="image" src="https://github.com/user-attachments/assets/e9f60fc0-933a-4000-9e4e-3e1f273c3b32" />
<img width="1849" height="967" alt="image" src="https://github.com/user-attachments/assets/6781a985-b549-46c8-b74e-372f81dbe390" />


## Project Structure

```
testthekey/
│
├── app.py                          # ★ Main dashboard (3-tab Streamlit UI)
├── streamlit_app.py                # Legacy single-receipt UI
│
├── src/
│   ├── config.py                   # Settings dataclass + env loading
│   ├── dataset.py                  # load_label_table, build_records,
│   │                               #   extract_total_from_text, image_basic_stats
│   ├── features.py                 # blur_variance_of_laplacian, brightness_contrast
│   ├── judges.py                   # JudgeConfig, run_judge, prompt builder
│   ├── vote.py                     # majority_vote, vote_tally
│   ├── eval.py                     # JudgeResult / AggregatedResult dataclasses
│   ├── langfuse_mcp_client.py      # HTTP client for the Langfuse MCP server
│   └── langfuse_logger.py          # Legacy direct Langfuse SDK logger
│
├── scripts/
│   ├── run_full_eval_langfuse.py   # ★ Batch eval + full Langfuse logging
│   ├── run_eval20.py               # Batch eval on a sampled CSV
│   ├── run_one.py                  # Single-receipt CLI run
│   ├── sample_eval20.py            # Sample 20 balanced receipts from a dataset
│   └── summarize_dataset.py        # CLI dataset statistics report
│
├── tests/
│   ├── conftest.py                 # sys.path setup
│   ├── test_vote.py                # 12 unit tests for voting logic
│   ├── test_dataset.py             # 22 tests: labels, paths, OCR, image stats
│   ├── test_judges.py              # 18 tests: parsing, normalisation (mocked API)
│   ├── test_features.py            # 9 tests: blur, brightness
│   └── test_integration.py         # 8 real OpenAI API tests (auto-skipped if no key)
│
├── notebooks/
│   └── 01_dataset_exploration.ipynb
│
├── data/
│   ├── sample/
│   │   ├── labels.csv              # Ground-truth manifest (image, forged, label)
│   │   ├── images/                 # Receipt images (.png)
│   │   └── ocr/                    # Per-image OCR text files (.txt)
│   └── eval_cache.json             # ★ Auto-created: persisted evaluation results
│
├── docs/
│   └── takehome_instructions.pdf
│
├── .env                            # Local secrets (never commit)
├── .env.example                    # Template
├── .gitignore
├── pytest.ini
└── requirements.txt
```

---

## Quick Start

### 1. Create and activate a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:

```env
OPENAI_API_KEY=sk-proj-...          # required

# Optional — only needed if using direct SDK logging (legacy streamlit_app.py)
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=http://localhost:3000

# Judge models (defaults shown)
JUDGE_MODEL_1=gpt-4o-mini
JUDGE_MODEL_2=gpt-4.1-mini
JUDGE_MODEL_3=gpt-4o
```

> **Langfuse via MCP** (used by `app.py` and `run_full_eval_langfuse.py`) does
> **not** need `LANGFUSE_*` env vars — authentication is handled internally by
> the MCP server at `http://localhost:8005/mcp/`.

### 4. Launch the dashboard

```bash
streamlit run app.py
```

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | **Required.** OpenAI API key |
| `JUDGE_MODEL_1` | `gpt-4o-mini` | Model for the strict judge |
| `JUDGE_MODEL_2` | `gpt-4.1-mini` | Model for the balanced judge |
| `JUDGE_MODEL_3` | `gpt-4o` | Model for the lenient judge |
| `OPENAI_TIMEOUT_SECONDS` | `60` | Per-request timeout |
| `LANGFUSE_PUBLIC_KEY` | — | Optional (legacy logger only) |
| `LANGFUSE_SECRET_KEY` | — | Optional (legacy logger only) |
| `LANGFUSE_HOST` | — | Optional (legacy logger only) |

---

## Ways to Run

### A — Dashboard (recommended)

```bash
streamlit run app.py
```

Full three-tab UI. See [Dashboard Guide](#dashboard-guide) below.

### B — Batch evaluation + full Langfuse logging

```bash
python -m scripts.run_full_eval_langfuse
```

Processes every row in `data/sample/labels.csv`. Logs to Langfuse:

- Prompt version in Prompt Management (`receipt-forensic-judge`)
- Dataset `receipt-detection-sample` with expected outputs
- One trace per receipt with spans, generations, and 8 scores
- A summary event with accuracy, avg confidence, disagreement rate

### C — Single receipt (CLI)

```bash
python scripts/run_one.py \
  --image data/sample/images/X00016469622.png \
  --ground_truth FAKE
```

### D — Sample + batch evaluate a larger dataset

```bash
# 1. Sample 20 balanced receipts
python scripts/sample_eval20.py \
  --labels data/findit2/train.txt \
  --seed 7 \
  --out_csv eval_samples/eval_20.csv

# 2. Run evaluation on them
python scripts/run_eval20.py \
  --eval_csv eval_samples/eval_20.csv \
  --image_dir data/findit2/images \
  --out_json reports/eval20_results.json
```

### E — CLI dataset summary report

```bash
python scripts/summarize_dataset.py \
  --labels data/sample/labels.csv \
  --image_dir data/sample/images \
  --ocr_dir data/sample/ocr \
  --out_dir reports
```

### F — Jupyter exploration notebook

```bash
jupyter notebook notebooks/01_dataset_exploration.ipynb
```

Auto-detects the project root regardless of the directory Jupyter is launched
from.

---

## Dashboard Guide

```
┌──────────────────────────────────────────────────────────────────┐
│  SIDEBAR                  │  MAIN AREA                          │
│                           │                                     │
│  Data source              │  ┌─────────────────────────────┐   │
│  ○ Sample data            │  │  📊 Dataset Stats           │   │
│  ○ Custom path            │  ├─────────────────────────────┤   │
│                           │  │  🔍 Browse                  │   │
│  [🔄 Reload dataset]      │  ├─────────────────────────────┤   │
│                           │  │  ▶  Evaluate                │   │
│  ───────────────────────  │  └─────────────────────────────┘   │
│  12 receipts loaded        │                                     │
│  🔴 FAKE: 6  🟢 REAL: 6   │                                     │
│  ✅ Evaluated: 8 / 12     │                                     │
│                           │                                     │
│  Langfuse:                │                                     │
│  http://20.119.121.220:3000│                                    │
└──────────────────────────────────────────────────────────────────┘
```

### Tab: 📊 Dataset Stats

Displays automatically on load — **no API calls, no cost**:

| Section | Charts |
|---|---|
| KPI row | Total receipts · FAKE count · REAL count · With OCR |
| Row 1 | REAL vs FAKE bar chart · Receipt totals histogram (by label) |
| Row 2 | Totals by label box plot (are fakes skewed?) · File size box plot |
| Row 3 | Resolution scatter (W×H) · Blur box plot · Brightness box plot · Aspect ratio box plot |
| Summary table | `describe()` statistics grouped by label |

### Tab: 🔍 Browse

```
Filter by label: [All ▼]    Select receipt: [✅ X00016469622.png  [GT: FAKE] ▼]
─────────────────────────────────────────────────────────────────────
│  Receipt image              │  Ground truth: FAKE (red)           │
│                             │                                      │
│  [image displayed here]     │  Image features:                     │
│                             │   dimensions: 461 × 933 px           │
│                             │   aspect_ratio: 0.494                │
│                             │   file_kb: 225.3                     │
│                             │   blur_variance: 2455.4              │
│                             │   brightness: 0.9646                 │
│                             │   ocr_total: 88.91                   │
│                             │                                      │
│                             │  Verdict: UNCERTAIN  ❌              │
│                             │  FAKE: 1  REAL: 0  UNCERTAIN: 2      │
│                             │  ▶ Judge details (expandable)        │
─────────────────────────────────────────────────────────────────────
```

- `✅` = already evaluated (result shown instantly from cache)
- `⬜` = not yet evaluated ("Go to ▶ Evaluate" message shown)
- Judge details expand to show per-judge label, confidence, reasons, latency

### Tab: ▶ Evaluate

```
Total: 12     Evaluated: 8     Pending: 4

Run mode: ● Run missing only   ○ Run all (overwrite)

[▶  Start evaluation]

████████████████████░░░░  [3/4]  X00016469623.png  (GT=REAL)
⚙️ Running 3 judges on X00016469623.png…

┌──────────────────┬────┬──────────┬───┬──────┬──────┬───────────┐
│ Receipt          │ GT │ Verdict  │ ✓ │ FAKE │ REAL │ UNCERTAIN │
├──────────────────┼────┼──────────┼───┼──────┼──────┼───────────┤
│ X00016469622.png │FAKE│ UNCERTAIN│ ❌│  1   │  0   │     2     │
│ X00016469623.png │REAL│ REAL     │ ✅│  0   │  2   │     1     │
│ ...              │    │          │   │      │      │           │
└──────────────────┴────┴──────────┴───┴──────┴──────┴───────────┘
```

- Results are saved to `data/eval_cache.json` **after every receipt** — safe to
  interrupt and resume
- Langfuse logging happens silently; if the MCP server is unreachable a warning
  is shown but evaluation continues
- After completion: Accuracy · Correct count · Uncertain/Wrong summary

---

## Dataset Management

### labels.csv format

```csv
image,forged,label
X00016469622.png,1,FAKE
X00016469623.png,0,REAL
your_new_receipt.png,1,FAKE
```

| Column | Values | Notes |
|---|---|---|
| `image` | filename | Must match a file in the images directory |
| `forged` | `1` or `0` | `1` = FAKE, `0` = REAL |
| `label` | `FAKE` / `REAL` | Auto-derived from `forged`; including it is fine |

### Adding new receipts

```
data/sample/
├── images/
│   ├── X00016469622.png        ← existing
│   └── your_new_receipt.png    ← 1. drop image here
├── ocr/
│   ├── X00016469622.txt        ← existing
│   └── your_new_receipt.txt    ← 2. drop OCR text here (optional)
└── labels.csv                  ← 3. add a row here
```

After adding files:

1. Click **🔄 Reload dataset** in the sidebar (clears the feature cache)
2. Go to **▶ Evaluate** → **Run missing only** → all new receipts are evaluated
3. Results appear immediately in **🔍 Browse**

> OCR is optional. Without it, `ocr_total` will be `null` but the LLM judges
> still run using the image.

### Using a different dataset

In the sidebar select **Custom path** and provide:

- Labels CSV path (must have `image` and `forged` columns)
- Images directory path
- OCR directory path (optional)

The `scripts/sample_eval20.py` script can sample a balanced subset from any
large dataset with a `train.txt` / CSV format.

---

## Langfuse Observability

The project uses a **local MCP server** at `http://localhost:8005/mcp/` to send
data to the Langfuse instance at `http://20.119.121.220:3000`.

```
Your code                    MCP Server              Langfuse
    │                            │                       │
    │  HTTP POST /mcp/           │                       │
    │  (JSON-RPC 2.0 + SSE)      │                       │
    │──langfuse_log_generation──▶│                       │
    │                            │──Langfuse SDK call───▶│
    │                            │                       │  stores trace
    │◀──────── result ───────────│                       │  generation
    │                            │                       │  score
```

<img width="1849" height="975" alt="image" src="https://github.com/user-attachments/assets/e5262000-da4c-42b6-ba9e-98a524bdb1c2" />
<img width="1849" height="863" alt="image" src="https://github.com/user-attachments/assets/3e7f7b3d-05e7-4af1-8e43-94b8c7a2c253" />

<img width="1849" height="975" alt="image" src="https://github.com/user-attachments/assets/1391d2d3-ae71-4f28-a1e2-ab950f95070c" />


### What gets logged per evaluation run

```
Langfuse Prompt Management
└── receipt-forensic-judge  (versioned text prompt)

Dataset: receipt-detection-sample
├── item: X00016469622.png  (input + expected_output)
└── item: X00016469623.png

Dataset run: eval-3judges-gpt4o
├── run item → trace be3d81f3…
└── run item → trace 9c63670a…

Trace: receipt_eval_X00016469622.png
├── span:       dataset_analysis     ← image stats + OCR features
├── generation: judge_1              ← model · prompt · response · tokens · latency
├── generation: judge_2
├── generation: judge_3
├── span:       vote_aggregation     ← labels list → final_label + tally
└── scores (8):
      judge_1_confidence   · judge_1_correctness
      judge_2_confidence   · judge_2_correctness
      judge_3_confidence   · judge_3_correctness
      final_correct        · inter_judge_agreement

Event: eval_summary
└── accuracy · avg_confidence · disagreement_rate
```

### MCP client

`src/langfuse_mcp_client.py` provides a thin Python wrapper over the MCP
transport. It handles the `initialize` handshake automatically on first use:

```python
from src.langfuse_mcp_client import LangfuseMCPClient

lf = LangfuseMCPClient()
lf.auth_check()                                  # verify connectivity
lf.log_generation(observation={...}, trace={...})
lf.create_score(trace_id, "accuracy", 0.85)
lf.dataset_add_item(dataset_name=..., input=..., expected_output=...)
```

---

## Test Suite

```
tests/
├── conftest.py           sys.path setup
├── test_vote.py          12 tests — majority_vote, vote_tally
├── test_dataset.py       22 tests — label loading, path resolution, OCR parsing
├── test_judges.py        18 tests — JSON parsing, output normalisation (mocked API)
├── test_features.py       9 tests — blur variance, brightness contrast
└── test_integration.py    8 tests — real OpenAI API calls (auto-skipped if no key)
```

### Run all unit tests (no API calls, fast)

```bash
pytest tests/ -v
```

### Run integration tests (calls OpenAI, ~$0.01)

```bash
pytest tests/test_integration.py -v -m integration
```

Integration tests auto-skip when `OPENAI_API_KEY` is not set. They use
`gpt-4o-mini` (cheapest vision model) and verify:

- Schema validity of judge responses
- That a strict judge does not call a known-FAKE receipt REAL
- That a strict judge does not call a known-REAL receipt FAKE
- Full 3-judge pipeline produces a valid tally summing to 3

---

## Judge Output Schema

Every judge call returns a validated, normalised dict:

```json
{
  "label":      "FAKE | REAL | UNCERTAIN",
  "confidence": 85.0,
  "reasons": [
    "Font inconsistency on line items",
    "Total does not match item sum"
  ],
  "flags": ["pixel_artifact", "font_mismatch"]
}
```

| Field | Type | Constraints |
|---|---|---|
| `label` | string | One of `FAKE`, `REAL`, `UNCERTAIN` |
| `confidence` | float | 0–100 (clamped); 100 = fully certain |
| `reasons` | list[string] | 1–5 short, observable statements |
| `flags` | list[string] | 0–8 optional categorical tags |

`_normalize_output()` in `src/judges.py` enforces all constraints and provides
safe defaults if the model returns malformed JSON.

---

## Security Notes

- **Never commit `.env`** — it is listed in `.gitignore`
- **Never commit API keys** to version control or chat; rotate immediately if
  exposed
- The `.env.example` file contains only placeholder values and is safe to commit
- The sample receipt images are included for development/testing only

---

## Requirements

```
openai>=1.40.0          LLM calls + structured JSON output
langfuse>=3.0.0,<4.0.0  Direct SDK (legacy logger)
python-dotenv>=1.0.1    .env loading
pandas>=2.1.0           Label tables + dataset stats
numpy>=1.26.0           Numerical operations
matplotlib>=3.8.0       Dashboard charts
Pillow>=10.0.0          Image loading + stats
opencv-python>=4.9.0    Blur detection (Laplacian variance)
streamlit>=1.35.0       Dashboard UI
tqdm>=4.66.0            Progress bars (batch scripts)
pytest>=8.0.0           Test runner
httpx                   MCP HTTP transport (installed with openai)
```
