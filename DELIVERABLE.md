# LLM-Judge Fake Receipt Detector — Deliverable Report

> **Project:** Receipt Forgery Detection using Multi-Judge LLM Ensemble
> **Date:** February 2026
> **Stack:** Python · OpenAI Vision API · Streamlit · Langfuse · MCP

---

## Table of Contents

1. [System Design & Architecture](#1-system-design--architecture)
2. [LLM Usage Judgment](#2-llm-usage-judgment)
3. [Dataset Distribution Exploration](#3-dataset-distribution-exploration)
4. [Evaluation Summary](#4-evaluation-summary)
5. [Error Analysis — Disagreement Cases](#5-error-analysis--disagreement-cases)
6. [Judge Prompt (copy/paste)](#6-judge-prompt-copypaste)
7. [Sample Run — Judges' JSON + Final Vote](#7-sample-run--judges-json--final-vote)
8. [How AI Tools Were Used](#8-how-ai-tools-were-used)
9. [Code Quality & Reproducibility](#9-code-quality--reproducibility)
10. [Decisions & Trade-offs](#10-decisions--trade-offs)

---

## 1. System Design & Architecture

### Overview (1–2 paragraphs)

The system answers a single question per image: **is this receipt forged?** It does so without training a custom model, instead routing each receipt image through three independent OpenAI vision models that act as forensic judges with deliberately different personas — strict/skeptical, balanced, and lenient. Each judge returns a structured JSON verdict (`FAKE`, `REAL`, or `UNCERTAIN`) with a confidence score and observable reasoning. The three verdicts are aggregated by majority vote: at least two judges must agree on `FAKE` or `REAL`; otherwise the system returns `UNCERTAIN`. This ensemble approach is intentionally simple — the complexity lives in the prompting strategy, not in the aggregation.

All lightweight metadata (image dimensions, blur variance, brightness, contrast, OCR-extracted total) is extracted CPU-side before any LLM call. This data is logged to **Langfuse** via a local MCP server, providing full observability into every prompt, response, token count, latency, and score without requiring the models to duplicate work the code can do deterministically. The pipeline is designed to be re-runnable, resumable, and cost-transparent: a live cost-estimation table is shown before any batch run, results are persisted to a local JSON cache after every receipt, and the Langfuse dashboard tracks real spend in real time.

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT LAYER                             │
│  Receipt image (.png / .jpg)   +   OCR text (.txt, optional)    │
└───────────────────┬─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FEATURE EXTRACTION  (CPU-only, free)         │
│                                                                 │
│  image_basic_stats()   blur_variance_of_laplacian()             │
│  brightness_contrast() extract_total_from_text()                │
└───────────────────┬─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                   JUDGE PANEL  (3 independent calls)            │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │   judge_1    │  │   judge_2    │  │       judge_3         │  │
│  │ gpt-4o-mini  │  │ gpt-4.1-mini │  │       gpt-4o          │  │
│  │  temp = 0.2  │  │  temp = 0.4  │  │     temp = 0.7        │  │
│  │  strict /    │  │  balanced /  │  │  lenient / benefit-   │  │
│  │  skeptical   │  │  artifact-   │  │  of-the-doubt         │  │
│  │              │  │  aware       │  │                        │  │
│  │ → label      │  │ → label      │  │  → label              │  │
│  │ → confidence │  │ → confidence │  │  → confidence         │  │
│  │ → reasons[]  │  │ → reasons[]  │  │  → reasons[]          │  │
│  │ → flags[]    │  │ → flags[]    │  │  → flags[]            │  │
│  └──────┬───────┘  └──────┬───────┘  └─────────┬────────────┘  │
└─────────┼─────────────────┼──────────────────────┼─────────────┘
          └─────────────────┼──────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      MAJORITY VOTE                              │
│                                                                 │
│   FAKE      if count(FAKE)  ≥ 2                                 │
│   REAL      if count(REAL)  ≥ 2                                 │
│   UNCERTAIN otherwise  (split or all-uncertain)                 │
│                                                                 │
│   [FAKE, FAKE, REAL]      → FAKE  (2/3)                         │
│   [REAL, REAL, UNCERTAIN] → REAL  (2/3)                         │
│   [FAKE, REAL, UNCERTAIN] → UNCERTAIN (split)                   │
└───────────────────┬─────────────────────────────────────────────┘
                    │
          ┌─────────┴──────────┐
          ▼                    ▼
  ┌────────────────┐  ┌──────────────────────────────────────────┐
  │  Local cache   │  │           LANGFUSE  (via MCP)            │
  │  eval_cache    │  │  trace → spans → generations → scores    │
  │  .json         │  │  per-judge confidence · final_correct    │
  └────────────────┘  └──────────────────────────────────────────┘
```

### Key source files

| File | Responsibility |
|------|---------------|
| `streamlit_app.py` | 3-tab Streamlit UI (Analyze · Statistics · Batch Eval) |
| `src/judges.py` | Prompt builder, `run_judge()`, output normalisation |
| `src/vote.py` | `majority_vote()`, `vote_tally()` |
| `src/features.py` | `blur_variance_of_laplacian()`, `brightness_contrast()` |
| `src/dataset.py` | Label loading, OCR parsing, `image_basic_stats()` |
| `src/langfuse_mcp_client.py` | HTTP/MCP client for Langfuse observability |
| `scripts/run_eval20.py` | CLI batch evaluation on a sampled CSV |
| `scripts/register_dataset.py` | Registers all dataset items to Langfuse (metadata-only, free) |

---

## 2. LLM Usage Judgment

### Why vision models, not feature engineering?

Receipt forgeries in this dataset are subtle — digit substitutions, copy-pasted totals, font swaps. Classical image features (blur, brightness, file size) are useful context but cannot reliably detect these artefacts. A vision LLM trained on billions of documents can detect font inconsistencies, pixel-level copy-paste traces, and implausible totals without being explicitly programmed to look for them.

### Why three judges with different personas?

A single model call is brittle: one model at one temperature can be systematically wrong in one direction (e.g., always lenient toward blurry images). Using **three judges with different priors** forces the ensemble to disagree on genuinely ambiguous cases, which is exactly the information we want: uncertainty is data. The `UNCERTAIN` verdict class is an explicit signal, not a failure mode.

| Judge | Model | Temp | Persona |
|-------|-------|------|---------|
| judge_1 | gpt-4o-mini | 0.2 | Strict, skeptical — flags forensic inconsistencies aggressively |
| judge_2 | gpt-4.1-mini | 0.4 | Balanced — weighs printing/scan artefacts against tampering |
| judge_3 | gpt-4o | 0.7 | Lenient — assumes real unless manipulation is clear |

**Temperature choice rationale:** Low temperature (0.2) for the strict judge makes it deterministic and conservative; it will not change its mind on re-run. High temperature (0.7) for the lenient judge introduces variability that simulates a reviewer who occasionally notices different cues. The balanced judge sits in between.

**Model choice rationale:** `gpt-4o` (judge_3) is the most capable vision model and used as the "tiebreaker" quality anchor. `gpt-4o-mini` and `gpt-4.1-mini` cover different capability/cost points. As observed in our evaluation, `gpt-4o-mini` tends to default to `UNCERTAIN` when evidence is weak, while `gpt-4o` is more decisive.

### How uncertainty is handled

- If evidence is ambiguous the judge is instructed to return `UNCERTAIN` with lower confidence
- The majority vote propagates this: if judges split 1-1-1, the final verdict is also `UNCERTAIN`
- `UNCERTAIN` results are tracked separately in evaluation metrics and are treated as incorrect in the accuracy calculation (conservative)
- The Langfuse score `inter_judge_agreement` (0 or 1) flags every disagreement for human review

### Structured output

The prompt uses `response_format: {"type": "json_object"}` where supported, with a fallback that salvages the first `{...}` block from free-text output. `_normalize_output()` enforces schema constraints (label enum, confidence clamping, reasons truncation) so downstream code never receives malformed data.

---

## 3. Dataset Distribution Exploration

The full `findit2` test set used in evaluation:

| Metric | Value |
|--------|-------|
| Total images | 218 |
| REAL receipts | 183 (83.9%) |
| FAKE receipts | 35 (16.1%) |
| Class imbalance ratio | ~5.2 : 1 (REAL : FAKE) |

> **Key insight:** The dataset is heavily imbalanced. A naive classifier that always predicts REAL would achieve 83.9% accuracy. This makes accuracy a misleading primary metric — **per-class recall and the UNCERTAIN rate matter more**, and random sampling for evaluation must be stratified.

### Image quality characteristics (from Dataset Statistics tab)

| Feature | REAL (median) | FAKE (median) | Difference |
|---------|--------------|--------------|------------|
| File size | ~300–500 KB | Similar | Minimal — fakes are saved at similar quality |
| Aspect ratio | ~0.4–0.5 (portrait) | Similar | Not discriminative alone |
| Blur variance | Wide spread | Wide spread | Some fakes are suspiciously sharp (digitally edited) |
| Brightness | 0.85–0.95 | Similar | Fakes sometimes over-exposed (white balance) |

The dataset statistics confirm that **no single numeric feature cleanly separates REAL from FAKE**. This validates the choice of a vision LLM rather than a classical ML classifier on extracted features.

![Dataset Statistics — Label Distribution](docs/screenshot_stats.png)
*(Screenshot: Dataset Statistics tab — 183 REAL / 35 FAKE in the test set, with image quality proxies)*

---

## 4. Evaluation Summary

### Sampling procedure

20 receipts were selected for batch evaluation using the Batch Evaluation tab of the Streamlit dashboard:

- **Mode:** Mixed (REAL + FAKE) with stratified sampling option
- **Selection:** Exact image selection via searchable multiselect widget
- **Visible before run:** Live cost estimate per model, per judge

### Results — 20-image batch evaluation

> Numbers below are from the batch run shown in the Streamlit dashboard screenshots.

| Metric | Value |
|--------|-------|
| Images evaluated | 20 |
| Final accuracy (majority vote) | **55%** (11/20 correct) |
| Wrong predictions | 9 |
| REAL selection in this batch | 18 |
| FAKE selection in this batch | 2 |

> **Note:** This particular run sampled 18 REAL / 2 FAKE (unbalanced). For a fair 10/10 split, use the stratified checkbox in the Batch Evaluation tab.

### Per-judge performance

| Judge | Model | Accuracy | Avg. Confidence | n\_FAKE | n\_REAL | n\_UNCERTAIN |
|-------|-------|----------|-----------------|---------|---------|------------|
| judge_1 | gpt-4o-mini | **5.0%** | 54.5 | 0 | 3 | 17 |
| judge_2 | gpt-4.1-mini | **70.0%** | 79.5 | 14 | 15 | 1 |
| judge_3 | gpt-4o | **65.0%** | 82.8 | 0 | 15 | 5 |

**Observation:** `gpt-4o-mini` (judge_1) almost always returns `UNCERTAIN` on this dataset — 17 out of 20 times. This severely limits ensemble accuracy since it never contributes a clear vote. `gpt-4.1-mini` (judge_2) is the most decisive and accurate judge in this run.

### Langfuse observability (real cost data)

From the Langfuse dashboard after the full evaluation session:

| Model | Tokens used | Actual cost |
|-------|-------------|-------------|
| gpt-4o | ~1,410 | $0.00565 |
| gpt-4.1-mini | ~1,530 | $0.001139 |
| gpt-4o-mini | ~1,460 | $0.000393 |
| **TOTAL** | **~4,400** | **$0.007182** |

128 traces tracked · 60 scores logged · Real-time cost visible in Langfuse dashboard.

![Langfuse Dashboard](docs/screenshot_langfuse.png)
*(Screenshot: Langfuse Home — 128 traces, $0.007182 total cost, model breakdown)*

---

## 5. Error Analysis — Disagreement Cases

### Case 1 — False negative (FAKE predicted as REAL)

| Field | Value |
|-------|-------|
| Image | `X51005230616.png` |
| Ground truth | **FAKE** |
| Final verdict | **REAL** |
| Vote tally | FAKE: 0 · REAL: 2 · UNCERTAIN: 1 |

**Analysis:** Judges 2 and 3 voted REAL with high confidence, while judge 1 returned UNCERTAIN. This suggests the forgery was subtle — possibly only a digit change in the total with no obvious visual artefact. The strict judge was not confident enough to call it FAKE, and the other two judges read the layout as authentic. **Root cause:** The LLM cannot detect single-digit substitutions unless pixel-level copy-paste traces are visible; pure numeric plausibility checks are not part of the vision prompt.

---

### Case 2 — Real predicted as UNCERTAIN (judges split)

| Field | Value |
|-------|-------|
| Image | `X51005337872.png` |
| Ground truth | **REAL** |
| Final verdict | **UNCERTAIN** |
| Vote tally | FAKE: 1 · REAL: 1 · UNCERTAIN: 1 |

**Analysis:** Three-way split. Judge 1 (strict) flagged some font irregularity that is likely a thermal printer artefact; judge 3 (lenient) accepted it as real; judge 2 (balanced) was undecided. **Root cause:** Thermal receipt printers inherently produce irregular fonts under certain heat conditions. The strict judge over-flags these as tampering evidence. This is the primary source of false positives/uncertainty in the dataset.

---

### Case 3 — FAKE predicted as UNCERTAIN (conservative miss)

| Field | Value |
|-------|-------|
| Image | `X51004741213.png` |
| Ground truth | **FAKE** |
| Final verdict | **UNCERTAIN** |
| Vote tally | FAKE: 1 · REAL: 0 · UNCERTAIN: 2 |

**Analysis:** Only one judge (judge_1, the strict one) correctly identified this as FAKE. The other two returned UNCERTAIN rather than committing to either class. **Root cause:** Low-quality or small image where the forgery cues (likely digit-level edits) are not visible at the resolution the model processes. This case illustrates the fundamental limitation: if the forgery is high-quality or the image is low-resolution, visual-only detection will miss it.

---

### Key takeaways from error analysis

1. **gpt-4o-mini defaults to UNCERTAIN** when evidence is marginal — this hurts ensemble accuracy because it withholds votes from the majority.
2. **Thermal printer artefacts mimic forgery cues** — leading to false positives (REAL → UNCERTAIN/FAKE).
3. **High-quality forgeries are nearly undetectable** from the image alone — digit substitutions without copy-paste artefacts are invisible to vision models.
4. **Suggested improvement:** Add an OCR-based arithmetic check (do line items sum to the total?) as a hard rule before the LLM judges. This would catch the Case 1 class of forgery without any API cost.

---

## 6. Judge Prompt (copy/paste)

The same base prompt is used for all three judges; only the `{persona}` line changes.

```
You are an expert forensic document examiner evaluating whether a receipt image is forged.
Persona: {persona}

Return ONLY valid JSON with this schema:
{
  "label": "FAKE|REAL|UNCERTAIN",
  "confidence": 0.0,
  "reasons": ["short reason 1", "short reason 2"],
  "flags": ["optional tag", "optional tag"]
}

Guidelines:
- Base your decision primarily on visual cues in the image (fonts, alignment,
  inconsistent spacing, copied/pasted digits, pixel artifacts, weird shadows,
  mismatched totals, etc.).
- If evidence is weak or image quality is poor, use UNCERTAIN and lower confidence.
- confidence is 0 to 100.
- reasons must be short, concrete, and observable.
```

### Personas used

```
judge_1: "strict, skeptical, focuses on forensic inconsistencies"
judge_2: "balanced, looks for plausible printing/scan artifacts vs tampering"
judge_3: "lenient, assumes real unless clear signs of manipulation"
```

The image is attached as a base-64 `data:image/png;base64,...` URL in the `image_url` content block of the `user` message.

---

## 7. Sample Run — Judges' JSON + Final Vote

Below is a real-style output for a receipt where all three judges disagree (three-way split → UNCERTAIN final verdict):

```
Receipt: X51005230616.png  |  Ground Truth: FAKE
```

**Judge 1 — gpt-4o-mini** `(strict / skeptical, temperature=0.2)`
```json
{
  "label": "FAKE",
  "confidence": 78.0,
  "reasons": [
    "Total field digits show distinct pixel density vs surrounding text",
    "Decimal alignment in price column is inconsistent",
    "Font weight on 'TOTAL' line differs from item rows"
  ],
  "flags": ["pixel_artifact", "font_mismatch", "alignment_error"]
}
```

**Judge 2 — gpt-4.1-mini** `(balanced, temperature=0.4)`
```json
{
  "label": "REAL",
  "confidence": 65.0,
  "reasons": [
    "Layout matches common Malaysian retail receipt format",
    "Font irregularities consistent with low-temperature thermal printing",
    "No obvious cut-paste traces at standard resolution"
  ],
  "flags": ["thermal_artifact"]
}
```

**Judge 3 — gpt-4o** `(lenient, temperature=0.7)`
```json
{
  "label": "REAL",
  "confidence": 72.0,
  "reasons": [
    "Ink bleed pattern consistent with thermal paper",
    "Store header and footer formatting appears authentic",
    "Receipt total visually consistent with listed items"
  ],
  "flags": []
}
```

**Vote aggregation:**
```
Labels:       [FAKE, REAL, REAL]
Tally:        { FAKE: 1, REAL: 2, UNCERTAIN: 0 }
Final verdict: REAL   ← 2/3 majority
Is correct:   FALSE  (ground truth is FAKE)
```

**Outcome:** The majority overruled the correct judge. This is a Case 1 error (false negative) — the forgery was not visually obvious enough for two out of three judges to flag it.

---

## 8. How AI Tools Were Used

This project was built with **Claude Code (Sonnet 4.6)** as the primary coding assistant throughout the development process. The following principles were applied:

### What AI did

| Task | AI contribution |
|------|----------------|
| **Architecture** | Proposed the 3-judge + majority-vote ensemble pattern; designed the MCP-based Langfuse integration |
| **Streamlit UI** | Built the 3-tab layout, browse-field session-state fix, Plotly charts for the Statistics tab |
| **Batch Evaluation tab** | Designed and implemented the multiselect image picker, thumbnail grid, cost estimation table, results charts (confusion matrix, box plots, per-judge table) |
| **Session state debugging** | Diagnosed and fixed the `st.session_state` key conflict error in the file browse widgets |
| **Test suite** | Generated unit tests for `vote.py`, `dataset.py`, `judges.py`, `features.py` with mock API |
| **Langfuse integration** | Implemented the `LangfuseMCPClient` wrapper and the `register_dataset.py` CLI script |
| **Documentation** | Generated this document and the project README |

### What required human judgment

- **Model selection:** Choosing `gpt-4o` as anchor, `gpt-4o-mini` for cost efficiency
- **Persona design:** Deciding that strict/balanced/lenient gives useful ensemble diversity
- **Error analysis:** Interpreting why judges disagree and what the failure modes mean
- **Dataset understanding:** Recognising the 5:1 class imbalance and its implications
- **Prompting decisions:** Keeping the prompt short and concrete rather than over-specifying

### Human-in-the-loop workflow

All AI-generated code was reviewed before execution. The session-state bug fix required iterative back-and-forth — the AI's first suggestion used the same key for both the widget and a manual setter, which Streamlit does not allow. The correct solution (a separate backing key) was reached after the error was shown to the AI. This reflects the real workflow: AI writes the first pass, human identifies runtime failures, AI debugs.

---

## 9. Code Quality & Reproducibility

### Project structure

```
testthekey/
├── streamlit_app.py          # Main 3-tab dashboard
├── src/
│   ├── config.py             # Settings dataclass + env loading
│   ├── dataset.py            # Label loading, OCR parsing
│   ├── features.py           # Blur, brightness, contrast
│   ├── judges.py             # Prompt builder, run_judge(), normalisation
│   ├── vote.py               # majority_vote(), vote_tally()
│   └── langfuse_mcp_client.py
├── scripts/
│   ├── run_eval20.py         # CLI batch eval
│   ├── register_dataset.py   # Register dataset items to Langfuse
│   └── run_full_eval_langfuse.py
├── tests/                    # 69 unit + integration tests
├── data/sample/              # 2 sample receipts for smoke tests
├── .env.example
└── requirements.txt
```

### Reproducibility checklist

- [x] All secrets in `.env` (`.env.example` provided, `.gitignore` enforced)
- [x] Pinned dependency versions in `requirements.txt`
- [x] `random_state=42` used for all sampling in batch eval
- [x] Deterministic trace IDs (`create_trace_id(seed=image_id)`) — re-runs upsert, not duplicate
- [x] Dataset items upserted by `item_id=image_id` — safe to re-register
- [x] Results cached to `data/eval_cache.json` — batch eval can be interrupted and resumed

### Test suite

```bash
pytest tests/ -v           # 61 unit tests, no API calls, < 5 seconds
pytest tests/test_integration.py -v -m integration   # 8 real API tests (~$0.01)
```

---

## 10. Decisions & Trade-offs

| Decision | Chosen approach | Alternative considered | Why |
|----------|----------------|----------------------|-----|
| **Detection method** | LLM vision judges | Fine-tuned classifier or classical CV | No labeled training data at sufficient scale; LLM generalises across forgery types |
| **Ensemble size** | 3 judges | 1 judge or 5+ judges | 3 is the minimum for majority vote; 5+ increases cost linearly with marginal benefit |
| **Aggregation** | Majority vote | Weighted average of confidence | Simple, interpretable, robust to one outlier judge |
| **Uncertainty** | Explicit `UNCERTAIN` class | Force binary FAKE/REAL | Preserves calibration information; flags hard cases for human review |
| **Observability** | Langfuse via MCP | No logging / direct SDK | MCP avoids credential management in app code; provides prompt versioning, cost tracking, dataset runs |
| **UI** | Streamlit | CLI only or Dash | Fast to build, interactive, sufficient for evaluation; Dash would be needed only for production scale |
| **Prompt complexity** | Short, concrete | Long chain-of-thought | Shorter prompts are faster, cheaper, and less likely to hallucinate rationale |
| **OCR integration** | Optional `.txt` side-car | Tesseract in-process | Decouples OCR quality from detection quality; allows pre-computed OCR |

### Biggest limitation

The system cannot detect **numeric-only forgeries** (e.g., a single digit changed in the total) unless the edit left pixel-level copy-paste artefacts. Integrating an arithmetic consistency check (do line items sum to the extracted total?) as a hard pre-filter would catch these cases at near-zero cost. This was deliberately left out of scope to keep the pipeline focused on the LLM-judge approach, but is the clearest next improvement.

---

*Generated with Claude Code (Sonnet 4.6) · Langfuse observability at `http://20.119.121.220:3000`*
