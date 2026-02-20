"""
Full evaluation with comprehensive Langfuse logging via the local MCP server.

What gets logged to Langfuse:
  - Prompt versions (judge system prompt stored in Prompt Management)
  - Dataset "receipt-detection-sample" with 2 items + expected outputs
  - Per receipt — one trace containing:
      • span  : dataset_analysis  (image stats, blur, brightness, OCR extraction)
      • generation x3 : each judge (model, prompt, response, tokens, latency_ms)
      • span  : vote_aggregation  (labels list → final_label + tally)
      • scores x8 :
            judge_1/2/3_confidence   (0-100)
            judge_1/2/3_correctness  (1.0 correct / 0.5 uncertain / 0.0 wrong)
            final_correct            (0 or 1)
            inter_judge_agreement    (1 if unanimous, 0 if split)
  - Dataset run items linking each trace to its dataset item
  - Summary event: accuracy, avg_confidence, disagreement_rate

Usage (from project root):
    python -m scripts.run_full_eval_langfuse
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from src.config import load_settings
from src.dataset import (
    extract_total_from_text,
    image_basic_stats,
    load_label_table,
)
from src.features import blur_variance_of_laplacian, brightness_contrast
from src.judges import JudgeConfig, _base_prompt, run_judge
from src.langfuse_mcp_client import LangfuseMCPClient
from src.vote import majority_vote, vote_tally

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
SAMPLE_DIR   = Path("data/sample")
LABELS       = SAMPLE_DIR / "labels.csv"
IMAGE_DIR    = SAMPLE_DIR / "images"
OCR_DIR      = SAMPLE_DIR / "ocr"
DATASET_NAME = "receipt-detection-sample"
RUN_NAME     = "eval-3judges-gpt4o"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _sep(title: str = "") -> None:
    print(f"\n{'='*60}" + (f"  {title}" if title else ""))


def _extract_item_id(result: dict) -> str | None:
    if isinstance(result, dict):
        return result.get("id") or result.get("itemId") or result.get("item_id")
    return None


def _judge_correctness(judge_label: str, ground_truth: str) -> float:
    """Score a single judge: 1.0 correct, 0.5 uncertain, 0.0 wrong."""
    if judge_label == ground_truth:
        return 1.0
    if judge_label == "UNCERTAIN":
        return 0.5
    return 0.0


def _extract_analysis(image_path: Path, ocr_path: Path) -> dict:
    """Compute all lightweight image + OCR features for one receipt."""
    img_stats = image_basic_stats(image_path)
    bc        = brightness_contrast(image_path)
    blur      = blur_variance_of_laplacian(image_path)
    ocr_text  = ocr_path.read_text(errors="ignore") if ocr_path.exists() else ""
    ocr_total = extract_total_from_text(ocr_text) if ocr_text else None
    return {
        **img_stats,
        "brightness_mean":     round(bc["brightness_mean"], 4),
        "contrast_std":        round(bc["contrast_std"], 4),
        "blur_variance":       round(blur, 2),
        "ocr_total_extracted": ocr_total,
        "ocr_text_chars":      len(ocr_text),
        "_ocr_text":           ocr_text,          # internal, stripped before logging
    }


# ---------------------------------------------------------------------------
# Per-receipt evaluation (extracted to keep main() complexity low)
# ---------------------------------------------------------------------------

def _eval_receipt(
    image_id: str,
    ground_truth: str,
    oai: OpenAI,
    lf: LangfuseMCPClient,
    judge_cfgs: list[JudgeConfig],
) -> dict:
    """Run the full pipeline for one receipt and log everything to Langfuse."""
    image_path = IMAGE_DIR / image_id
    ocr_path   = OCR_DIR / Path(image_id).with_suffix(".txt").name

    _sep(f"{image_id}  GT={ground_truth}")

    # ── Feature extraction ──────────────────────────────────────────────────
    analysis  = _extract_analysis(image_path, ocr_path)
    ocr_text  = analysis.pop("_ocr_text")
    print(f"  Analysis: {analysis}")

    # ── Register in Langfuse dataset ────────────────────────────────────────
    item_result     = lf.dataset_add_item(
        dataset_name=DATASET_NAME,
        input={
            "image_id":    image_id,
            "image_path":  str(image_path),
            "ocr_preview": ocr_text[:600],
            "analysis":    analysis,
        },
        expected_output={"label": ground_truth},
        item_id=image_id,
        metadata={"ground_truth": ground_truth},
    )
    dataset_item_id = _extract_item_id(item_result) or image_id
    print(f"  Dataset item id: {dataset_item_id}")

    # ── Deterministic trace ID + shared trace metadata ──────────────────────
    trace_id   = lf.create_trace_id(seed=image_id)
    trace_meta = {
        "name":     f"receipt_eval_{image_id}",
        "id":       trace_id,
        "user_id":  "run_full_eval_langfuse",
        "metadata": {
            "receipt_id":   image_id,
            "ground_truth": ground_truth,
            "dataset":      DATASET_NAME,
            "run":          RUN_NAME,
        },
        "tags": ["receipt", "eval", ground_truth.lower()],
    }

    # ── Dataset-analysis span ───────────────────────────────────────────────
    lf.log_observation(
        name="dataset_analysis",
        as_type="span",
        trace=trace_meta,
        observation={
            "input":  {"image_id": image_id},
            "output": analysis,
            "metadata": {
                "ocr_available": bool(ocr_text),
                "image_exists":  image_path.exists(),
            },
        },
    )
    print(f"  dataset_analysis span logged (trace={trace_id})")

    # ── Run 3 judges + log each as a generation ─────────────────────────────
    labels:        list[str]  = []
    judge_outputs: list[dict] = []

    for cfg in judge_cfgs:
        t0         = time.time()
        parsed, meta = run_judge(oai, cfg, image_path=image_path, receipt_id=image_id)
        latency_ms = int((time.time() - t0) * 1000)

        labels.append(parsed["label"])
        judge_outputs.append(parsed)

        lf.log_generation(
            observation={
                "name":  cfg.name,
                "model": cfg.model,
                "input": {
                    "system": meta["input"]["prompt"],
                    "user":   f"Receipt ID: {image_id}  [image attached]",
                },
                "output": meta["output"]["parsed"],
                "usage": {
                    "input":  meta["usage"].get("input_tokens"),
                    "output": meta["usage"].get("output_tokens"),
                    "total":  meta["usage"].get("total_tokens"),
                    "unit":   "TOKENS",
                },
                "metadata": {
                    "temperature":  cfg.temperature,
                    "persona":      cfg.persona,
                    "receipt_id":   image_id,
                    "ground_truth": ground_truth,
                    "raw_response": meta["output"]["raw"][:300],
                },
                "level": "DEFAULT",
            },
            trace=trace_meta,
            latency_ms=latency_ms,
        )
        print(f"    {cfg.name}: label={parsed['label']:9s} conf={parsed['confidence']:5.1f} "
              f"latency={latency_ms}ms tokens={meta['usage'].get('total_tokens')}")

    # ── Vote aggregation span ───────────────────────────────────────────────
    final_label = majority_vote(labels)
    tally       = vote_tally(labels)
    is_correct  = final_label == ground_truth

    lf.log_observation(
        name="vote_aggregation",
        as_type="span",
        trace=trace_meta,
        observation={
            "input":    {"judge_labels": labels},
            "output":   {"final_label": final_label, "tally": tally, "is_correct": is_correct},
            "metadata": {"ground_truth": ground_truth},
        },
    )
    print(f"  Vote: {final_label}  tally={tally}  correct={is_correct}")

    # ── Scores ──────────────────────────────────────────────────────────────
    for cfg, parsed in zip(judge_cfgs, judge_outputs):
        lf.create_score(trace_id, f"{cfg.name}_confidence",
                        parsed["confidence"],
                        comment=f"label={parsed['label']}")
        lf.create_score(trace_id, f"{cfg.name}_correctness",
                        _judge_correctness(parsed["label"], ground_truth),
                        comment=f"judge={parsed['label']} gt={ground_truth}")

    lf.create_score(trace_id, "final_correct",
                    1.0 if is_correct else 0.0,
                    comment=f"final={final_label} gt={ground_truth}")
    lf.create_score(trace_id, "inter_judge_agreement",
                    1.0 if len(set(labels)) == 1 else 0.0,
                    comment=f"labels={labels}")
    print(f"  Scores logged ({2 * len(judge_cfgs) + 2} total)")

    # ── Link trace to dataset run ────────────────────────────────────────────
    lf.dataset_run_log(
        run_name=RUN_NAME,
        dataset_item_id=dataset_item_id,
        trace_id=trace_id,
        metadata={"is_correct": is_correct, "final_label": final_label},
    )
    print(f"  Dataset run linked: run={RUN_NAME}  trace={trace_id}")

    return {
        "image_id":     image_id,
        "ground_truth": ground_truth,
        "final_label":  final_label,
        "tally":        tally,
        "is_correct":   is_correct,
        "trace_id":     trace_id,
        "judges":       judge_outputs,
        "analysis":     analysis,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    load_dotenv()
    settings = load_settings()
    oai      = OpenAI(api_key=settings.openai_api_key)
    lf       = LangfuseMCPClient()

    # 1. Verify MCP connectivity
    _sep("MCP AUTH CHECK")
    auth         = lf.auth_check(verbose=True)
    langfuse_url = auth.get("base_url", "http://20.119.121.220:3000")
    print(json.dumps(auth, indent=2))

    # 2. Store judge prompt in Langfuse Prompt Management
    _sep("PROMPT VERSIONING")
    prompt_result = lf.prompt_create_version(
        name="receipt-forensic-judge",
        prompt=_base_prompt("{persona}"),
        labels=["production"],
        tags=["receipt", "forgery-detection"],
        commit_message="Base forensic judge prompt — 3 personas",
    )
    print("Prompt stored:", json.dumps(prompt_result, indent=2))

    # 3. Create Langfuse dataset
    _sep("DATASET CREATION")
    ds_result = lf.dataset_create(
        name=DATASET_NAME,
        description="2 sample Malaysian receipts (1 FAKE, 1 REAL) for forgery detection eval",
        metadata={"source": "data/sample", "labels": "FAKE,REAL"},
    )
    print("Dataset:", json.dumps(ds_result, indent=2))

    # 4. Load labels & define judge configs
    df = load_label_table(LABELS)
    judge_cfgs = [
        JudgeConfig("judge_1", settings.judge_models[0], 0.2,
                    "strict, skeptical, focuses on forensic inconsistencies"),
        JudgeConfig("judge_2", settings.judge_models[1], 0.4,
                    "balanced, looks for plausible printing/scan artifacts vs tampering"),
        JudgeConfig("judge_3", settings.judge_models[2], 0.7,
                    "lenient, assumes real unless clear signs of manipulation"),
    ]

    # 5. Evaluate every receipt
    eval_results = [
        _eval_receipt(str(row["image"]), str(row["label"]), oai, lf, judge_cfgs)
        for _, row in df.iterrows()
    ]

    # 6. Summary
    _sep("SUMMARY")
    n          = len(eval_results)
    accuracy   = sum(r["is_correct"] for r in eval_results) / n
    avg_conf   = (sum(j["confidence"] for r in eval_results for j in r["judges"])
                  / (n * len(judge_cfgs)))
    n_disagree = sum(1 for r in eval_results if len(set(r["tally"].values())) > 1)

    summary = {
        "n":                 n,
        "accuracy":          round(accuracy, 4),
        "avg_confidence":    round(avg_conf, 2),
        "disagreement_rate": round(n_disagree / n, 4),
        "correct":           sum(r["is_correct"] for r in eval_results),
        "run":               RUN_NAME,
        "dataset":           DATASET_NAME,
    }

    lf.log_observation(
        name="eval_summary",
        as_type="event",
        trace={
            "name":     "receipt_eval_summary",
            "user_id":  "run_full_eval_langfuse",
            "metadata": summary,
            "tags":     ["summary", "eval", RUN_NAME],
        },
        observation={"output": summary},
    )

    print(json.dumps(summary, indent=2))
    print(f"\nDone. Open Langfuse at {langfuse_url} to see results.")


if __name__ == "__main__":
    main()
