"""Langfuse observability logging for receipt analysis traces."""
import os
from typing import Optional


def _get_client():
    """Get Langfuse client if credentials available."""
    try:
        from langfuse import Langfuse
        public_key = os.environ.get('LANGFUSE_PUBLIC_KEY', '')
        secret_key = os.environ.get('LANGFUSE_SECRET_KEY', '')
        host = os.environ.get('LANGFUSE_HOST', 'https://cloud.langfuse.com')
        if public_key and secret_key:
            return Langfuse(public_key=public_key, secret_key=secret_key, host=host)
    except ImportError:
        pass
    return None


def log_receipt_trace(
    image_path: str,
    ground_truth: Optional[str],
    image_stats: dict,
    judge_results: list[dict],
    final_verdict: str,
    vote_tally_result: dict,
    run_name: Optional[str] = None,
    dataset_item_id: Optional[str] = None,
) -> Optional[str]:
    """Log full receipt analysis trace to Langfuse. Returns trace_id or None."""
    client = _get_client()
    if client is None:
        return None

    from pathlib import Path
    import time

    trace_name = f"receipt-analysis-{Path(image_path).stem}"
    trace = client.trace(
        name=trace_name,
        input={"image_path": image_path, "image_stats": image_stats},
        output={"verdict": final_verdict, "tally": vote_tally_result},
        metadata={"ground_truth": ground_truth, "run_name": run_name},
    )

    # dataset_analysis span
    trace.span(
        name="dataset_analysis",
        input={"image_path": image_path},
        output=image_stats,
    )

    # Judge generations
    for judge in judge_results:
        judge_name = judge.get("judge_name", "judge")
        trace.generation(
            name=judge_name,
            model=judge.get("model", "unknown"),
            input={"image_path": image_path},
            output={"label": judge["label"], "confidence": judge["confidence"],
                    "reasons": judge.get("reasons", []), "flags": judge.get("flags", [])},
            usage={
                "input": judge.get("input_tokens", 0),
                "output": judge.get("output_tokens", 0),
                "total": judge.get("total_tokens", 0),
            },
            metadata={"latency_ms": judge.get("latency_ms", 0)},
        )

    # vote_aggregation span
    trace.span(
        name="vote_aggregation",
        input={"labels": [j["label"] for j in judge_results]},
        output={"verdict": final_verdict, "tally": vote_tally_result},
    )

    # Scores
    labels = [j["label"] for j in judge_results]
    for i, judge in enumerate(judge_results, 1):
        trace.score(name=f"judge_{i}_confidence", value=judge.get("confidence", 0))

        if ground_truth:
            jlabel = judge.get("label", "UNCERTAIN")
            if jlabel == ground_truth:
                correctness = 1.0
            elif jlabel == "UNCERTAIN":
                correctness = 0.5
            else:
                correctness = 0.0
            trace.score(name=f"judge_{i}_correctness", value=correctness)

    if ground_truth:
        final_correct = 1.0 if final_verdict == ground_truth else 0.0
        trace.score(name="final_correct", value=final_correct)

    agreement = 1.0 if len(set(labels)) == 1 else 0.0
    trace.score(name="inter_judge_agreement", value=agreement)

    if run_name and dataset_item_id:
        client.dataset_run_log(
            dataset_item_id=dataset_item_id,
            run_name=run_name,
            observation_id=trace.id,
        )

    client.flush()
    return trace.id
