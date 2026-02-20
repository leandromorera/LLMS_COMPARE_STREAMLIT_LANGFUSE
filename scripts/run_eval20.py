#!/usr/bin/env python3
"""Batch eval on a pre-sampled CSV, JSON output."""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import load_dataset, find_image_path, find_ocr_path, load_eval_cache, save_eval_cache
from src.judge import run_all_judges
from src.voting import majority_vote, vote_tally
from src.ocr_parser import load_ocr_text


def main():
    parser = argparse.ArgumentParser(description="Batch evaluation on CSV dataset")
    parser.add_argument("csv", help="Path to labels CSV")
    parser.add_argument("image_dir", help="Image directory")
    parser.add_argument("--ocr-dir", help="OCR directory", default=None)
    parser.add_argument("--output", help="Output JSON", default="eval_results.json")
    parser.add_argument("--limit", type=int, default=20, help="Max receipts to evaluate")
    parser.add_argument("--run-missing-only", action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    from openai import OpenAI
    client = OpenAI(api_key=api_key, timeout=float(os.environ.get("OPENAI_TIMEOUT_SECONDS", "60")))

    dataset = load_dataset(args.csv)[:args.limit]
    cache_path = str(Path(args.output).with_suffix('.cache.json'))
    cache = load_eval_cache(cache_path)

    results = []
    for i, row in enumerate(dataset):
        name = row["image"]
        print(f"[{i+1}/{len(dataset)}] {name}...", end=" ", flush=True)

        if args.run_missing_only and name in cache:
            print("(cached)")
            results.append(cache[name])
            continue

        img_path = find_image_path(name, args.image_dir)
        if not img_path:
            print("IMAGE NOT FOUND")
            continue

        ocr_text = None
        if args.ocr_dir:
            ocr_path = find_ocr_path(name, args.ocr_dir)
            if ocr_path:
                ocr_text = load_ocr_text(ocr_path)

        try:
            judge_results = run_all_judges(client, img_path, ocr_text)
            labels = [r["label"] for r in judge_results]
            verdict = majority_vote(labels)
            tally = vote_tally(labels)

            entry = {
                "image": name,
                "ground_truth": row.get("label"),
                "verdict": verdict,
                "tally": tally,
                "judges": judge_results,
            }
            cache[name] = entry
            results.append(entry)
            save_eval_cache(cache_path, cache)
            print(f"{verdict}")
        except Exception as e:
            print(f"ERROR: {e}")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {args.output}")

    # Summary
    correct = sum(1 for r in results if r.get("ground_truth") and r["verdict"] == r["ground_truth"])
    total = sum(1 for r in results if r.get("ground_truth"))
    if total:
        print(f"Accuracy: {correct}/{total} = {correct/total:.1%}")


if __name__ == "__main__":
    main()
