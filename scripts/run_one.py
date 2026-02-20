#!/usr/bin/env python3
"""Single receipt evaluation from command line."""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.judge import run_all_judges
from src.voting import majority_vote, vote_tally
from src.image_features import image_basic_stats, blur_variance_of_laplacian, brightness_contrast
from src.ocr_parser import load_ocr_text, extract_total_from_text


def main():
    parser = argparse.ArgumentParser(description="Evaluate a single receipt image")
    parser.add_argument("image", help="Path to receipt image")
    parser.add_argument("--ocr", help="Path to OCR text file", default=None)
    parser.add_argument("--output", help="Output JSON file", default=None)
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    # Features
    try:
        stats = image_basic_stats(args.image)
        blur = blur_variance_of_laplacian(args.image)
        bc = brightness_contrast(args.image)
        features = {**stats, "sharpness": round(blur, 2), **bc}
        print(f"Features: {json.dumps(features, indent=2)}")
    except Exception as e:
        print(f"Feature extraction failed: {e}", file=sys.stderr)
        features = {}

    # OCR
    ocr_text = None
    if args.ocr:
        ocr_text = load_ocr_text(args.ocr)
        if ocr_text:
            total = extract_total_from_text(ocr_text)
            if total:
                print(f"Extracted total: ${total:.2f}")

    # Judges
    from openai import OpenAI
    client = OpenAI(api_key=api_key, timeout=float(os.environ.get("OPENAI_TIMEOUT_SECONDS", "60")))

    print("Running 3 judges...")
    results = run_all_judges(client, args.image, ocr_text)

    labels = [r["label"] for r in results]
    verdict = majority_vote(labels)
    tally = vote_tally(labels)

    print(f"\n{'='*40}")
    print(f"VERDICT: {verdict}")
    print(f"Tally: {tally}")
    print(f"{'='*40}")

    for r in results:
        print(f"\n{r['judge_name']}: {r['label']} ({r['confidence']}%) [{r['latency_ms']}ms]")
        for reason in r.get("reasons", [])[:3]:
            print(f"  - {reason}")

    output = {
        "image": args.image,
        "verdict": verdict,
        "tally": tally,
        "features": features,
        "judges": results,
    }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")

    return output


if __name__ == "__main__":
    main()
