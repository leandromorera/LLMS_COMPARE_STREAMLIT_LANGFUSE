#!/usr/bin/env python3
"""Sample a balanced (stratified) subset from a large dataset."""
import argparse
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.dataset import load_dataset


def main():
    parser = argparse.ArgumentParser(description="Sample balanced subset from dataset")
    parser.add_argument("csv", help="Input labels CSV")
    parser.add_argument("--n", type=int, default=20, help="Total samples (split evenly)")
    parser.add_argument("--output", default="sampled_labels.csv", help="Output CSV")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    rows = load_dataset(args.csv)

    by_label = {}
    for row in rows:
        label = row["label"]
        by_label.setdefault(label, []).append(row)

    per_label = args.n // len(by_label)
    sampled = []
    for label, label_rows in by_label.items():
        sampled.extend(random.sample(label_rows, min(per_label, len(label_rows))))

    random.shuffle(sampled)

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "label"])
        writer.writeheader()
        writer.writerows(sampled)

    print(f"Sampled {len(sampled)} rows → {args.output}")
    for label, items in by_label.items():
        count = sum(1 for r in sampled if r["label"] == label)
        print(f"  {label}: {count}")


if __name__ == "__main__":
    main()
