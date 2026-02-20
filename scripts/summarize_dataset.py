#!/usr/bin/env python3
"""CLI dataset statistics report."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.dataset import load_dataset, find_image_path
from src.image_features import image_basic_stats, blur_variance_of_laplacian, brightness_contrast
from src.ocr_parser import load_ocr_text, extract_total_from_text


def main():
    parser = argparse.ArgumentParser(description="Dataset statistics report")
    parser.add_argument("csv", help="Labels CSV")
    parser.add_argument("image_dir", help="Image directory")
    parser.add_argument("--ocr-dir", default=None)
    parser.add_argument("--output-dir", default="reports")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    rows = load_dataset(args.csv)
    print(f"Total receipts: {len(rows)}")

    by_label = {}
    for row in rows:
        by_label.setdefault(row["label"], []).append(row)

    for label, items in by_label.items():
        print(f"  {label}: {len(items)}")

    features = []
    for i, row in enumerate(rows):
        img_path = find_image_path(row["image"], args.image_dir)
        if not img_path:
            continue
        try:
            stats = image_basic_stats(img_path)
            bc = brightness_contrast(img_path)
            blur = blur_variance_of_laplacian(img_path)
            features.append({
                "image": row["image"],
                "label": row["label"],
                **stats,
                **bc,
                "sharpness": round(blur, 2),
            })
        except Exception:
            pass

        if (i + 1) % 10 == 0:
            print(f"Processed {i+1}/{len(rows)}...")

    output_path = Path(args.output_dir) / "dataset_summary.json"
    summary = {
        "total": len(rows),
        "by_label": {k: len(v) for k, v in by_label.items()},
        "features": features,
    }
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Report saved to {output_path}")


if __name__ == "__main__":
    main()
