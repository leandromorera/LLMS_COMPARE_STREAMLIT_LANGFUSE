#!/usr/bin/env python3
"""Register all dataset items to Langfuse."""
import argparse
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.dataset import load_dataset, find_image_path
from src.image_features import image_basic_stats


def main():
    parser = argparse.ArgumentParser(description="Register dataset items to Langfuse")
    parser.add_argument("csv", help="Labels CSV")
    parser.add_argument("image_dir", help="Image directory")
    parser.add_argument("--dataset-name", default="receipt-forgery-dataset")
    args = parser.parse_args()

    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY", "")
    if not public_key or not secret_key:
        print("ERROR: LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY must be set", file=sys.stderr)
        sys.exit(1)

    try:
        from langfuse import Langfuse
        host = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
        client = Langfuse(public_key=public_key, secret_key=secret_key, host=host)
    except ImportError:
        print("langfuse not installed. Run: pip install langfuse", file=sys.stderr)
        sys.exit(1)

    rows = load_dataset(args.csv)
    print(f"Registering {len(rows)} items to dataset '{args.dataset_name}'...")

    errors = []
    for i, row in enumerate(rows):
        img_path = find_image_path(row["image"], args.image_dir)
        metadata = {"label": row["label"]}
        if img_path:
            try:
                metadata.update(image_basic_stats(img_path))
            except Exception:
                pass

        try:
            client.create_dataset_item(
                dataset_name=args.dataset_name,
                input={"image_name": row["image"], "image_path": img_path or ""},
                expected_output={"label": row["label"]},
                metadata=metadata,
            )
            print(f"[{i+1}/{len(rows)}] {row['image']} → {row['label']}")
        except Exception as e:
            print(f"[{i+1}/{len(rows)}] ERROR: {e}")
            errors.append(str(e))

    client.flush()
    print(f"\nDone. {len(rows) - len(errors)} registered, {len(errors)} errors.")
    if errors:
        print("Errors:", errors[:5])


if __name__ == "__main__":
    main()
