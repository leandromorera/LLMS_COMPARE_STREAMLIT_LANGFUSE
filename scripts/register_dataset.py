"""
Register any label CSV/TXT into a Langfuse dataset (no LLM calls).

For each image the script stores:
  - ground-truth label (REAL / FAKE)
  - image dimensions, file size, aspect ratio
  - brightness, contrast, sharpness (blur variance)
  - OCR-extracted total amount (if --ocr-dir is provided)

Dataset items are upserted by image filename — re-running is safe (no duplicates).
When judges are later run from the Streamlit app or eval scripts, their traces
will be linked to these items automatically.

Usage
-----
# Register all images (metadata only, free):
python -m scripts.register_dataset \
    --labels    "E:/DESCARGAS/findit2 (1)/findit2/train.txt" \
    --image-dir "E:/DESCARGAS/findit2 (1)/findit2/train" \
    --dataset-name findit2-train

# With OCR totals:
python -m scripts.register_dataset \
    --labels    "E:/DESCARGAS/findit2 (1)/findit2/train.txt" \
    --image-dir "E:/DESCARGAS/findit2 (1)/findit2/train" \
    --ocr-dir   "E:/DESCARGAS/findit2 (1)/findit2/ocr" \
    --dataset-name findit2-train

# Test with first 10 images only:
python -m scripts.register_dataset \
    --labels    "E:/DESCARGAS/findit2 (1)/findit2/train.txt" \
    --image-dir "E:/DESCARGAS/findit2 (1)/findit2/train" \
    --dataset-name findit2-train \
    --limit 10
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from src.dataset import extract_total_from_text, image_basic_stats
from src.features import blur_variance_of_laplacian, brightness_contrast
from src.langfuse_mcp_client import LangfuseMCPClient


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Register a receipt dataset into Langfuse (metadata only, no LLM calls)."
    )
    p.add_argument("--labels",       required=True,
                   help="CSV/TXT with 'image' + 'forged'/'label' columns")
    p.add_argument("--image-dir",    required=True,
                   help="Directory containing the receipt images")
    p.add_argument("--ocr-dir",      default="",
                   help="(optional) Directory with .txt OCR files")
    p.add_argument("--dataset-name", required=True,
                   help="Langfuse dataset name — created if it does not exist")
    p.add_argument("--dataset-desc", default="",
                   help="Description for the Langfuse dataset")
    p.add_argument("--limit",        type=int, default=0,
                   help="Process at most N images (0 = all). Useful for testing.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_df(labels_path: Path) -> pd.DataFrame:
    df = pd.read_csv(labels_path)
    if "label" not in df.columns:
        if "forged" in df.columns:
            df["label"] = df["forged"].apply(lambda x: "FAKE" if int(x) == 1 else "REAL")
        else:
            raise ValueError("Label file must contain a 'label' or 'forged' column.")
    return df


def _collect_metadata(image_path: Path, ocr_path: Path | None) -> dict:
    """Return all lightweight metadata for one image (same fields the app shows)."""
    stats = image_basic_stats(image_path)
    bc    = brightness_contrast(image_path)
    blur  = blur_variance_of_laplacian(image_path)

    ocr_text  = ocr_path.read_text(errors="ignore") if (ocr_path and ocr_path.exists()) else ""
    ocr_total = extract_total_from_text(ocr_text) if ocr_text else None

    return {
        # dimensions & file
        "width":               stats["width"],
        "height":              stats["height"],
        "aspect_ratio":        round(stats["aspect_ratio"] or 0, 4),
        "file_kb":             round(stats["file_kb"], 2),
        # image quality
        "brightness_mean":     round(bc["brightness_mean"], 4),
        "contrast_std":        round(bc["contrast_std"], 4),
        "blur_var_laplacian":  round(blur, 2),
        # OCR
        "ocr_total_extracted": ocr_total,
        "ocr_text_preview":    ocr_text[:400] if ocr_text else None,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    load_dotenv()
    args = _parse_args()

    labels_path = Path(args.labels)
    image_dir   = Path(args.image_dir)
    ocr_dir     = Path(args.ocr_dir) if args.ocr_dir else None

    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    df = _load_df(labels_path)
    if args.limit:
        df = df.head(args.limit)

    lf = LangfuseMCPClient()

    # Verify connectivity
    auth = lf.auth_check(verbose=True)
    print(f"Langfuse connected: {auth.get('base_url', '?')}\n")

    # Create / ensure dataset exists
    ds_desc = args.dataset_desc or f"Receipts — {labels_path.name}  ({len(df)} images)"
    lf.dataset_create(
        name=args.dataset_name,
        description=ds_desc,
        metadata={"source": str(labels_path), "n_rows": len(df)},
    )
    print(f"Dataset '{args.dataset_name}'  →  {len(df)} images to register\n")

    # ── Register each image ─────────────────────────────────────────────────
    n_ok     = 0
    n_skip   = 0
    errors   = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Registering"):
        image_id = str(row["image"])
        gt       = str(row["label"]).upper().strip()
        img_path = image_dir / image_id
        ocr_path = (ocr_dir / Path(image_id).with_suffix(".txt").name) if ocr_dir else None

        if not img_path.exists():
            errors.append({"image_id": image_id, "error": "image file not found"})
            n_skip += 1
            continue

        try:
            meta = _collect_metadata(img_path, ocr_path)

            lf.dataset_add_item(
                dataset_name=args.dataset_name,
                input={
                    "image_id":   image_id,
                    "image_path": str(img_path),
                    "metadata":   meta,
                },
                expected_output={"label": gt},
                item_id=image_id,          # upsert — safe to re-run
                metadata={"ground_truth": gt, **meta},
            )
            n_ok += 1

        except Exception as exc:
            errors.append({"image_id": image_id, "error": str(exc)})
            tqdm.write(f"  ERROR {image_id}: {exc}")

    # ── Summary ─────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Registered : {n_ok}/{len(df)}")
    if n_skip:
        print(f"Skipped    : {n_skip}  (image file not found)")
    if errors:
        print(f"Errors     : {len(errors)}")
        print(json.dumps(errors[:10], indent=2))
    print(f"Dataset    : '{args.dataset_name}'  visible in Langfuse")
    print(
        "\nNext step: run judges from the Streamlit app or eval scripts — "
        "their traces will link to these dataset items automatically."
    )


if __name__ == "__main__":
    main()
