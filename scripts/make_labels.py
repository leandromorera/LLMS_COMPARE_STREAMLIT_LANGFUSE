"""
Generate a clean labels.csv from a raw annotations file (e.g. train.txt),
cross-referencing which image and OCR files actually exist on disk.

Reads any CSV/TXT file that has at minimum:
  image   — filename of the receipt image
  forged  — 1 = FAKE, 0 = REAL

Extra columns (digital annotation, handwritten annotation, forgery annotations,
etc.) are silently dropped in the output.

Usage:
    python scripts/make_labels.py \
        --annotations data/sample/train.txt \
        --image_dir   data/sample/images \
        --ocr_dir     data/sample/ocr \
        --out         data/sample/labels.csv

Flags:
    --all      Include rows even when the image file is not found on disk
               (default: only rows whose image exists are written)
    --dry-run  Print the result table without writing the file
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def build_labels(
    annotations: Path,
    image_dir: Path,
    ocr_dir: Path | None = None,
    images_only: bool = True,
) -> pd.DataFrame:
    """
    Read annotations file and return a clean labels DataFrame.

    Columns in output: image, forged, label, img_exists, ocr_exists
    """
    df = pd.read_csv(annotations)

    missing = [c for c in ("image", "forged") if c not in df.columns]
    if missing:
        raise ValueError(
            f"Annotations file is missing required columns: {missing}\n"
            f"Found columns: {df.columns.tolist()}"
        )

    df = df[["image", "forged"]].copy()
    df["forged"] = df["forged"].apply(lambda x: int(x))
    df["label"]  = df["forged"].apply(lambda x: "FAKE" if x == 1 else "REAL")

    # Cross-reference with files on disk
    df["img_exists"] = df["image"].apply(lambda x: (image_dir / x).exists())

    if ocr_dir:
        df["ocr_exists"] = df["image"].apply(
            lambda x: (ocr_dir / Path(x).with_suffix(".txt").name).exists()
        )
    else:
        df["ocr_exists"] = False

    if images_only:
        df = df[df["img_exists"]].copy()

    return df.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate labels.csv from annotations file")
    parser.add_argument("--annotations", required=True, help="Path to train.txt or similar")
    parser.add_argument("--image_dir",   required=True, help="Directory containing receipt images")
    parser.add_argument("--ocr_dir",     default="",    help="Directory with OCR .txt files (optional)")
    parser.add_argument("--out",         required=True, help="Output labels.csv path")
    parser.add_argument("--all",         action="store_true",
                        help="Include rows even when image file is missing")
    parser.add_argument("--dry-run",     action="store_true",
                        help="Print result without writing file")
    args = parser.parse_args()

    ann_path  = Path(args.annotations)
    img_dir   = Path(args.image_dir)
    ocr_dir   = Path(args.ocr_dir) if args.ocr_dir.strip() else None
    out_path  = Path(args.out)

    if not ann_path.exists():
        print(f"ERROR: annotations file not found: {ann_path}", file=sys.stderr)
        sys.exit(1)
    if not img_dir.exists():
        print(f"ERROR: image directory not found: {img_dir}", file=sys.stderr)
        sys.exit(1)

    df = build_labels(ann_path, img_dir, ocr_dir, images_only=not args.all)

    total_ann  = len(pd.read_csv(ann_path))
    n_written  = len(df)
    n_missing  = total_ann - n_written if not args.all else (df["img_exists"] == False).sum()
    n_fake     = int((df["label"] == "FAKE").sum())
    n_real     = int((df["label"] == "REAL").sum())
    n_ocr      = int(df["ocr_exists"].sum()) if ocr_dir else 0

    print(f"\nAnnotations file : {ann_path}  ({total_ann} rows)")
    print(f"Images directory : {img_dir}")
    print(f"OCR directory    : {ocr_dir or '(not provided)'}")
    print()
    print(f"  Total in annotations : {total_ann}")
    print(f"  Images found on disk : {n_written}  ({n_missing} missing)")
    print(f"  OCR files found      : {n_ocr}")
    print()
    print(f"  FAKE  : {n_fake}")
    print(f"  REAL  : {n_real}")
    print()

    out_df = df[["image", "forged", "label"]]

    if args.dry_run:
        print("--- DRY RUN — first 20 rows ---")
        print(out_df.head(20).to_string(index=False))
        print(f"\n(would write {len(out_df)} rows to {out_path})")
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_path, index=False)
        print(f"Written: {out_path}  ({len(out_df)} rows)")


if __name__ == "__main__":
    main()
