from __future__ import annotations

import os

import re

from dataclasses import dataclass

from pathlib import Path

from typing import Iterable



import pandas as pd

from PIL import Image



@dataclass(frozen=True)

class ReceiptRecord:

    image_id: str

    image_path: Path

    label: str  # "REAL" or "FAKE"

    ocr_path: Path | None



def load_label_table(label_csv_or_txt: Path) -> pd.DataFrame:

    df = pd.read_csv(label_csv_or_txt)

    if "image" not in df.columns or "forged" not in df.columns:

        raise ValueError("Label table must include columns: image, forged (0/1)")

    df = df.copy()

    df["label"] = df["forged"].apply(lambda x: "FAKE" if int(x) == 1 else "REAL")

    return df



def resolve_image_path(image_id: str, image_roots: Iterable[Path]) -> Path:

    for root in image_roots:

        candidate = root / image_id

        if candidate.exists():

            return candidate

    raise FileNotFoundError(f"Could not find image {image_id} in any of: {list(image_roots)}")



def resolve_ocr_path(image_id: str, ocr_roots: Iterable[Path]) -> Path | None:

    txt_name = Path(image_id).with_suffix(".txt").name

    for root in ocr_roots:

        candidate = root / txt_name

        if candidate.exists():

            return candidate

    return None



def build_records(

    label_table: pd.DataFrame,

    image_roots: Iterable[Path],

    ocr_roots: Iterable[Path] | None = None,

) -> list[ReceiptRecord]:

    ocr_roots = list(ocr_roots or [])

    image_roots = list(image_roots)

    records: list[ReceiptRecord] = []

    for _, row in label_table.iterrows():

        image_id = str(row["image"])

        label = str(row["label"])

        image_path = resolve_image_path(image_id, image_roots)

        ocr_path = resolve_ocr_path(image_id, ocr_roots) if ocr_roots else None

        records.append(ReceiptRecord(image_id=image_id, image_path=image_path, label=label, ocr_path=ocr_path))

    return records



_money_re = re.compile(r"(\d+[\d,]*)(?:\.(\d{1,2}))?")



def extract_total_from_text(text: str) -> float | None:

    # Prefer totals with keywords first.

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    keywords = ("total", "amount due", "total rounded", "total sales", "grand total", "balance")

    # Collect (line, next_line) pairs so we can scan values that follow a keyword
    # on the very next line — a common receipt layout (label line, value line).
    keyword_pairs: list[str] = []
    for i, ln in enumerate(lines):
        low = ln.lower()
        if any(k in low for k in keywords):
            keyword_pairs.append(ln)
            if i + 1 < len(lines):
                keyword_pairs.append(lines[i + 1])

    def extract_candidates(s: str) -> list[float]:

        vals = []

        for m in _money_re.finditer(s.replace("RM", " ").replace("$", " ").replace("€", " ")):

            whole = m.group(1).replace(",", "")

            frac = m.group(2)

            try:

                v = float(f"{whole}.{frac}" if frac else whole)

                if 0.0 < v < 1e7:

                    vals.append(v)

            except Exception:

                continue

        return vals



    candidates: list[float] = []

    for ln in keyword_pairs[:40]:

        candidates.extend(extract_candidates(ln))



    if candidates:

        return max(candidates)



    # Fallback: global max numeric value.

    candidates = extract_candidates(text)

    if candidates:

        return max(candidates)

    return None



def image_basic_stats(image_path: Path) -> dict:

    with Image.open(image_path) as im:

        w, h = im.size

    size_bytes = image_path.stat().st_size

    return {

        "width": int(w),

        "height": int(h),

        "aspect_ratio": float(w) / float(h) if h else None,

        "file_kb": float(size_bytes) / 1024.0,

    }
