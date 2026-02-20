"""Dataset loading and management."""
import csv
import json
import os
from pathlib import Path
from typing import Optional


def load_labels_csv(csv_path: str) -> list[dict]:
    """Load label CSV with 'image' and 'forged'/'label' columns."""
    rows = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Normalize column names
            image = row.get('image') or row.get('filename') or row.get('file', '')

            # Determine label
            if 'label' in row:
                label = row['label'].upper()
            elif 'forged' in row:
                forged = row['forged'].strip()
                label = 'FAKE' if forged in ('1', 'true', 'True', 'TRUE', 'fake', 'FAKE') else 'REAL'
            else:
                label = 'UNKNOWN'

            rows.append({
                'image': image.strip(),
                'label': label,
                **{k: v for k, v in row.items() if k not in ('image', 'filename', 'file', 'label', 'forged')},
            })
    return rows


def load_labels_txt(txt_path: str) -> list[dict]:
    """Load train.txt style label file: 'image_path label' per line."""
    rows = []
    with open(txt_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                image = parts[0]
                forged = parts[1]
                label = 'FAKE' if forged in ('1', 'fake', 'FAKE') else 'REAL'
                rows.append({'image': image, 'label': label})
    return rows


def load_dataset(labels_path: str) -> list[dict]:
    """Auto-detect CSV or TXT format and load dataset."""
    path = Path(labels_path)
    if path.suffix.lower() == '.csv':
        return load_labels_csv(labels_path)
    elif path.suffix.lower() in ('.txt', '.text'):
        return load_labels_txt(labels_path)
    else:
        # Try CSV first
        try:
            return load_labels_csv(labels_path)
        except Exception:
            return load_labels_txt(labels_path)


def find_image_path(image_name: str, image_dir: str) -> Optional[str]:
    """Find image file in directory, trying common extensions."""
    image_dir = Path(image_dir)

    # Try as-is
    candidate = image_dir / image_name
    if candidate.exists():
        return str(candidate)

    # Try with extensions
    stem = Path(image_name).stem
    for ext in ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'):
        candidate = image_dir / f"{stem}{ext}"
        if candidate.exists():
            return str(candidate)

    return None


def find_ocr_path(image_name: str, ocr_dir: str) -> Optional[str]:
    """Find OCR text file for an image."""
    if not ocr_dir:
        return None
    ocr_dir = Path(ocr_dir)
    stem = Path(image_name).stem
    for ext in ('.txt', '.text'):
        candidate = ocr_dir / f"{stem}{ext}"
        if candidate.exists():
            return str(candidate)
    return None


def load_eval_cache(cache_path: str) -> dict:
    """Load evaluation cache from JSON file."""
    path = Path(cache_path)
    if not path.exists():
        return {}
    with open(cache_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_eval_cache(cache_path: str, cache: dict) -> None:
    """Save evaluation cache to JSON file."""
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(cache, f, indent=2, default=str)
