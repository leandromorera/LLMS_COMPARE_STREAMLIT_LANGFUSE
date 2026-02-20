"""Tests for src/dataset.py — label loading, path resolution, OCR extraction."""
import textwrap
from pathlib import Path

import pandas as pd
import pytest

from src.dataset import (
    build_records,
    extract_total_from_text,
    image_basic_stats,
    load_label_table,
    resolve_image_path,
    resolve_ocr_path,
)

# ---------------------------------------------------------------------------
# Paths to the included sample data
# ---------------------------------------------------------------------------
SAMPLE_DIR = Path(__file__).parent.parent / "data" / "sample"
SAMPLE_LABELS = SAMPLE_DIR / "labels.csv"
SAMPLE_IMAGES = SAMPLE_DIR / "images"
SAMPLE_OCR = SAMPLE_DIR / "ocr"


# ---------------------------------------------------------------------------
# load_label_table
# ---------------------------------------------------------------------------
class TestLoadLabelTable:
    def test_reads_labels_csv(self):
        df = load_label_table(SAMPLE_LABELS)
        assert "image" in df.columns
        assert "label" in df.columns

    def test_label_values_are_canonical(self):
        df = load_label_table(SAMPLE_LABELS)
        assert set(df["label"].unique()).issubset({"FAKE", "REAL"})

    def test_forged_1_maps_to_fake(self, tmp_path):
        csv = tmp_path / "labels.csv"
        csv.write_text("image,forged\nfoo.png,1\nbar.png,0\n")
        df = load_label_table(csv)
        assert df.loc[df["image"] == "foo.png", "label"].values[0] == "FAKE"
        assert df.loc[df["image"] == "bar.png", "label"].values[0] == "REAL"

    def test_missing_column_raises(self, tmp_path):
        csv = tmp_path / "bad.csv"
        csv.write_text("image,label\nfoo.png,FAKE\n")
        with pytest.raises(ValueError, match="forged"):
            load_label_table(csv)

    def test_train_txt_format(self):
        """train.txt has image,forged columns (no pre-built label column)."""
        train_txt = SAMPLE_DIR / "train.txt"
        if not train_txt.exists():
            pytest.skip("train.txt not present")
        df = load_label_table(train_txt)
        assert "label" in df.columns
        assert len(df) > 0


# ---------------------------------------------------------------------------
# resolve_image_path / resolve_ocr_path
# ---------------------------------------------------------------------------
class TestResolvePaths:
    def test_resolve_image_found(self):
        images = list(SAMPLE_IMAGES.glob("*.png"))
        if not images:
            pytest.skip("No sample images")
        image_id = images[0].name
        result = resolve_image_path(image_id, [SAMPLE_IMAGES])
        assert result.exists()

    def test_resolve_image_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            resolve_image_path("nonexistent.png", [tmp_path])

    def test_resolve_ocr_found(self):
        txts = list(SAMPLE_OCR.glob("*.txt"))
        if not txts:
            pytest.skip("No sample OCR files")
        image_id = txts[0].stem + ".png"
        result = resolve_ocr_path(image_id, [SAMPLE_OCR])
        assert result is not None and result.exists()

    def test_resolve_ocr_missing_returns_none(self, tmp_path):
        result = resolve_ocr_path("nonexistent.png", [tmp_path])
        assert result is None


# ---------------------------------------------------------------------------
# build_records
# ---------------------------------------------------------------------------
class TestBuildRecords:
    def test_builds_from_sample(self):
        df = load_label_table(SAMPLE_LABELS)
        records = build_records(df, image_roots=[SAMPLE_IMAGES], ocr_roots=[SAMPLE_OCR])
        assert len(records) == len(df)
        for r in records:
            assert r.image_path.exists()
            assert r.label in ("FAKE", "REAL")

    def test_ocr_path_resolved(self):
        df = load_label_table(SAMPLE_LABELS)
        records = build_records(df, image_roots=[SAMPLE_IMAGES], ocr_roots=[SAMPLE_OCR])
        assert any(r.ocr_path is not None for r in records)


# ---------------------------------------------------------------------------
# extract_total_from_text
# ---------------------------------------------------------------------------
class TestExtractTotalFromText:
    def test_keyword_line_preferred(self):
        text = "subtotal RM10.00\nTotal RM88.90\nchange RM11.10"
        result = extract_total_from_text(text)
        assert result == pytest.approx(88.90)

    def test_dollar_symbol(self):
        text = "Total $45.99"
        assert extract_total_from_text(text) == pytest.approx(45.99)

    def test_grand_total_keyword(self):
        text = "items 5.00\nGrand Total 120.50"
        assert extract_total_from_text(text) == pytest.approx(120.50)

    def test_no_numbers_returns_none(self):
        assert extract_total_from_text("no numbers here") is None

    def test_empty_string_returns_none(self):
        assert extract_total_from_text("") is None

    def test_fallback_to_max_number(self):
        # No keyword lines — should return max numeric value
        text = "qty 2\nprice 9.99\nsubtotal 19.98"
        result = extract_total_from_text(text)
        assert result == pytest.approx(19.98)

    def test_real_ocr_fake_receipt(self):
        """YONGFATT receipt — total is RM 88.91 (label on one line, value on next)."""
        ocr_path = SAMPLE_OCR / "X00016469622.txt"
        if not ocr_path.exists():
            pytest.skip("Sample OCR not present")
        text = ocr_path.read_text(errors="ignore")
        result = extract_total_from_text(text)
        assert result is not None
        assert result == pytest.approx(88.91)

    def test_real_ocr_real_receipt(self):
        """MR D.I.Y. receipt — total is RM 30.91 (label on one line, value on next)."""
        ocr_path = SAMPLE_OCR / "X00016469623.txt"
        if not ocr_path.exists():
            pytest.skip("Sample OCR not present")
        text = ocr_path.read_text(errors="ignore")
        result = extract_total_from_text(text)
        assert result is not None
        assert result == pytest.approx(30.91)


# ---------------------------------------------------------------------------
# image_basic_stats
# ---------------------------------------------------------------------------
class TestImageBasicStats:
    def test_returns_expected_keys(self):
        images = list(SAMPLE_IMAGES.glob("*.png"))
        if not images:
            pytest.skip("No sample images")
        stats = image_basic_stats(images[0])
        assert set(stats.keys()) == {"width", "height", "aspect_ratio", "file_kb"}

    def test_positive_dimensions(self):
        images = list(SAMPLE_IMAGES.glob("*.png"))
        if not images:
            pytest.skip("No sample images")
        stats = image_basic_stats(images[0])
        assert stats["width"] > 0
        assert stats["height"] > 0
        assert stats["file_kb"] > 0

    def test_aspect_ratio_consistent(self):
        images = list(SAMPLE_IMAGES.glob("*.png"))
        if not images:
            pytest.skip("No sample images")
        stats = image_basic_stats(images[0])
        expected = stats["width"] / stats["height"]
        assert stats["aspect_ratio"] == pytest.approx(expected)
