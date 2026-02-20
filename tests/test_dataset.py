"""Tests for dataset loading."""
import pytest
import csv
import json
import os
import tempfile
from pathlib import Path
from src.dataset import (
    load_labels_csv,
    load_labels_txt,
    load_dataset,
    find_image_path,
    find_ocr_path,
    load_eval_cache,
    save_eval_cache,
)


class TestLoadLabelsCsv:
    def test_basic_csv_with_forged_column(self, tmp_path):
        csv_file = tmp_path / "labels.csv"
        csv_file.write_text("image,forged\nreceipt1.jpg,1\nreceipt2.jpg,0\n")
        rows = load_labels_csv(str(csv_file))
        assert len(rows) == 2
        assert rows[0]["label"] == "FAKE"
        assert rows[1]["label"] == "REAL"

    def test_csv_with_label_column(self, tmp_path):
        csv_file = tmp_path / "labels.csv"
        csv_file.write_text("image,label\nreceipt1.jpg,FAKE\nreceipt2.jpg,REAL\n")
        rows = load_labels_csv(str(csv_file))
        assert rows[0]["label"] == "FAKE"
        assert rows[1]["label"] == "REAL"

    def test_forged_zero_is_real(self, tmp_path):
        csv_file = tmp_path / "labels.csv"
        csv_file.write_text("image,forged\nreceipt.jpg,0\n")
        rows = load_labels_csv(str(csv_file))
        assert rows[0]["label"] == "REAL"

    def test_forged_one_is_fake(self, tmp_path):
        csv_file = tmp_path / "labels.csv"
        csv_file.write_text("image,forged\nreceipt.jpg,1\n")
        rows = load_labels_csv(str(csv_file))
        assert rows[0]["label"] == "FAKE"

    def test_image_field_stripped(self, tmp_path):
        csv_file = tmp_path / "labels.csv"
        csv_file.write_text("image,forged\n  receipt.jpg  ,1\n")
        rows = load_labels_csv(str(csv_file))
        assert rows[0]["image"] == "receipt.jpg"

    def test_empty_csv_returns_empty_list(self, tmp_path):
        csv_file = tmp_path / "labels.csv"
        csv_file.write_text("image,forged\n")
        rows = load_labels_csv(str(csv_file))
        assert rows == []


class TestLoadLabelsTxt:
    def test_basic_txt(self, tmp_path):
        txt_file = tmp_path / "train.txt"
        txt_file.write_text("receipt1.jpg 1\nreceipt2.jpg 0\n")
        rows = load_labels_txt(str(txt_file))
        assert len(rows) == 2
        assert rows[0]["label"] == "FAKE"
        assert rows[1]["label"] == "REAL"

    def test_comments_ignored(self, tmp_path):
        txt_file = tmp_path / "train.txt"
        txt_file.write_text("# comment\nreceipt.jpg 1\n")
        rows = load_labels_txt(str(txt_file))
        assert len(rows) == 1

    def test_blank_lines_ignored(self, tmp_path):
        txt_file = tmp_path / "train.txt"
        txt_file.write_text("\nreceipt.jpg 0\n\n")
        rows = load_labels_txt(str(txt_file))
        assert len(rows) == 1


class TestFindImagePath:
    def test_finds_existing_file(self, tmp_path):
        img = tmp_path / "receipt.jpg"
        img.write_bytes(b"fake")
        result = find_image_path("receipt.jpg", str(tmp_path))
        assert result == str(img)

    def test_returns_none_if_not_found(self, tmp_path):
        result = find_image_path("missing.jpg", str(tmp_path))
        assert result is None

    def test_finds_by_stem_with_extension(self, tmp_path):
        img = tmp_path / "receipt.jpg"
        img.write_bytes(b"fake")
        result = find_image_path("receipt", str(tmp_path))
        assert result is not None


class TestEvalCache:
    def test_save_and_load(self, tmp_path):
        cache_path = str(tmp_path / "cache.json")
        data = {"receipt.jpg": {"verdict": "FAKE"}}
        save_eval_cache(cache_path, data)
        loaded = load_eval_cache(cache_path)
        assert loaded["receipt.jpg"]["verdict"] == "FAKE"

    def test_load_nonexistent_returns_empty(self, tmp_path):
        result = load_eval_cache(str(tmp_path / "nonexistent.json"))
        assert result == {}

    def test_save_creates_parent_dirs(self, tmp_path):
        cache_path = str(tmp_path / "subdir" / "cache.json")
        save_eval_cache(cache_path, {})
        assert Path(cache_path).exists()


# Integration tests (require real OpenAI API key)
@pytest.mark.integration
class TestJudgeIntegration:
    def test_real_api_single_judge(self, tmp_path):
        """Run one judge against a real synthetic image."""
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("No OPENAI_API_KEY")

        try:
            from PIL import Image
            from openai import OpenAI
        except ImportError:
            pytest.skip("PIL or openai not available")

        from src.judge import run_judge, JUDGE_CONFIGS

        img_path = str(tmp_path / "test_receipt.png")
        img = Image.new("RGB", (200, 300), color=(255, 255, 255))
        img.save(img_path)

        client = OpenAI(api_key=api_key, timeout=30)
        result = run_judge(client, img_path, JUDGE_CONFIGS[0])

        assert result["label"] in ("FAKE", "REAL", "UNCERTAIN")
        assert 0 <= result["confidence"] <= 100
        assert isinstance(result["reasons"], list)
        assert isinstance(result["flags"], list)
        assert result["latency_ms"] > 0

    def test_real_api_majority_vote(self, tmp_path):
        """Run all 3 judges and check majority vote."""
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("No OPENAI_API_KEY")

        try:
            from PIL import Image
            from openai import OpenAI
        except ImportError:
            pytest.skip("PIL or openai not available")

        from src.judge import run_all_judges
        from src.voting import majority_vote

        img_path = str(tmp_path / "test_receipt.png")
        img = Image.new("RGB", (200, 300), color=(255, 255, 255))
        img.save(img_path)

        client = OpenAI(api_key=api_key, timeout=60)
        results = run_all_judges(client, img_path)

        assert len(results) == 3
        labels = [r["label"] for r in results]
        verdict = majority_vote(labels)
        assert verdict in ("FAKE", "REAL", "UNCERTAIN")

    def test_real_api_with_ocr_text(self, tmp_path):
        """Run judge with OCR text context."""
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("No OPENAI_API_KEY")

        try:
            from PIL import Image
            from openai import OpenAI
        except ImportError:
            pytest.skip("PIL or openai not available")

        from src.judge import run_judge, JUDGE_CONFIGS

        img_path = str(tmp_path / "test_receipt.png")
        img = Image.new("RGB", (200, 300), color=(255, 255, 255))
        img.save(img_path)

        client = OpenAI(api_key=api_key, timeout=30)
        result = run_judge(client, img_path, JUDGE_CONFIGS[0], ocr_text="Total: $25.99")

        assert result["label"] in ("FAKE", "REAL", "UNCERTAIN")

    def test_real_api_json_output_schema(self, tmp_path):
        """Verify output schema is correct for real API call."""
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("No OPENAI_API_KEY")

        try:
            from PIL import Image
            from openai import OpenAI
        except ImportError:
            pytest.skip("PIL or openai not available")

        from src.judge import run_judge, JUDGE_CONFIGS

        img_path = str(tmp_path / "test_receipt.png")
        img = Image.new("RGB", (200, 300), color=(255, 255, 255))
        img.save(img_path)

        client = OpenAI(api_key=api_key, timeout=30)
        result = run_judge(client, img_path, JUDGE_CONFIGS[0])

        required_keys = {"label", "confidence", "reasons", "flags", "latency_ms", "model"}
        assert required_keys.issubset(result.keys())

    def test_real_api_confidence_range(self, tmp_path):
        """Confidence should be between 0 and 100."""
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("No OPENAI_API_KEY")

        try:
            from PIL import Image
            from openai import OpenAI
        except ImportError:
            pytest.skip("PIL or openai not available")

        from src.judge import run_judge, JUDGE_CONFIGS

        img_path = str(tmp_path / "test_receipt.png")
        img = Image.new("RGB", (200, 300), color=(255, 255, 255))
        img.save(img_path)

        client = OpenAI(api_key=api_key, timeout=30)
        result = run_judge(client, img_path, JUDGE_CONFIGS[1])

        assert 0 <= result["confidence"] <= 100

    def test_real_api_token_counts(self, tmp_path):
        """Token counts should be non-negative integers."""
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("No OPENAI_API_KEY")

        try:
            from PIL import Image
            from openai import OpenAI
        except ImportError:
            pytest.skip("PIL or openai not available")

        from src.judge import run_judge, JUDGE_CONFIGS

        img_path = str(tmp_path / "test_receipt.png")
        img = Image.new("RGB", (200, 300), color=(255, 255, 255))
        img.save(img_path)

        client = OpenAI(api_key=api_key, timeout=30)
        result = run_judge(client, img_path, JUDGE_CONFIGS[2])

        assert result["input_tokens"] >= 0
        assert result["output_tokens"] >= 0
        assert result["total_tokens"] >= 0

    def test_real_api_all_judges_different_configs(self, tmp_path):
        """All 3 judges have different temperatures, verify all respond."""
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("No OPENAI_API_KEY")

        try:
            from PIL import Image
            from openai import OpenAI
        except ImportError:
            pytest.skip("PIL or openai not available")

        from src.judge import run_all_judges

        img_path = str(tmp_path / "test_receipt.png")
        img = Image.new("RGB", (200, 300), color=(255, 255, 255))
        img.save(img_path)

        client = OpenAI(api_key=api_key, timeout=60)
        results = run_all_judges(client, img_path)

        assert len(results) == 3
        judge_names = [r["judge_name"] for r in results]
        assert "judge_1" in judge_names
        assert "judge_2" in judge_names
        assert "judge_3" in judge_names

    def test_real_api_latency_measured(self, tmp_path):
        """Latency should be positive milliseconds."""
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("No OPENAI_API_KEY")

        try:
            from PIL import Image
            from openai import OpenAI
        except ImportError:
            pytest.skip("PIL or openai not available")

        from src.judge import run_judge, JUDGE_CONFIGS

        img_path = str(tmp_path / "test_receipt.png")
        img = Image.new("RGB", (200, 300), color=(255, 255, 255))
        img.save(img_path)

        client = OpenAI(api_key=api_key, timeout=30)
        result = run_judge(client, img_path, JUDGE_CONFIGS[0])

        assert result["latency_ms"] > 0
