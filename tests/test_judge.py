"""Tests for judge module with mocked OpenAI client."""
import pytest
import json
from unittest.mock import MagicMock, patch
from src.judge import (
    _normalize_output,
    _extract_json_block,
    _encode_image,
    _image_media_type,
    run_judge,
    run_all_judges,
    JUDGE_CONFIGS,
)


class TestNormalizeOutput:
    def test_valid_fake(self):
        raw = {"label": "FAKE", "confidence": 90, "reasons": ["r1"], "flags": ["f1"]}
        result = _normalize_output(raw)
        assert result["label"] == "FAKE"
        assert result["confidence"] == 90

    def test_valid_real(self):
        raw = {"label": "REAL", "confidence": 80, "reasons": [], "flags": []}
        result = _normalize_output(raw)
        assert result["label"] == "REAL"

    def test_invalid_label_becomes_uncertain(self):
        raw = {"label": "MAYBE", "confidence": 50, "reasons": [], "flags": []}
        result = _normalize_output(raw)
        assert result["label"] == "UNCERTAIN"

    def test_lowercase_label_normalized(self):
        raw = {"label": "fake", "confidence": 70, "reasons": [], "flags": []}
        result = _normalize_output(raw)
        assert result["label"] == "FAKE"

    def test_confidence_clamped_max(self):
        raw = {"label": "REAL", "confidence": 150, "reasons": [], "flags": []}
        result = _normalize_output(raw)
        assert result["confidence"] == 100

    def test_confidence_clamped_min(self):
        raw = {"label": "REAL", "confidence": -10, "reasons": [], "flags": []}
        result = _normalize_output(raw)
        assert result["confidence"] == 0

    def test_string_confidence_converted(self):
        raw = {"label": "REAL", "confidence": "75", "reasons": [], "flags": []}
        result = _normalize_output(raw)
        assert result["confidence"] == 75

    def test_missing_fields_have_defaults(self):
        result = _normalize_output({})
        assert result["label"] == "UNCERTAIN"
        assert result["confidence"] == 50
        assert result["reasons"] == []
        assert result["flags"] == []

    def test_non_list_reasons_converted(self):
        raw = {"label": "FAKE", "confidence": 80, "reasons": "single reason", "flags": []}
        result = _normalize_output(raw)
        assert isinstance(result["reasons"], list)

    def test_non_list_flags_converted(self):
        raw = {"label": "FAKE", "confidence": 80, "reasons": [], "flags": "single flag"}
        result = _normalize_output(raw)
        assert isinstance(result["flags"], list)


class TestExtractJsonBlock:
    def test_simple_json(self):
        text = '{"label": "FAKE", "confidence": 90}'
        result = _extract_json_block(text)
        assert result["label"] == "FAKE"

    def test_json_with_surrounding_text(self):
        text = 'Here is my analysis: {"label": "REAL", "confidence": 80} end'
        result = _extract_json_block(text)
        assert result["label"] == "REAL"

    def test_no_json_raises_error(self):
        with pytest.raises((ValueError, Exception)):
            _extract_json_block("no json here at all")


class TestImageMediaType:
    def test_png_extension(self):
        assert _image_media_type("receipt.png") == "image/png"

    def test_jpg_extension(self):
        assert _image_media_type("receipt.jpg") == "image/jpeg"

    def test_jpeg_extension(self):
        assert _image_media_type("receipt.jpeg") == "image/jpeg"

    def test_uppercase_png(self):
        assert _image_media_type("receipt.PNG") == "image/png"
