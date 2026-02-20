"""Tests for src/judges.py — pure helpers and mocked OpenAI calls."""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.judges import (
    JudgeConfig,
    _normalize_output,
    _safe_parse_json,
    run_judge,
)

SAMPLE_IMAGE = Path(__file__).parent.parent / "data" / "sample" / "images" / "X00016469622.png"


# ---------------------------------------------------------------------------
# _safe_parse_json
# ---------------------------------------------------------------------------
class TestSafeParseJson:
    def test_valid_json(self):
        result = _safe_parse_json('{"label": "FAKE", "confidence": 90}')
        assert result == {"label": "FAKE", "confidence": 90}

    def test_json_wrapped_in_markdown(self):
        text = '```json\n{"label": "REAL", "confidence": 70}\n```'
        result = _safe_parse_json(text)
        assert result.get("label") == "REAL"

    def test_json_with_surrounding_text(self):
        text = 'Here is my answer: {"label": "UNCERTAIN", "confidence": 50} done.'
        result = _safe_parse_json(text)
        assert result.get("label") == "UNCERTAIN"

    def test_garbage_returns_empty_dict(self):
        assert _safe_parse_json("not json at all") == {}

    def test_empty_string_returns_empty_dict(self):
        assert _safe_parse_json("") == {}


# ---------------------------------------------------------------------------
# _normalize_output
# ---------------------------------------------------------------------------
class TestNormalizeOutput:
    def test_valid_input_passthrough(self):
        d = {"label": "FAKE", "confidence": 85.0, "reasons": ["fonts differ"], "flags": []}
        result = _normalize_output(d)
        assert result["label"] == "FAKE"
        assert result["confidence"] == 85.0
        assert result["reasons"] == ["fonts differ"]

    def test_unknown_label_becomes_uncertain(self):
        d = {"label": "YES", "confidence": 50, "reasons": ["r"]}
        assert _normalize_output(d)["label"] == "UNCERTAIN"

    def test_label_lowercased_input_normalised(self):
        d = {"label": "fake", "confidence": 50, "reasons": ["r"]}
        assert _normalize_output(d)["label"] == "FAKE"

    def test_confidence_clamped_above_100(self):
        d = {"label": "REAL", "confidence": 999, "reasons": ["r"]}
        assert _normalize_output(d)["confidence"] == 100.0

    def test_confidence_clamped_below_0(self):
        d = {"label": "REAL", "confidence": -5, "reasons": ["r"]}
        assert _normalize_output(d)["confidence"] == 0.0

    def test_missing_reasons_gets_default(self):
        d = {"label": "REAL", "confidence": 50}
        result = _normalize_output(d)
        assert len(result["reasons"]) >= 1

    def test_reasons_truncated_to_5(self):
        d = {"label": "REAL", "confidence": 50, "reasons": [f"r{i}" for i in range(10)]}
        assert len(_normalize_output(d)["reasons"]) <= 5

    def test_flags_default_to_empty_list(self):
        d = {"label": "REAL", "confidence": 50, "reasons": ["r"]}
        assert _normalize_output(d)["flags"] == []

    def test_missing_confidence_defaults_to_zero(self):
        d = {"label": "FAKE", "reasons": ["r"]}
        assert _normalize_output(d)["confidence"] == 0.0

    def test_empty_dict_returns_uncertain(self):
        result = _normalize_output({})
        assert result["label"] == "UNCERTAIN"
        assert result["confidence"] == 0.0


# ---------------------------------------------------------------------------
# run_judge (mocked OpenAI)
# ---------------------------------------------------------------------------
class TestRunJudge:
    def _make_mock_client(self, content: str):
        usage = MagicMock()
        usage.prompt_tokens = 100
        usage.completion_tokens = 50
        usage.total_tokens = 150

        choice = MagicMock()
        choice.message.content = content

        response = MagicMock()
        response.choices = [choice]
        response.usage = usage

        client = MagicMock()
        client.chat.completions.create.return_value = response
        return client

    def test_run_judge_returns_parsed_and_meta(self):
        if not SAMPLE_IMAGE.exists():
            pytest.skip("Sample image not present")

        payload = '{"label": "FAKE", "confidence": 80, "reasons": ["inconsistent font"]}'
        client = self._make_mock_client(payload)
        cfg = JudgeConfig(name="judge_1", model="gpt-4o-mini", temperature=0.2, persona="strict")

        parsed, meta = run_judge(client, cfg, image_path=SAMPLE_IMAGE, receipt_id="test-001")

        assert parsed["label"] in ("FAKE", "REAL", "UNCERTAIN")
        assert 0.0 <= parsed["confidence"] <= 100.0
        assert isinstance(parsed["reasons"], list)
        assert "usage" in meta
        assert "input" in meta
        assert "output" in meta

    def test_run_judge_handles_bad_json_gracefully(self):
        if not SAMPLE_IMAGE.exists():
            pytest.skip("Sample image not present")

        client = self._make_mock_client("Sorry, I cannot evaluate this.")
        cfg = JudgeConfig(name="judge_1", model="gpt-4o-mini", temperature=0.2, persona="strict")

        parsed, _ = run_judge(client, cfg, image_path=SAMPLE_IMAGE, receipt_id="test-002")

        # Should fall back to UNCERTAIN, not raise
        assert parsed["label"] == "UNCERTAIN"

    def test_run_judge_meta_contains_receipt_id(self):
        if not SAMPLE_IMAGE.exists():
            pytest.skip("Sample image not present")

        payload = '{"label": "REAL", "confidence": 60, "reasons": ["clean print"]}'
        client = self._make_mock_client(payload)
        cfg = JudgeConfig(name="judge_2", model="gpt-4o", temperature=0.4, persona="balanced")

        _, meta = run_judge(client, cfg, image_path=SAMPLE_IMAGE, receipt_id="my-receipt")

        assert meta["input"]["receipt_id"] == "my-receipt"
        assert meta["input"]["model"] == "gpt-4o"
