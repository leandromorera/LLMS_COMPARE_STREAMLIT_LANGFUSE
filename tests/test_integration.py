"""
Integration tests — call the real OpenAI API against the two sample receipts.

Run with:
    pytest tests/test_integration.py -v -m integration

Skipped automatically if OPENAI_API_KEY is not set in the environment or .env file.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ---------------------------------------------------------------------------
# Skip the entire module if no API key is available
# ---------------------------------------------------------------------------
pytestmark = pytest.mark.integration

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    pytest.skip("OPENAI_API_KEY not set — skipping integration tests", allow_module_level=True)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
SAMPLE_DIR   = Path(__file__).parent.parent / "data" / "sample"
FAKE_IMAGE   = SAMPLE_DIR / "images" / "X00016469622.png"
REAL_IMAGE   = SAMPLE_DIR / "images" / "X00016469623.png"
FAKE_OCR     = SAMPLE_DIR / "ocr"    / "X00016469622.txt"
REAL_OCR     = SAMPLE_DIR / "ocr"    / "X00016469623.txt"

# Use the cheapest vision-capable model for integration tests to minimise cost.
TEST_MODEL = "gpt-4o-mini"


@pytest.fixture(scope="module")
def client() -> OpenAI:
    return OpenAI(api_key=OPENAI_API_KEY)


@pytest.fixture(scope="module")
def judge_cfg():
    from src.judges import JudgeConfig
    return JudgeConfig(
        name="judge_integration",
        model=TEST_MODEL,
        temperature=0.2,
        persona="strict, skeptical, focuses on forensic inconsistencies",
    )


# ---------------------------------------------------------------------------
# Single-judge schema tests
# ---------------------------------------------------------------------------
class TestSingleJudgeSchema:
    """Verify that a real API call returns a structurally valid response."""

    def _assert_valid_schema(self, parsed: dict) -> None:
        assert parsed["label"] in ("FAKE", "REAL", "UNCERTAIN"), \
            f"label must be FAKE/REAL/UNCERTAIN, got {parsed['label']!r}"
        assert 0.0 <= parsed["confidence"] <= 100.0, \
            f"confidence out of range: {parsed['confidence']}"
        assert isinstance(parsed["reasons"], list) and len(parsed["reasons"]) >= 1, \
            "reasons must be a non-empty list"
        assert all(isinstance(r, str) for r in parsed["reasons"]), \
            "all reasons must be strings"
        assert isinstance(parsed.get("flags", []), list), \
            "flags must be a list"

    def test_fake_receipt_valid_schema(self, client, judge_cfg):
        from src.judges import run_judge
        parsed, meta = run_judge(client, judge_cfg, image_path=FAKE_IMAGE, receipt_id="X00016469622")
        self._assert_valid_schema(parsed)
        assert "usage" in meta
        assert meta["usage"]["input_tokens"] is not None

    def test_real_receipt_valid_schema(self, client, judge_cfg):
        from src.judges import run_judge
        parsed, meta = run_judge(client, judge_cfg, image_path=REAL_IMAGE, receipt_id="X00016469623")
        self._assert_valid_schema(parsed)
        assert "usage" in meta

    def test_fake_receipt_label_is_fake_or_uncertain(self, client, judge_cfg):
        """The FAKE receipt should not be classified as REAL with a strict judge."""
        from src.judges import run_judge
        parsed, _ = run_judge(client, judge_cfg, image_path=FAKE_IMAGE, receipt_id="X00016469622")
        assert parsed["label"] in ("FAKE", "UNCERTAIN"), \
            f"Strict judge classified the known-FAKE receipt as REAL (confidence={parsed['confidence']})"

    def test_real_receipt_label_is_real_or_uncertain(self, client, judge_cfg):
        """The REAL receipt should not be classified as FAKE with a strict judge."""
        from src.judges import run_judge
        parsed, _ = run_judge(client, judge_cfg, image_path=REAL_IMAGE, receipt_id="X00016469623")
        assert parsed["label"] in ("REAL", "UNCERTAIN"), \
            f"Strict judge classified the known-REAL receipt as FAKE (confidence={parsed['confidence']})"


# ---------------------------------------------------------------------------
# Full pipeline test (3 judges + majority vote)
# ---------------------------------------------------------------------------
class TestFullPipeline:
    """Run all 3 judges and aggregate — mirrors what run_one.py / streamlit does."""

    @pytest.fixture(scope="class")
    def three_cfgs(self):
        from src.judges import JudgeConfig
        return [
            JudgeConfig(name="judge_1", model=TEST_MODEL, temperature=0.2,
                        persona="strict, skeptical, focuses on forensic inconsistencies"),
            JudgeConfig(name="judge_2", model=TEST_MODEL, temperature=0.4,
                        persona="balanced, looks for plausible printing/scan artifacts vs tampering"),
            JudgeConfig(name="judge_3", model=TEST_MODEL, temperature=0.7,
                        persona="lenient, assumes real unless clear signs of manipulation"),
        ]

    def _run_pipeline(self, client, cfgs, image_path, receipt_id):
        from src.judges import run_judge
        from src.vote import majority_vote, vote_tally

        labels, outputs = [], []
        for cfg in cfgs:
            parsed, meta = run_judge(client, cfg, image_path=image_path, receipt_id=receipt_id)
            labels.append(parsed["label"])
            outputs.append(parsed)

        final = majority_vote(labels)
        tally = vote_tally(labels)
        return final, tally, outputs

    def test_pipeline_fake_receipt(self, client, three_cfgs):
        final, tally, outputs = self._run_pipeline(
            client, three_cfgs, FAKE_IMAGE, "X00016469622"
        )
        assert final in ("FAKE", "REAL", "UNCERTAIN")
        assert tally["FAKE"] + tally["REAL"] + tally["UNCERTAIN"] == 3
        assert len(outputs) == 3

    def test_pipeline_real_receipt(self, client, three_cfgs):
        final, tally, outputs = self._run_pipeline(
            client, three_cfgs, REAL_IMAGE, "X00016469623"
        )
        assert final in ("FAKE", "REAL", "UNCERTAIN")
        assert tally["FAKE"] + tally["REAL"] + tally["UNCERTAIN"] == 3
        assert len(outputs) == 3

    def test_pipeline_fake_majority_not_real(self, client, three_cfgs):
        """At least 2 of 3 judges should not call the FAKE receipt REAL."""
        final, tally, _ = self._run_pipeline(
            client, three_cfgs, FAKE_IMAGE, "X00016469622"
        )
        assert tally["REAL"] <= 1, \
            f"Too many judges called the FAKE receipt REAL: tally={tally}"

    def test_pipeline_real_majority_not_fake(self, client, three_cfgs):
        """At least 2 of 3 judges should not call the REAL receipt FAKE."""
        final, tally, _ = self._run_pipeline(
            client, three_cfgs, REAL_IMAGE, "X00016469623"
        )
        assert tally["FAKE"] <= 1, \
            f"Too many judges called the REAL receipt FAKE: tally={tally}"
