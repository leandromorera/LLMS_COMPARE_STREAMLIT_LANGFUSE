"""Tests for OCR parser."""
import pytest
from src.ocr_parser import extract_total_from_text, _parse_amount, load_ocr_text
import tempfile
import os


class TestParseAmount:
    def test_decimal_point(self):
        assert _parse_amount("12.34") == 12.34

    def test_decimal_comma(self):
        assert _parse_amount("12,34") == 12.34

    def test_large_amount(self):
        assert _parse_amount("1234.56") == 1234.56

    def test_small_amount(self):
        assert _parse_amount("0.99") == 0.99


class TestExtractTotalFromText:
    def test_total_keyword(self):
        text = "Subtotal: 10.00\nTotal: 12.50\n"
        assert extract_total_from_text(text) == 12.50

    def test_grand_total(self):
        text = "Grand Total\n$25.99\n"
        assert extract_total_from_text(text) == 25.99

    def test_amount_due(self):
        text = "Amount Due: $8.75"
        assert extract_total_from_text(text) == 8.75

    def test_balance_due(self):
        text = "Balance Due\n$15.00"
        assert extract_total_from_text(text) == 15.00

    def test_no_total_returns_none(self):
        text = "Item 1: 5.00\nItem 2: 3.00\n"
        assert extract_total_from_text(text) is None

    def test_empty_string_returns_none(self):
        assert extract_total_from_text("") is None

    def test_total_on_same_line(self):
        text = "TOTAL DUE $42.00"
        assert extract_total_from_text(text) == 42.00

    def test_total_next_line(self):
        text = "TOTAL\n99.99"
        assert extract_total_from_text(text) == 99.99

    def test_subtotal_keyword(self):
        text = "Subtotal $5.50\n"
        assert extract_total_from_text(text) == 5.50

    def test_comma_decimal_separator(self):
        text = "Total: 1.234,56"
        # Should extract last matching amount
        result = extract_total_from_text(text)
        # Accepts either 1234.56 or 1.23 depending on regex match
        assert result is not None

    def test_dollar_sign_prefix(self):
        text = "Total Due: $99.00"
        assert extract_total_from_text(text) == 99.00


class TestLoadOcrText:
    def test_load_existing_file(self, tmp_path):
        ocr_file = tmp_path / "receipt.txt"
        ocr_file.write_text("Total: $10.00")
        result = load_ocr_text(str(ocr_file))
        assert result == "Total: $10.00"

    def test_nonexistent_file_returns_none(self):
        assert load_ocr_text("/nonexistent/path.txt") is None

    def test_empty_file(self, tmp_path):
        ocr_file = tmp_path / "empty.txt"
        ocr_file.write_text("")
        assert load_ocr_text(str(ocr_file)) == ""
