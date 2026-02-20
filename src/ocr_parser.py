"""Extract receipt total amount from OCR text."""
import re
from pathlib import Path


_TOTAL_KEYWORDS_PRIMARY = re.compile(
    r'\b(grand\s+total|amount\s+due|balance\s+due|total\s+due|total)\b',
    re.IGNORECASE
)
_TOTAL_KEYWORDS_FALLBACK = re.compile(r'\b(subtotal)\b', re.IGNORECASE)
_AMOUNT_PATTERN = re.compile(r'\$?\s*(\d{1,6}[.,]\d{2})')


def _search_keyword_lines(lines: list[str], pattern) -> float | None:
    """Search lines for keyword pattern and return first matched amount."""
    for i, line in enumerate(lines):
        if pattern.search(line):
            match = _AMOUNT_PATTERN.search(line)
            if match:
                return _parse_amount(match.group(1))
            if i + 1 < len(lines):
                match = _AMOUNT_PATTERN.search(lines[i + 1])
                if match:
                    return _parse_amount(match.group(1))
    return None


def extract_total_from_text(ocr_text: str) -> float | None:
    """Parse receipt total from OCR text using keyword + next-line search.

    Primary keywords (total, grand total, etc.) are checked first; subtotal
    is used as a fallback so that an explicit 'Total' line takes priority.
    """
    lines = ocr_text.splitlines()
    result = _search_keyword_lines(lines, _TOTAL_KEYWORDS_PRIMARY)
    if result is not None:
        return result
    return _search_keyword_lines(lines, _TOTAL_KEYWORDS_FALLBACK)


def _parse_amount(amount_str: str) -> float:
    """Convert amount string like '12,34' or '12.34' to float."""
    cleaned = amount_str.replace(',', '.').replace(' ', '')
    return float(cleaned)


def load_ocr_text(ocr_path: str) -> str | None:
    """Load OCR text from file, return None if not found."""
    path = Path(ocr_path)
    if not path.exists():
        return None
    return path.read_text(encoding='utf-8', errors='replace')
