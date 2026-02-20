"""Shared pytest configuration — adds project root to sys.path."""
import sys
from pathlib import Path

# Ensure the project root is on sys.path so `from src.xxx import` works
# regardless of where pytest is invoked from.
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
