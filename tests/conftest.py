"""Pytest configuration for ensuring repository root on sys.path."""
from __future__ import annotations

import sys
from pathlib import Path

# Add the project root to ``sys.path`` so tests can import local packages
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
