"""Environment helpers for configuration and credentials.

This module centralises access to project directories by reading optional
settings from a ``.env`` file.  If the environment variables are missing,
sensible defaults relative to the repository root are used.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


# Resolve repository root (``src/utils`` -> repo root)
BASE_DIR = Path(__file__).resolve().parents[2]

# Load environment variables from ``.env`` if present
load_dotenv(BASE_DIR / ".env")


def _resolve_dir(var_name: str, default: str) -> Path:
    """Return a directory path from an environment variable.

    Parameters
    ----------
    var_name:
        Name of the environment variable to look up.
    default:
        Fallback relative path when the variable is undefined.

    Returns
    -------
    pathlib.Path
        Absolute path to the requested directory.
    """

    value = os.getenv(var_name, default)
    path = Path(value)
    if not path.is_absolute():
        path = BASE_DIR / path
    return path


@lru_cache()
def get_data_dir() -> Path:
    """Return the directory where datasets are stored."""

    return _resolve_dir("DATA_DIR", "data")


@lru_cache()
def get_models_dir() -> Path:
    """Return the directory containing trained models."""

    return _resolve_dir("MODELS_DIR", "models")


@lru_cache()
def get_logs_dir() -> Path:
    """Return the directory for log files."""

    return _resolve_dir("LOGS_DIR", "logs")


@lru_cache()
def get_reports_dir() -> Path:
    """Return the directory for generated reports."""

    return _resolve_dir("REPORTS_DIR", "reports")


__all__ = [
    "get_data_dir",
    "get_models_dir",
    "get_logs_dir",
    "get_reports_dir",
]


