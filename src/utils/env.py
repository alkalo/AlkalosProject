"""Helpers for accessing configuration and project directories.

This module provides a thin wrapper around :mod:`configs.settings` so that
other parts of the codebase can easily access common paths (and later
credentials) without needing to know about the underlying settings
implementation.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from configs.settings import Settings, get_settings


@lru_cache()
def _settings() -> Settings:
    """Return the cached application settings."""

    return get_settings()


def get_data_dir() -> Path:
    """Directory where datasets are stored."""

    return _settings().data_dir


def get_models_dir() -> Path:
    """Directory containing trained models."""

    return _settings().models_dir


def get_logs_dir() -> Path:
    """Directory used for log files."""

    return _settings().logs_dir


def get_reports_dir() -> Path:
    """Directory for generated reports."""

    return _settings().reports_dir


__all__ = [
    "get_data_dir",
    "get_models_dir",
    "get_logs_dir",
    "get_reports_dir",
]

