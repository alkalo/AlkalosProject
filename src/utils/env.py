from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from configs.settings import Settings


@lru_cache()
def get_data_dir() -> Path:
    return Settings().data_dir


@lru_cache()
def get_models_dir() -> Path:
    return Settings().models_dir


@lru_cache()
def get_logs_dir() -> Path:
    return Settings().logs_dir


@lru_cache()
def get_reports_dir() -> Path:
    return Settings().reports_dir


__all__ = [
    "get_data_dir",
    "get_models_dir",
    "get_logs_dir",
    "get_reports_dir",
]

