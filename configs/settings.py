from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    coingecko_api_key: Optional[str] = None
    data_dir: Path = Path("./data")
    models_dir: Path = Path("./models")
    logs_dir: Path = Path("./logs")
    symbols: List[str] = Field(default_factory=lambda: ["BTC", "ETH"])
    fiat: str = "USD"

    @field_validator("symbols", mode="before")
    def split_symbols(cls, v: str | List[str]):
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        return v

    @field_validator("data_dir", "models_dir", "logs_dir", mode="before")
    def ensure_directory(cls, v: str | Path) -> Path:
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return path

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
