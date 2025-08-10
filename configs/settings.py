

from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    data_dir: Path = Path("./data")
    models_dir: Path = Path("./models")
    logs_dir: Path = Path("./logs")
    symbols: List[str] = ["BTC", "ETH"]
    fiat: str = "USD"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    @field_validator("data_dir", "models_dir", "logs_dir", mode="before")
    @classmethod
    def create_dir(cls, v: str) -> Path:
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return path


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

