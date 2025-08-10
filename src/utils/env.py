from functools import lru_cache

from configs.settings import Settings


@lru_cache(maxsize=1)
def load_settings() -> Settings:
    """Load application settings, returning a singleton instance."""
    return Settings()
