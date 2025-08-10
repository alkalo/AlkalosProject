import logging
from logging.handlers import RotatingFileHandler

from .env import get_logs_dir


def setup_logging(name: str, level: int = logging.INFO) -> None:
    """Configure root logging to a rotating file in ``logs/``.

    Parameters
    ----------
    name:
        Base filename for the log. The log file will be ``logs/<name>.log``.
    level:
        Logging level for the root logger. Defaults to ``logging.INFO``.
    """
    logs_dir = get_logs_dir()
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{name}.log"

    handler = RotatingFileHandler(log_path, maxBytes=1_000_000, backupCount=3)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers = []
    root.addHandler(handler)
