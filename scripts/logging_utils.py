from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def configure_logging(log_dir: Path, debug: bool, log_name: str) -> Optional[Path]:
    """Configure console logging and optional debug file logging."""
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG if debug else logging.INFO)

    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if debug else logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    log_file = None
    if debug:
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{log_name}_{timestamp}.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    logging.captureWarnings(True)
    return log_file
