"""
logger.py
=========
Central logging configuration for the Internship & Hiring Trend Analyzer.

Usage in any module:
    from logger import get_logger
    log = get_logger(__name__)

    log.debug("Granular detail only written to file")
    log.info("Normal progress milestone")
    log.warning("Something unexpected but recoverable")
    log.error("Something failed", exc_info=True)
    log.critical("Full stop failure")

Output:
    Terminal  — INFO and above (clean progress view)
    File      — DEBUG and above (full detail for debugging)

Log files:
    logs/app.log        current log file
    logs/app.log.1      previous (auto-rotated at 5 MB)
    logs/app.log.2      oldest backup
"""

import logging
import os
from logging.handlers import RotatingFileHandler

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LOG_DIR          = "logs"
LOG_FILE         = os.path.join(LOG_DIR, "app.log")
LOG_FORMAT       = "%(asctime)s | %(levelname)-8s | %(module)-22s | %(message)s"
LOG_DATE_FORMAT  = "%Y-%m-%d %H:%M:%S"
MAX_BYTES        = 5 * 1024 * 1024   # 5 MB per file
BACKUP_COUNT     = 3                  # Keep 3 rotated files


def get_logger(name: str = "internship_analyzer") -> logging.Logger:
    """
    Return a configured logger that writes to both terminal and log file.

    Calling this multiple times with the same name returns the same logger
    instance (Python logging is a singleton per name) — handlers are only
    added once to avoid duplicate log lines.

    Args:
        name: Logger name, use __name__ in each module so log lines
              show the exact module they came from.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)

    # If handlers already attached (called twice with same name), return as-is
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    # --- Terminal handler: INFO and above ---
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # --- File handler: DEBUG and above, rotates at 5 MB ---
    os.makedirs(LOG_DIR, exist_ok=True)
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=MAX_BYTES,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger