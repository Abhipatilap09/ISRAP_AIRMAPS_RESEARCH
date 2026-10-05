"""
utils.py
--------
Shared utilities: logging setup, cyclical encoding, safe arithmetic.
"""

from __future__ import annotations

import logging
import math
import sys
from pathlib import Path
from typing import Union


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(
    log_file: Union[str, Path] = "logs/pipeline.log",
    level: str = "INFO",
) -> logging.Logger:
    """Configure root logger with file + console handlers (UTF-8 safe)."""
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt=datefmt)

    # File handler – always UTF-8
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    # Console handler – reconfigure stdout to UTF-8 on Windows
    try:
        import io
        utf8_stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
        )
        console_handler = logging.StreamHandler(utf8_stdout)
    except AttributeError:
        # sys.stdout has no .buffer in some environments (e.g. IDLE)
        console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    root_logger.handlers.clear()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    return logging.getLogger("pipeline")


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


# ---------------------------------------------------------------------------
# Cyclical / temporal encoding
# ---------------------------------------------------------------------------

def hour_sin_cos(hour: float) -> tuple[float, float]:
    """Return (sin, cos) encoding of hour-of-day in [0, 23]."""
    angle = 2.0 * math.pi * hour / 24.0
    return math.sin(angle), math.cos(angle)


def dayofyear_sin_cos(doy: float) -> tuple[float, float]:
    """Return (sin, cos) encoding of day-of-year in [1, 365]."""
    angle = 2.0 * math.pi * (doy - 1) / 365.0
    return math.sin(angle), math.cos(angle)


# ---------------------------------------------------------------------------
# Safe arithmetic
# ---------------------------------------------------------------------------

def safe_divide(a: float, b: float, default: float = float("nan")) -> float:
    """Divide a / b; return *default* when b == 0 or b is NaN."""
    try:
        if b == 0 or math.isnan(b):
            return default
        return a / b
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: Union[str, Path]) -> Path:
    """Create directory (and parents) if needed, return Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


