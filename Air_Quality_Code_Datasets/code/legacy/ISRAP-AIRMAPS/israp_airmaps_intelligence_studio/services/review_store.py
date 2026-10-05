"""Anomaly review decision store (SQLite)."""
from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from src.config import EXPERIMENTS_DIR

REVIEW_DB = EXPERIMENTS_DIR / "anomaly_reviews.db"

REVIEW_STATUS = [
    "confirm_invalid",
    "genuine_event",
    "manual_review",
    "ignore",
    "convert_to_nan",
    "exclude",
]


def _conn() -> sqlite3.Connection:
    REVIEW_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(REVIEW_DB))
    conn.row_factory = sqlite3.Row
    return conn


def _init() -> None:
    with _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                review_timestamp TEXT,
                station TEXT,
                pollutant TEXT,
                datetime TEXT,
                original_value REAL,
                methods_flagged TEXT,
                anomaly_score REAL,
                review_status TEXT,
                reviewer_note TEXT
            )
        """)


_init()


def save_review(
    station: str,
    pollutant: str,
    dt: str,
    original_value: float,
    methods_flagged: list[str],
    anomaly_score: float,
    review_status: str,
    reviewer_note: str = "",
) -> None:
    with _conn() as c:
        c.execute("""
            INSERT OR REPLACE INTO reviews
            (review_timestamp, station, pollutant, datetime, original_value,
             methods_flagged, anomaly_score, review_status, reviewer_note)
            VALUES (?,?,?,?,?,?,?,?,?)
        """, (
            datetime.utcnow().isoformat(), station, pollutant, str(dt),
            original_value, ",".join(methods_flagged), anomaly_score,
            review_status, reviewer_note,
        ))


def get_reviews(station: str | None = None, pollutant: str | None = None) -> list[dict[str, Any]]:
    sql = "SELECT * FROM reviews WHERE 1=1"
    params: list[Any] = []
    if station:
        sql += " AND station=?"
        params.append(station)
    if pollutant:
        sql += " AND pollutant=?"
        params.append(pollutant)
    sql += " ORDER BY review_timestamp DESC"
    with _conn() as c:
        rows = c.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def count_open_reviews() -> int:
    with _conn() as c:
        row = c.execute("SELECT COUNT(*) FROM reviews WHERE review_status='manual_review'").fetchone()
    return row[0] if row else 0

