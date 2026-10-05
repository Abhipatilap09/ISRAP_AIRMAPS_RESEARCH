"""Local SQLite-backed experiment history store."""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from src.config import EXPERIMENTS_DIR

DB_PATH = EXPERIMENTS_DIR / "experiments.db"


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def _init_db() -> None:
    with _conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                id TEXT PRIMARY KEY,
                name TEXT,
                created_at TEXT,
                station TEXT,
                pollutant TEXT,
                method TEXT,
                category TEXT,
                params TEXT,
                metrics TEXT,
                status TEXT,
                runtime_sec REAL,
                output_path TEXT,
                notes TEXT,
                dataset_version TEXT,
                random_seed INTEGER
            )
        """)


_init_db()


def save_experiment(
    station: str,
    pollutant: str,
    method: str,
    category: str,
    params: dict[str, Any],
    metrics: dict[str, Any],
    status: str,
    runtime_sec: float,
    output_path: str = "",
    notes: str = "",
    name: str = "",
    random_seed: int = 42,
    dataset_version: str = "1.0",
) -> str:
    exp_id = str(uuid.uuid4())[:8]
    auto_name = name or f"{station}/{pollutant} — {method}"
    with _conn() as conn:
        conn.execute("""
            INSERT INTO experiments VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            exp_id, auto_name, datetime.utcnow().isoformat(),
            station, pollutant, method, category,
            json.dumps(params), json.dumps(metrics),
            status, runtime_sec, output_path, notes, dataset_version, random_seed,
        ))
    return exp_id


def list_experiments(limit: int = 100) -> list[dict[str, Any]]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM experiments ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_experiment(exp_id: str) -> dict[str, Any] | None:
    with _conn() as conn:
        row = conn.execute("SELECT * FROM experiments WHERE id=?", (exp_id,)).fetchone()
    return dict(row) if row else None


def delete_experiment(exp_id: str) -> None:
    with _conn() as conn:
        conn.execute("DELETE FROM experiments WHERE id=?", (exp_id,))


def update_notes(exp_id: str, notes: str) -> None:
    with _conn() as conn:
        conn.execute("UPDATE experiments SET notes=? WHERE id=?", (notes, exp_id))

